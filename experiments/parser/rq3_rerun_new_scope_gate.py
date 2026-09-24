# -*- coding: utf-8 -*-
"""
RQ3 partial re-run on the new (sole) conformal Scope Gate architecture
========================================================================
Full RQ3 is 50 prompts x 3 repeats for the Guarded configuration alone
(150 live calls) -- infeasible today under the free-tier daily quota
(20 requests/day, already exhausted once this session). This script runs a
SINGLE pass (no repeats) over a representative subset of RQ3's own 50-prompt
set (data/benchmark/parser_benchmark_prompts.json), stratified across all
four categories, through the CURRENT ParserAgent (conformal two-threshold
Scope Gate -- the OR-rule no longer exists in the codebase).

Unlike the original parser_benchmark_suite.py's GuardedHybridRunner, which
only captures (injection_service, injection_delta_pct, core_services,
elapsed, llm_was_called) and has NO visibility into is_out_of_scope /
needs_human_review at all, this script explicitly records those three
fields -- because the exact open question this re-run exists to answer is
whether NEEDS_HUMAN_REVIEW appears often on real RQ3-style prompts (a
question the original harness cannot even ask).

Computes PBVR/SHR/GMR/Anchor-MAE identically to parser_benchmark_suite.py
for direct comparability with Table rq3's Guarded row, PLUS the outcome
distribution (pass / refuse / review) the original harness never tracked.

Uses the real GOOGLE_API_KEY-configured LLM. Respects the same 4.5s
inter-call spacing as the original harness (per-minute rate limit) --
the binding constraint today is the ~20/day quota, which caps this run's
sample size, not the per-minute one.
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.architecture_agent import ArchitectureAgent
from agents.parser_agent import ParserAgent, KNOWN_SERVICES

GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')
PROMPTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'parser_benchmark_prompts.json')

# Stratified sample sized to fit today's remaining daily quota (~18 calls),
# proportionally thinner than RQ3's own 20/10/10/10 split but covering all
# four categories rather than concentrating on one.
SAMPLE_PER_CATEGORY = {
    'In_Distribution': 6,
    'Complex_MultiHop': 4,
    'Subtle_ReadOnly': 4,
    'Adversarial_Stress': 4,
}


def main():
    print("=" * 80)
    print("  RQ3 PARTIAL RE-RUN -- current (sole) conformal Scope Gate architecture")
    print("=" * 80)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not set.")
        return None

    with open(PROMPTS_FILE, encoding='utf-8') as f:
        all_prompts = json.load(f)

    sample = []
    for cat, n in SAMPLE_PER_CATEGORY.items():
        cat_prompts = [p for p in all_prompts if p['category'] == cat]
        sample.extend(cat_prompts[:n])
    print(f"[Setup] Sampled {len(sample)} of {len(all_prompts)} RQ3 prompts "
          f"({', '.join(f'{k}={v}' for k,v in SAMPLE_PER_CATEGORY.items())})")

    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.2)  # matches paper_draft.tex's stated RQ3 backend
    arch = ArchitectureAgent(GRAPH_PATH)
    parser = ParserAgent(llm=llm, arch_agent=arch)

    rows = []
    for i, p in enumerate(sample):
        t0 = time.time()
        res = parser.parse(p['requirement'])
        elapsed = (time.time() - t0) * 1000

        pbv = (res.injection_delta_pct < 5.0) or (res.injection_delta_pct > 50.0)
        hallucinated = [s for s in res.core_services if s not in KNOWN_SERVICES]
        sh = len(hallucinated) > 0
        gm = (res.injection_service != p['expected_gateway'])
        ae = abs(res.injection_delta_pct - p['expected_anchor'])

        rows.append({
            'prompt_id': p['id'], 'category': p['category'],
            'is_out_of_scope': res.is_out_of_scope, 'needs_human_review': res.needs_human_review,
            'confidence': res.confidence, 'scope_gate_score': res.scope_gate_score,
            'physical_boundary_violation': pbv, 'service_hallucination': sh,
            'gateway_misdirection': gm, 'abs_error': ae,
            'injection_delta_pct': res.injection_delta_pct, 'expected_anchor': p['expected_anchor'],
            'latency_ms': elapsed, 'requirement': p['requirement'],
        })
        print(f"[{i+1:2d}/{len(sample)}][{p['category']:20s}] conf={res.confidence:20s} "
              f"score={res.scope_gate_score:.3f} refuse={res.is_out_of_scope} "
              f"review={res.needs_human_review} delta={res.injection_delta_pct:.1f}% "
              f"(exp={p['expected_anchor']}%) PBV={pbv} SH={sh} GM={gm}")

        if i < len(sample) - 1:
            time.sleep(4.5)  # respect per-minute rate limit between live calls

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results', 'rq3_rerun_new_scope_gate.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

    n = len(df)
    print("\n" + "=" * 80)
    print(f"[RESULT] n={n} (partial sample, NOT the full 50-prompt x 3-repeat RQ3 protocol)")
    print(f"  Outcome distribution: PASS={((~df['is_out_of_scope']) & (~df['needs_human_review'])).sum()}, "
          f"REFUSED={df['is_out_of_scope'].sum()}, NEEDS_HUMAN_REVIEW={df['needs_human_review'].sum()}")
    print(f"  PBVR: {100*df['physical_boundary_violation'].mean():.1f}%")
    print(f"  SHR:  {100*df['service_hallucination'].mean():.1f}%")
    print(f"  GMR:  {100*df['gateway_misdirection'].mean():.1f}%")
    print(f"  Anchor MAE: {df['abs_error'].mean():.1f}%")
    print(f"\n[Reference] Table rq3's Guarded row (full 50x3 protocol, OLD OR-rule): "
          f"PBVR=0.0%, SHR=0.0%, GMR=0.0%, Anchor MAE=2.2%")
    print(f"\n[OK] Saved: {csv_path}")
    print("=" * 80)
    return df


if __name__ == '__main__':
    main()
