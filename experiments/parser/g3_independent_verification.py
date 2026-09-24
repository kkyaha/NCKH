# -*- coding: utf-8 -*-
"""
G3 Independent Verification — Does the Clamp Do Real Work Without Prompt Self-Constraint?
============================================================================================
experiments/g_threshold_sensitivity.py found that G3's specific bound (+/-10%) never
triggered across a full threshold sweep, because the OLD Guarded-pipeline prompt
(src/agents/parser_agent.py::_llm_full_parse) explicitly told the LLM "KHONG duoc dat
adjustment ngoai [-10, 10]" -- the guard's own bound, restated as an instruction. That
makes G3's empirical "0% violation" evidence circular: it never had to prove it works
independently of LLM compliance, which contradicts the LLM-Modulo premise the paper
already cites (guards must hold regardless of whether the LLM hallucinates or complies).

Fix applied: the prompt instruction was removed (parser_agent.py, `_llm_full_parse`) --
the LLM now proposes "adjustment" as any real number reflecting how unusual it judges
the requirement, with NO stated range. G3 (`_guard_delta`'s abs(actual_adj) > 10 check)
is now the ONLY thing standing between an unconstrained LLM proposal and the final
delta. This script re-runs ONLY the Guarded_Hybrid_Parser configuration (the one
affected by this prompt change; B0/B1/B2 use separate prompts/parsers untouched by
this edit) across the same RQ3 test set (50 prompts x 3 repeats), and reports whether
PBVR/SHR/GMR/Anchor MAE are still consistent with the previously-reported Guarded
numbers, plus how often G3's clamp now ACTUALLY fires (previously: never).

Output: data/processed/scm_results/g3_independent_verification.csv
"""

import os
import sys
import csv
import json
import time

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from agents.architecture_agent import ArchitectureAgent
from agents.parser_agent import ParserAgent, KNOWN_SERVICES

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
PROMPTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'parser_benchmark_prompts.json')
GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')


def get_live_llm():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set -- this script requires a live LLM pass.")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.2)


def main(n_repeats=3):
    for arg in sys.argv:
        if arg.startswith('--repeats='):
            n_repeats = int(arg.split('=', 1)[1])

    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    arch = ArchitectureAgent(GRAPH_PATH)
    llm = get_live_llm()

    # Write incrementally (row-by-row, flushed) so a mid-run quota exhaustion
    # (hit once already developing this script) never loses completed rows.
    fieldnames = ['repeat_id', 'prompt_id', 'category', 'expected_anchor', 'pred_delta',
                  'abs_error', 'physical_boundary_violation', 'service_hallucination',
                  'gateway_misdirection', 'g3_clamp_fired', 'confidence', 'llm_was_called',
                  'latency_ms', 'reasoning']
    csv_path = os.path.join(OUTPUT_DIR, 'g3_independent_verification.csv')
    f_out = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    rows = []
    for repeat_id in range(1, n_repeats + 1):
        parser = ParserAgent(llm=llm, arch_agent=arch)  # fresh instance per repeat
        for p in prompts:
            req = p['requirement']
            t0 = time.time()
            res = parser.parse(req)
            elapsed = (time.time() - t0) * 1000

            delta = res.injection_delta_pct
            pbv = (delta < 5.0) or (delta > 50.0)
            hallucinated = [s for s in res.core_services if s not in KNOWN_SERVICES]
            sh = len(hallucinated) > 0
            gm = (res.injection_service != p['expected_gateway'])
            ae = abs(delta - p['expected_anchor'])
            g3_fired = "CLAMPED" in res.reasoning

            row = {
                'repeat_id': repeat_id, 'prompt_id': p['id'], 'category': p['category'],
                'expected_anchor': p['expected_anchor'], 'pred_delta': delta,
                'abs_error': ae, 'physical_boundary_violation': pbv,
                'service_hallucination': sh, 'gateway_misdirection': gm,
                'g3_clamp_fired': g3_fired, 'confidence': res.confidence,
                'llm_was_called': res.llm_was_called, 'latency_ms': elapsed,
                'reasoning': res.reasoning,
            }
            rows.append(row)
            writer.writerow(row)
            f_out.flush()
            print(f"  [repeat {repeat_id}] {p['id']}: delta={delta} PBV={pbv} SH={sh} "
                  f"GM={gm} G3_fired={g3_fired}")
            if res.llm_was_called:
                time.sleep(4.5)

    f_out.close()
    df = pd.DataFrame(rows)
    print(f"\n[OK] Saved: {csv_path} ({len(df)} rows)")

    print("\n[Per-repeat rates]")
    for r in sorted(df['repeat_id'].unique()):
        sub = df[df['repeat_id'] == r]
        print(f"  repeat {r}: PBVR={100*sub['physical_boundary_violation'].mean():.1f}% "
              f"SHR={100*sub['service_hallucination'].mean():.1f}% "
              f"GMR={100*sub['gateway_misdirection'].mean():.1f}% "
              f"AnchorMAE={sub['abs_error'].mean():.2f} "
              f"G3_fire_rate={100*sub['g3_clamp_fired'].mean():.1f}%")

    print(f"\n[Overall] PBVR={100*df['physical_boundary_violation'].mean():.2f}% "
          f"SHR={100*df['service_hallucination'].mean():.2f}% "
          f"GMR={100*df['gateway_misdirection'].mean():.2f}% "
          f"AnchorMAE mean={df['abs_error'].mean():.3f} std={df.groupby('repeat_id')['abs_error'].mean().std():.3f}")
    print(f"[Overall] G3 clamp fire rate: {100*df['g3_clamp_fired'].mean():.2f}% "
          f"({df['g3_clamp_fired'].sum()}/{len(df)} rows)")
    if df['g3_clamp_fired'].any():
        print("\nSample rows where G3 fired:")
        print(df[df['g3_clamp_fired']][['prompt_id', 'category', 'pred_delta', 'reasoning']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
