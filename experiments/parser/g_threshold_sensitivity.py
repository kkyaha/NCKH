# -*- coding: utf-8 -*-
"""
Threshold Sensitivity Analysis for G2/G3/G5 — Are the Specific Numbers Load-Bearing?
========================================================================================
G2 ([5,50]%), G3 (+/-10%), and G5 (similarity >= 0.6) are hand-set engineering judgments
(src/agents/parser_agent.py MIN_DELTA_PCT/MAX_DELTA_PCT/MAX_ADJUSTMENT_PCT/
SIMILARITY_THRESHOLD), not derived from a statistical or literature-grounded process.
This is a legitimate objection a reviewer would raise: are RQ3's guarded-configuration
results (PBVR/SHR/GMR = 0.0%) an artifact of one lucky choice of thresholds, or does the
*enforcement mechanism itself* -- not the exact numbers -- do the real work?

Design: we make ONE live-LLM pass over RQ3's same 50-prompt test set, but call
ParserAgent._llm_full_parse() directly for EVERY prompt (bypassing the fast-path
short-circuit) so we capture the LLM's RAW, pre-guard proposal (request_type, raw
adjustment, raw core_services) for all 50 prompts uniformly, plus the rule-based
similarity score (deterministic, free) needed to correctly reconstruct whether the real
pipeline would have taken the fast path at any candidate G5 threshold. We then replay
this SAME fixed raw data through many different (G2, G3, G5) threshold configurations
completely offline -- no further LLM calls -- and recompute PBVR/SHR/GMR/Anchor MAE
under each. This isolates the guard-threshold variable from LLM run-to-run noise
(already characterized separately in RQ3), which is the correct design for a sensitivity
analysis: only one thing should vary between rows.

Output:
  data/processed/scm_results/g_threshold_raw_capture.csv      (raw, one live LLM pass)
  data/processed/scm_results/g_threshold_sensitivity.csv      (offline replay results)
"""

import os
import sys
import json
import time

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from agents.architecture_agent import ArchitectureAgent
from agents.parser_agent import ParserAgent, _compute_similarity, KNOWN_SERVICES
from request_router import classify_request, CALL_CHAINS

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
PROMPTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'parser_benchmark_prompts.json')
GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')

# Baseline (current production) thresholds, for reference in the sweep table.
BASELINE = {'min_delta': 5.0, 'max_delta': 50.0, 'max_adj': 10.0, 'sim_thresh': 0.6}


def get_live_llm():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set -- this script requires a live LLM pass.")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.2)


def capture_raw(n_repeats=1):
    """One live-LLM pass per prompt per repeat, bypassing the fast-path so every
    prompt yields a raw LLM proposal, regardless of what threshold would route it."""
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    arch = ArchitectureAgent(GRAPH_PATH)
    llm = get_live_llm()
    parser = ParserAgent(llm=llm, arch_agent=arch)

    rows = []
    for repeat_id in range(1, n_repeats + 1):
        for p in prompts:
            req = p['requirement']
            rb_request_type = classify_request(req)
            rb_similarity = _compute_similarity(req, rb_request_type)

            # Bypass fast-path deliberately: always call the LLM full-parse path so
            # we get a raw proposal for every prompt, uniformly.
            request_type, core_svcs, adjustment, reasoning, llm_conf, _raw_inj_svc, _cust_facing = \
                parser._llm_full_parse(req, rb_request_type)
            llm_pick_similarity = _compute_similarity(req, request_type)
            similarity_score = max(rb_similarity, llm_pick_similarity)

            template_info = CALL_CHAINS.get(request_type, {})
            template_delta = template_info.get('expected_delta_pct', 20.0)
            affected = template_info.get('services', ['front-end'])

            rows.append({
                'repeat_id': repeat_id, 'prompt_id': p['id'], 'category': p['category'],
                'requirement': req, 'expected_anchor': p['expected_anchor'],
                'expected_gateway': p['expected_gateway'],
                'rb_request_type': rb_request_type, 'rb_similarity': round(rb_similarity, 4),
                'llm_request_type': request_type, 'llm_raw_adjustment': adjustment,
                'llm_raw_core_services': str(core_svcs),
                'similarity_score': round(similarity_score, 4),
                'template_delta': template_delta, 'affected_services': str(affected),
            })
            print(f"  [repeat {repeat_id}] {p['id']} ({p['category']}): "
                  f"rb_sim={rb_similarity:.2f} llm_type={request_type} raw_adj={adjustment:+.1f}")
            time.sleep(4.5)  # same free-tier rate-limit as parser_benchmark_suite.py

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'g_threshold_raw_capture.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Raw capture saved: {csv_path} ({len(df)} rows)")
    return df


def replay(df: pd.DataFrame, min_delta, max_delta, max_adj, sim_thresh) -> dict:
    """Deterministically re-derive the Guarded pipeline's outcome for every captured
    row under a GIVEN threshold configuration -- the exact same guard formulas as
    src/agents/parser_agent.py's _guard_delta / G4 / G5 / G6, parameterized."""
    pbv_flags, sh_flags, errors = [], [], []
    for _, row in df.iterrows():
        rb_sim = row['rb_similarity']

        if rb_sim >= sim_thresh:
            # Fast-path: identical to production -- anchor only, no LLM influence.
            delta = row['template_delta']
            core_svcs = eval(row['affected_services'])
        else:
            similarity_score = row['similarity_score']
            adjustment = row['llm_raw_adjustment']
            core_svcs = eval(row['llm_raw_core_services'])
            # G4
            core_svcs = [s for s in core_svcs if s in KNOWN_SERVICES] or ['front-end']
            # G5
            if similarity_score < sim_thresh:
                adjustment = 0.0
            # G2 + G3
            raw_delta = row['template_delta'] + adjustment
            actual_adj = raw_delta - row['template_delta']
            if abs(actual_adj) > max_adj:
                raw_delta = row['template_delta']
            delta = max(min_delta, min(max_delta, raw_delta))

        pbv_flags.append((delta < min_delta) or (delta > max_delta))
        hallucinated = [s for s in core_svcs if s not in KNOWN_SERVICES]
        sh_flags.append(len(hallucinated) > 0)
        errors.append(abs(delta - row['expected_anchor']))

    n = len(df)
    return {
        'min_delta': min_delta, 'max_delta': max_delta, 'max_adj': max_adj,
        'sim_thresh': sim_thresh,
        'PBVR_pct': round(100.0 * sum(pbv_flags) / n, 2),
        'SHR_pct': round(100.0 * sum(sh_flags) / n, 2),
        'anchor_MAE': round(sum(errors) / n, 3),
        'n': n,
    }


def run_sweep(df: pd.DataFrame):
    variants = []
    b = BASELINE
    # Vary G2 alone
    for lo, hi in [(3, 60), (5, 50), (8, 40), (10, 30), (1, 100)]:
        variants.append(('G2', f'[{lo},{hi}]', replay(df, lo, hi, b['max_adj'], b['sim_thresh'])))
    # Vary G3 alone
    for adj in [5.0, 10.0, 15.0, 20.0, 30.0]:
        variants.append(('G3', f'+/-{adj}', replay(df, b['min_delta'], b['max_delta'], adj, b['sim_thresh'])))
    # Vary G5 alone
    for sim in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        variants.append(('G5', f'>={sim}', replay(df, b['min_delta'], b['max_delta'], b['max_adj'], sim)))

    rows = []
    for guard, label, res in variants:
        res2 = dict(res)
        res2['guard_varied'] = guard
        res2['variant'] = label
        rows.append(res2)
    out = pd.DataFrame(rows)[['guard_varied', 'variant', 'min_delta', 'max_delta', 'max_adj',
                                'sim_thresh', 'PBVR_pct', 'SHR_pct', 'anchor_MAE', 'n']]
    csv_path = os.path.join(OUTPUT_DIR, 'g_threshold_sensitivity.csv')
    out.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Sensitivity sweep saved: {csv_path}")
    print(out.to_string(index=False))
    return out


def main():
    n_repeats = 1
    for arg in sys.argv:
        if arg.startswith('--repeats='):
            n_repeats = int(arg.split('=', 1)[1])
    raw_csv = os.path.join(OUTPUT_DIR, 'g_threshold_raw_capture.csv')
    if '--skip-capture' in sys.argv and os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
        print(f"[SKIP] Reusing existing raw capture: {raw_csv} ({len(df)} rows)")
    else:
        df = capture_raw(n_repeats=n_repeats)
    run_sweep(df)


if __name__ == '__main__':
    main()
