# -*- coding: utf-8 -*-
"""
Scope Gate Redesign, Full Validation — Layer A (self-declared field) + Layer B (backstop)
================================================================================================
Semantic embeddings were tried and rejected as the Scope Gate fix (2 models x 2
description styles, all F1~0.31 with ~48-50/50 false refusals on legitimate RQ3
prompts -- experiments/scope_gate_embedding_calibration.py). The adopted fix instead
adds a structured self-declaration field (`is_customer_facing_feature`) directly into
the SAME `_llm_full_parse` extraction call the LLM already performs, removing the old
instruction that forced it to always pick an archetype with no way to say "none of
these fit" -- combined via OR with the existing, independent keyword-overlap backstop
(neither trusted alone, per LLM-Modulo).

This script validates BOTH directions in one live-LLM pass, at full scale (not a
sample), against the exact new is_out_of_scope logic in ParserAgent.parse():
  - All 11 Scope-Gate adversarial prompts (ground truth: SHOULD refuse) --
    data/processed/scm_results/g6_scope_gate_adversarial.csv (unique prompts).
  - All 50 RQ3 legitimate prompts (ground truth: should NOT refuse) --
    data/benchmark/parser_benchmark_prompts.json.

For each prompt we call `_llm_full_parse` directly (bypassing the fast-path
deliberately -- the RQ3 legitimate prompts include several that WOULD fast-path on
keyword match alone, but we need the LLM's is_customer_facing_feature judgment
specifically, which only the LLM branch produces), then replay the exact same
combined-OR decision `parse()` uses.
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
from agents.parser_agent import ParserAgent, _compute_similarity
from request_router import classify_request

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
PROMPTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'parser_benchmark_prompts.json')
ADVERSARIAL_FILE = os.path.join(OUTPUT_DIR, 'g6_scope_gate_adversarial.csv')
GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')


def get_live_llm():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set.")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.2)


def load_test_set():
    """Returns list of (prompt_id, requirement, true_should_refuse, source)."""
    rows = []
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        rq3 = json.load(f)
    for p in rq3:
        rows.append((p['id'], p['requirement'], False, 'RQ3_legitimate'))

    adv_df = pd.read_csv(ADVERSARIAL_FILE)
    adv_unique = adv_df.drop_duplicates(subset=['prompt_id'])
    for _, r in adv_unique.iterrows():
        rows.append((r['prompt_id'], r['requirement'], True, 'Adversarial_should_refuse'))
    return rows


def main():
    arch = ArchitectureAgent(GRAPH_PATH)
    llm = get_live_llm()
    parser = ParserAgent(llm=llm, arch_agent=arch)

    test_set = load_test_set()
    print(f"Loaded {len(test_set)} prompts ({sum(1 for r in test_set if not r[2])} legitimate, "
          f"{sum(1 for r in test_set if r[2])} adversarial).")

    fieldnames = ['prompt_id', 'source', 'requirement', 'true_should_refuse',
                  'rb_similarity', 'llm_request_type', 'llm_pick_similarity',
                  'keyword_similarity_score', 'keyword_signal', 'llm_is_customer_facing',
                  'scope_signal', 'predicted_refuse', 'correct']
    csv_path = os.path.join(OUTPUT_DIR, 'g6_scope_gate_layerA_validation.csv')
    f_out = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    rows = []
    for prompt_id, req, true_should_refuse, source in test_set:
        rb_request_type = classify_request(req)
        rb_similarity = _compute_similarity(req, rb_request_type)

        rt, core, raw_adj, reasoning, conf, raw_inj, is_customer_facing = \
            parser._llm_full_parse(req, rb_request_type)
        llm_pick_similarity = _compute_similarity(req, rt)
        similarity_score = max(rb_similarity, llm_pick_similarity)

        keyword_signal = (similarity_score == 0.0)
        scope_signal = (is_customer_facing == False)
        predicted_refuse = keyword_signal or scope_signal
        correct = (predicted_refuse == true_should_refuse)

        row = {
            'prompt_id': prompt_id, 'source': source, 'requirement': req,
            'true_should_refuse': true_should_refuse,
            'rb_similarity': round(rb_similarity, 4), 'llm_request_type': rt,
            'llm_pick_similarity': round(llm_pick_similarity, 4),
            'keyword_similarity_score': round(similarity_score, 4),
            'keyword_signal': keyword_signal, 'llm_is_customer_facing': is_customer_facing,
            'scope_signal': scope_signal, 'predicted_refuse': predicted_refuse,
            'correct': correct,
        }
        rows.append(row)
        writer.writerow(row)
        f_out.flush()
        mark = 'OK' if correct else '*** WRONG ***'
        print(f"  [{source}] {prompt_id}: true_refuse={true_should_refuse} "
              f"pred_refuse={predicted_refuse} (kw={keyword_signal}, scope={scope_signal}, "
              f"is_cf={is_customer_facing}) {mark}")
        time.sleep(4.5)

    f_out.close()
    df = pd.DataFrame(rows)
    print(f"\n[OK] Saved: {csv_path} ({len(df)} rows)")

    tp = ((df.predicted_refuse) & (df.true_should_refuse)).sum()
    fn = ((~df.predicted_refuse) & (df.true_should_refuse)).sum()   # evasions remaining
    fp = ((df.predicted_refuse) & (~df.true_should_refuse)).sum()   # false refusals on legit
    tn = ((~df.predicted_refuse) & (~df.true_should_refuse)).sum()

    print(f"\n[Confusion matrix]")
    print(f"  True refuse, predicted refuse   (TP, correct)        : {tp}")
    print(f"  True refuse, predicted allow     (FN, evasion)       : {fn}")
    print(f"  True allow,  predicted refuse    (FP, false-refusal) : {fp}")
    print(f"  True allow,  predicted allow     (TN, correct)       : {tn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    print(f"\n  Refusal precision: {precision:.3f} | Refusal recall: {recall:.3f}")
    print(f"  Overall accuracy: {df['correct'].mean()*100:.1f}%")

    print("\n[Signal contribution] how many refusals came from each signal alone:")
    kw_only = ((df.keyword_signal) & (~df.scope_signal) & (df.predicted_refuse)).sum()
    scope_only = ((~df.keyword_signal) & (df.scope_signal) & (df.predicted_refuse)).sum()
    both = ((df.keyword_signal) & (df.scope_signal)).sum()
    print(f"  Keyword-signal only: {kw_only} | Scope-signal (new) only: {scope_only} | Both: {both}")


if __name__ == '__main__':
    main()
