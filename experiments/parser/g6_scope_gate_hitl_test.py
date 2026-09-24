# -*- coding: utf-8 -*-
"""
Scope Gate HITL Escalation — Full Validation
==================================================================================
The Layer A self-declaration fix (experiments/g6_scope_gate_layerA_test.py) closed
the 100% evasion vulnerability but produced 5 new false-refusals on legitimate-but-
passive RQ3 content (REQ-30 automatic fraud-detection scan, REQ-32/34/37/38 static
FAQ/ToS/bilingual-UI/footer pages). A quick check ruled out "the two signals disagree"
as a clean escalation trigger (9/10 correctly-refused adversarial prompts ALSO show
only one signal firing, same pattern as the false positives). The adopted design
instead uses a confidence field on the SAME self-declaration (is_customer_facing_
confidence), giving three outcomes instead of two: confident refuse, confident allow,
and a new NEEDS_HUMAN_REVIEW state for the LLM's own self-reported uncertainty or for
disagreement between the keyword backstop and a CONFIDENT self-declaration -- this
targets the boundary cases directly rather than guessing either direction. This
applies to Layer 1 (Scope Gate) only; Layer 2 (Entity Grounding) and Layer 3
(Bounded Projection) remain fully automatic deterministic checks with no ambiguity
to route to a human.

Full-scale validation (not a sample): all 11 Scope-Gate adversarial prompts + all 50
RQ3 legitimate prompts, calling ParserAgent.parse() itself (not the internal
_llm_full_parse) so the exact production decision logic (hard_refuse / needs_human_
review / auto-allow) is what gets measured, not a re-implementation of it.
"""

import os
import sys
import csv
import json
import time

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from agents.architecture_agent import ArchitectureAgent
from agents.parser_agent import ParserAgent

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
PROMPTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'parser_benchmark_prompts.json')
ADVERSARIAL_FILE = os.path.join(OUTPUT_DIR, 'g6_scope_gate_adversarial.csv')
GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')

# The 5 prompts the Layer-A-only design wrongly hard-refused (real false positives,
# excluding the 7 Adversarial_Stress prompts that arguably deserved refusal anyway).
KNOWN_BORDERLINE = {'REQ-30', 'REQ-32', 'REQ-34', 'REQ-37', 'REQ-38'}


def get_live_llm():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set.")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.2)


def load_test_set():
    rows = []
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        rq3 = json.load(f)
    for p in rq3:
        category = 'RQ3_borderline' if p['id'] in KNOWN_BORDERLINE else 'RQ3_clear_legitimate'
        rows.append((p['id'], p['requirement'], category))

    adv_df = pd.read_csv(ADVERSARIAL_FILE)
    adv_unique = adv_df.drop_duplicates(subset=['prompt_id'])
    for _, r in adv_unique.iterrows():
        rows.append((r['prompt_id'], r['requirement'], 'Adversarial_should_refuse'))
    return rows


def main():
    arch = ArchitectureAgent(GRAPH_PATH)
    llm = get_live_llm()
    parser = ParserAgent(llm=llm, arch_agent=arch)

    test_set = load_test_set()
    print(f"Loaded {len(test_set)} prompts.")

    fieldnames = ['prompt_id', 'category', 'requirement', 'request_type',
                  'is_out_of_scope', 'needs_human_review', 'confidence', 'reasoning']
    csv_path = os.path.join(OUTPUT_DIR, 'g6_scope_gate_hitl_validation.csv')
    f_out = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    rows = []
    for prompt_id, req, category in test_set:
        res = parser.parse(req)
        row = {
            'prompt_id': prompt_id, 'category': category, 'requirement': req,
            'request_type': res.request_type, 'is_out_of_scope': res.is_out_of_scope,
            'needs_human_review': res.needs_human_review, 'confidence': res.confidence,
            'reasoning': res.reasoning,
        }
        rows.append(row)
        writer.writerow(row)
        f_out.flush()
        outcome = 'HARD_REFUSE' if res.is_out_of_scope else ('NEEDS_REVIEW' if res.needs_human_review else 'AUTO_ALLOW')
        print(f"  [{category}] {prompt_id}: outcome={outcome}")
        time.sleep(4.5)

    f_out.close()
    df = pd.DataFrame(rows)
    print(f"\n[OK] Saved: {csv_path} ({len(df)} rows)")

    def outcome_of(row):
        if row['is_out_of_scope']:
            return 'HARD_REFUSE'
        if row['needs_human_review']:
            return 'NEEDS_REVIEW'
        return 'AUTO_ALLOW'
    df['outcome'] = df.apply(outcome_of, axis=1)

    print("\n[Outcome by category]")
    print(df.groupby(['category', 'outcome']).size().unstack(fill_value=0))

    n_borderline = (df.category == 'RQ3_borderline').sum()
    borderline_reviewed = ((df.category == 'RQ3_borderline') & (df.outcome == 'NEEDS_REVIEW')).sum()
    borderline_still_wrongly_refused = ((df.category == 'RQ3_borderline') & (df.outcome == 'HARD_REFUSE')).sum()
    n_clear = (df.category == 'RQ3_clear_legitimate').sum()
    clear_auto_allowed = ((df.category == 'RQ3_clear_legitimate') & (df.outcome == 'AUTO_ALLOW')).sum()
    clear_needlessly_reviewed = ((df.category == 'RQ3_clear_legitimate') & (df.outcome == 'NEEDS_REVIEW')).sum()
    clear_wrongly_refused = ((df.category == 'RQ3_clear_legitimate') & (df.outcome == 'HARD_REFUSE')).sum()
    n_adv = (df.category == 'Adversarial_should_refuse').sum()
    adv_confidently_refused = ((df.category == 'Adversarial_should_refuse') & (df.outcome == 'HARD_REFUSE')).sum()
    adv_at_least_flagged = ((df.category == 'Adversarial_should_refuse') & (df.outcome != 'AUTO_ALLOW')).sum()

    print(f"\n[5 known-borderline] routed to human review: {borderline_reviewed}/{n_borderline} "
          f"| still silently hard-refused: {borderline_still_wrongly_refused}/{n_borderline}")
    print(f"[{n_clear} clear-legitimate] auto-allowed (no noise): {clear_auto_allowed}/{n_clear} "
          f"| needlessly sent to review: {clear_needlessly_reviewed}/{n_clear} "
          f"| wrongly hard-refused: {clear_wrongly_refused}/{n_clear}")
    print(f"[{n_adv} adversarial] confidently hard-refused: {adv_confidently_refused}/{n_adv} "
          f"| at least not silently allowed (refused OR flagged): {adv_at_least_flagged}/{n_adv}")


if __name__ == '__main__':
    main()
