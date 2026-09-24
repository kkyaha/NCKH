# -*- coding: utf-8 -*-
"""
Scope Gate: genuine held-out test on the conformal-calibrated Scope Gate
========================================================================
HISTORY: this script originally compared an OR-rule Scope Gate against a
single-threshold conformal alternative. The OR-rule has since been removed
from src/agents/parser_agent.py entirely -- the conformal, two-threshold
(TAU_PASS / TAU_REFUSE) design is now the only Scope Gate mechanism, after
this held-out run showed the single-threshold version merely tied the
OR-rule (10/10 recall, 1/10 false-refusal each) and a follow-up recalibration
using a pooled 81-point set (this script's 20 + the original 61) with a
restored HITL review band cut false-refusals to 3/60 -- see
docs/paper_draft.tex, "Behavior at the Edge of the Declared Scope", for the
full sequence. This script now serves to collect FURTHER held-out data
against the current (sole) mechanism, e.g. to eventually refit the logistic
score itself on more than the original 61 prompts.

Every existing Scope Gate dataset in the repo predating this script
(g6_scope_gate_adversarial.csv, g6_scope_gate_hitl_validation.csv,
scope_gate_embedding_calibration.csv) was checked and confirmed to be the
SAME 61 prompts used to fit/calibrate the conformal scorer
(experiments/scope_gate_conformal_calibration.py). HOLDOUT_SET below is 20
new, hand-authored prompts (10 legitimate, 10 out-of-scope -- 3 with no
keyword overlap at all, 7 using the SAME "incidental polysemous keyword"
adversarial design as the original 11-prompt set, but with different words:
rate, code, track, order, review, recommend, comment -- each meaning
something unrelated to its Sock Shop archetype).

Uses the real GOOGLE_API_KEY-configured LLM -- burns live API quota (20 calls).
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.architecture_agent import ArchitectureAgent
from agents.parser_agent import ParserAgent

GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')

# (requirement, true_should_refuse, category)
HOLDOUT_SET = [
    # ---- Legitimate (10) -- new wording, not paraphrases of the 61 calibration prompts ----
    ("Add a wishlist feature so customers can save items for later without adding them to the cart.", False, "legitimate"),
    ("Allow customers to filter products by price range on the catalogue page.", False, "legitimate"),
    ("Send an email notification to the customer when their order status changes to shipped.", False, "legitimate"),
    ("Add a 'buy now' button that skips the cart and goes straight to checkout.", False, "legitimate"),
    ("Let customers reset their password via a link sent to their email.", False, "legitimate"),
    ("Show the estimated delivery date on the product page before purchase.", False, "legitimate"),
    ("Add a loyalty points system that rewards customers for every purchase they make.", False, "legitimate"),
    ("Allow customers to leave a star rating without writing a text review.", False, "legitimate"),
    ("Add a live chat widget so customers can ask questions before checking out.", False, "legitimate"),
    ("Let customers save multiple shipping addresses to their account profile.", False, "legitimate"),
    # ---- Out-of-scope, no keyword overlap at all (3, easy) ----
    ("Migrate our internal CI/CD pipeline from Jenkins to GitHub Actions.", True, "out_of_scope_easy"),
    ("Set up a new VPN so remote employees can access the internal network.", True, "out_of_scope_easy"),
    ("Write a script to back up the production database nightly to S3.", True, "out_of_scope_easy"),
    # ---- Out-of-scope, incidental polysemous keyword (7, adversarial, NEW words vs. original 11) ----
    ("Rate-limit our public API to prevent abuse from bots.", True, "out_of_scope_adversarial"),
    ("Add code comments and refactor the checkout module for readability.", True, "out_of_scope_adversarial"),
    ("Track employee login attempts for a security audit dashboard.", True, "out_of_scope_adversarial"),
    ("Reorder the CI pipeline stages so linting runs before the test suite.", True, "out_of_scope_adversarial"),
    ("Review the code diff before merging the pull request into main.", True, "out_of_scope_adversarial"),
    ("Recommend a new caching library for reducing our database load.", True, "out_of_scope_adversarial"),
    ("Comment out the deprecated authentication endpoint before removing it.", True, "out_of_scope_adversarial"),
]


def main():
    print("=" * 80)
    print(f"  SCOPE GATE HELD-OUT TEST -- {len(HOLDOUT_SET)} new prompts, live LLM")
    print("=" * 80)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not set -- cannot run a live held-out test.")
        return None
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
    arch_agent = ArchitectureAgent(GRAPH_PATH)
    parser = ParserAgent(llm=llm, arch_agent=arch_agent)  # conformal Scope Gate is the only mechanism now

    rows = []
    for i, (req, true_refuse, category) in enumerate(HOLDOUT_SET):
        result = parser.parse(req)
        predicted_refuse = result.is_out_of_scope
        predicted_review = result.needs_human_review
        raw = result.scope_gate_raw_features or {}
        rows.append({
            'i': i, 'category': category, 'true_should_refuse': true_refuse,
            'scope_gate_score': result.scope_gate_score,
            'rb_similarity': raw.get('rb_similarity'),
            'llm_pick_similarity': raw.get('llm_pick_similarity'),
            'keyword_similarity_score': raw.get('keyword_similarity_score'),
            'llm_is_customer_facing': raw.get('llm_is_customer_facing'),
            'predicted_refuse': predicted_refuse, 'predicted_review': predicted_review,
            'correct': (predicted_refuse == true_refuse) if not predicted_review else None,
            'requirement': req,
        })
        outcome = 'REVIEW' if predicted_review else ('REFUSE' if predicted_refuse else 'PASS')
        print(f"[{i:2d}][{category:24s}] true={true_refuse!s:5s} "
              f"outcome={outcome:7s}(score={result.scope_gate_score:.3f})  -- {req[:60]}")

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results', 'scope_gate_holdout_test.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

    y = df['true_should_refuse']
    print("\n" + "=" * 80)
    print("[RESULT] Conformal Scope Gate (only mechanism) on this set:")
    print(f"  Auto-pass, true out-of-scope (MISSED):       {((~df['predicted_refuse']) & (~df['predicted_review']) & y).sum()}")
    print(f"  Auto-refuse, true legitimate (wrong-refuse): {(df['predicted_refuse'] & ~y).sum()}")
    print(f"  Routed to human review:                      {df['predicted_review'].sum()}/{len(df)}")

    print("\n[RESULT] New conformal rule on held-out set:")
    print(f"  Recall (out-of-scope caught): {(df['new_rule_refuse'] & y).sum()}/{y.sum()}")
    print(f"  Legitimate false-refusal:     {(df['new_rule_refuse'] & ~y).sum()}/{(~y).sum()}")
    print(f"  Overall accuracy: {df['new_rule_correct'].mean():.1%}")

    print(f"\n[OK] Saved: {csv_path}")
    print("=" * 80)
    return df


if __name__ == '__main__':
    main()
