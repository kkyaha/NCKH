# -*- coding: utf-8 -*-
"""
Scope Gate Redesign — Calibrating a Semantic-Embedding Threshold on Real Data
==================================================================================
The current Scope Gate (Layer 1) uses a literal zero-keyword-overlap test, which an
11-prompt adversarial set showed evades in 100% of cases where the out-of-scope
requirement happens to contain one polysemous word shared with an archetype's keyword
list (data/processed/scm_results/g6_scope_gate_adversarial.csv). This script tests the
proposed fix -- multilingual sentence embeddings (intfloat/multilingual-e5-base, Wang
et al. 2024, arXiv:2402.05672) with cosine similarity against each archetype's
description -- and calibrates the refusal threshold on REAL labeled data rather than
an assumed value (an initial spot-check showed a naive threshold of 0.65 would fail
outright: the wrong archetype for a genuinely out-of-scope prompt scored 0.80, higher
than that threshold).

Labeled data (no new LLM calls -- pure embedding computation):
  Positive class (should NOT refuse, label=1): all 50 requirements from RQ3's own
    benchmark (data/benchmark/parser_benchmark_prompts.json) -- genuinely in-taxonomy
    by construction.
  Negative class (SHOULD refuse, label=0): the 11 adversarial prompts built to attack
    the Scope Gate (data/processed/scm_results/g6_scope_gate_adversarial.csv) --
    genuinely out-of-taxonomy by construction, each with one incidental keyword.

For each prompt, we compute the max cosine similarity between its embedding and every
archetype's PASSAGE embedding (description + keywords, per e5's query/passage
convention), then sweep a threshold and report precision/recall/F1 for the "refuse"
decision (predicted negative = max similarity < threshold) against the true label.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from request_router import CALL_CHAINS

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
PROMPTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'parser_benchmark_prompts.json')
ADVERSARIAL_FILE = os.path.join(OUTPUT_DIR, 'g6_scope_gate_adversarial.csv')


def build_archetype_passages(enriched=False):
    """One passage per archetype: the natural-language description ONLY, e5 'passage:'
    convention. Deliberately excludes the keyword list: the adversarial prompts in
    g6_scope_gate_adversarial.csv were each constructed to contain exactly one literal
    keyword from CALL_CHAINS, so folding those same keywords into the passage text
    would let literal lexical overlap leak back into what is supposed to be a purely
    semantic (meaning-based) comparison -- re-introducing the exact confound this
    redesign is meant to escape, just laundered through an embedding model instead of
    a regex.

    `enriched=True` substitutes a longer, domain-grounded description for each
    archetype (explicitly naming "SockShop e-commerce", "customer", "online store")
    instead of CALL_CHAINS' original 4-8 word description -- testing whether the
    ORIGINAL short descriptions were simply too thin for the embedding model to latch
    onto a discriminative domain signal, independent of the keyword-overlap confound
    above."""
    if not enriched:
        return {rt: f"passage: {info.get('description', rt)}" for rt, info in CALL_CHAINS.items()}

    ENRICHED = {
        'GET_CATALOGUE': "A customer browsing the SockShop online sock e-commerce store views the product catalogue, product listings, or a product detail page while shopping for socks.",
        'ADD_TO_CART': "A customer shopping on the SockShop e-commerce website adds a sock product item to their shopping cart before checking out.",
        'VIEW_CART': "A customer on the SockShop online sock store views the current contents of their shopping cart before completing a purchase.",
        'REGISTER': "A new customer creates a shopping account by registering on the SockShop e-commerce website in order to buy socks online.",
        'LOGIN': "An existing customer logs into their shopping account on the SockShop e-commerce website to buy socks online.",
        'PLACE_ORDER': "A customer on the SockShop online sock store completes checkout and places a full purchase order, including payment and shipping of socks.",
        'APPLY_PROMO_CODE': "A customer shopping on the SockShop e-commerce website applies a discount promo code or voucher coupon during checkout to reduce the price of their sock order.",
        'RECOMMEND_PRODUCTS': "The SockShop e-commerce website shows a customer personalized product recommendations for socks based on their past shopping and browsing history.",
        'TRACK_PACKAGE': "A customer of the SockShop online sock store tracks the real-time delivery status and shipping location of their purchased package order.",
        'WRITE_PRODUCT_REVIEW': "A customer of the SockShop e-commerce website writes a product review and star rating for a sock item they previously purchased.",
    }
    return {rt: f"passage: {ENRICHED.get(rt, info.get('description', rt))}" for rt, info in CALL_CHAINS.items()}


def main():
    model_name = 'intfloat/multilingual-e5-base'
    enriched = False
    for arg in sys.argv:
        if arg.startswith('--model='):
            model_name = arg.split('=', 1)[1]
        if arg == '--enriched':
            enriched = True

    print(f"[Config] model={model_name} enriched_descriptions={enriched}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    archetype_passages = build_archetype_passages(enriched=enriched)
    archetype_names = list(archetype_passages.keys())
    archetype_texts = [archetype_passages[k] for k in archetype_names]
    archetype_emb = model.encode(archetype_texts, normalize_embeddings=True)

    # ---- Positive class: RQ3's 50 real, in-taxonomy prompts ----
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        rq3_prompts = json.load(f)
    pos_rows = []
    pos_texts = [f"query: {p['requirement']}" for p in rq3_prompts]
    pos_emb = model.encode(pos_texts, normalize_embeddings=True)
    for p, emb in zip(rq3_prompts, pos_emb):
        sims = archetype_emb @ emb
        best_idx = int(np.argmax(sims))
        pos_rows.append({
            'source': 'RQ3_positive', 'prompt_id': p['id'], 'category': p['category'],
            'requirement': p['requirement'], 'label': 1,
            'max_sim': float(sims[best_idx]), 'best_archetype': archetype_names[best_idx],
        })

    # ---- Negative class: the 11 Scope-Gate adversarial prompts (unique, not x3 repeats) ----
    adv_df = pd.read_csv(ADVERSARIAL_FILE)
    adv_unique = adv_df.drop_duplicates(subset=['prompt_id'])[['prompt_id', 'requirement']]
    neg_rows = []
    neg_texts = [f"query: {r}" for r in adv_unique['requirement']]
    neg_emb = model.encode(neg_texts, normalize_embeddings=True)
    for (_, row), emb in zip(adv_unique.iterrows(), neg_emb):
        sims = archetype_emb @ emb
        best_idx = int(np.argmax(sims))
        neg_rows.append({
            'source': 'Adversarial_negative', 'prompt_id': row['prompt_id'], 'category': 'adversarial',
            'requirement': row['requirement'], 'label': 0,
            'max_sim': float(sims[best_idx]), 'best_archetype': archetype_names[best_idx],
        })

    df = pd.DataFrame(pos_rows + neg_rows)
    variant = model_name.split('/')[-1] + ('_enriched' if enriched else '_shortdesc')
    csv_path = os.path.join(OUTPUT_DIR, f'scope_gate_embedding_calibration_{variant}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"[OK] Saved: {csv_path} ({len(df)} rows: {len(pos_rows)} positive, {len(neg_rows)} negative)")

    print("\n[Positive class max_sim distribution]")
    print(df[df.label == 1]['max_sim'].describe())
    print("\n[Negative class max_sim distribution]")
    print(df[df.label == 0]['max_sim'].describe())

    # ---- Threshold sweep ----
    print("\n[Threshold sweep] refuse if max_sim < threshold")
    results = []
    for thresh in np.arange(0.70, 0.96, 0.01):
        pred_refuse = df['max_sim'] < thresh  # predicted negative (refuse)
        true_negative_mask = df['label'] == 0
        tp = (pred_refuse & true_negative_mask).sum()          # correctly refused
        fn = (~pred_refuse & true_negative_mask).sum()         # should refuse, didn't (evasion)
        fp = (pred_refuse & ~true_negative_mask).sum()         # falsely refused a legit prompt
        tn = (~pred_refuse & ~true_negative_mask).sum()        # correctly allowed
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')
        results.append({'threshold': round(thresh, 2), 'refusal_precision': round(precision, 3),
                         'refusal_recall': round(recall, 3), 'f1': round(f1, 3),
                         'false_positive_on_legit': int(fp), 'evasions_remaining': int(fn)})
    res_df = pd.DataFrame(results)
    res_csv = os.path.join(OUTPUT_DIR, f'scope_gate_threshold_sweep_{variant}.csv')
    res_df.to_csv(res_csv, index=False)
    print(res_df.to_string(index=False))

    best = res_df.loc[res_df['f1'].idxmax()]
    print(f"\n[Best F1] threshold={best['threshold']}, precision={best['refusal_precision']}, "
          f"recall={best['refusal_recall']}, f1={best['f1']}, "
          f"false_positives_on_legit={best['false_positive_on_legit']}, "
          f"evasions_remaining={best['evasions_remaining']}")
    print(f"[OK] Sweep saved: {res_csv}")


if __name__ == '__main__':
    main()
