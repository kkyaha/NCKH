# -*- coding: utf-8 -*-
"""
Conformal Calibration for the Scope Gate (Layer 1) — a Real Trial
========================================================================
Question: can split-conformal calibration give the Scope Gate's abstention
decision a FORMAL, distribution-free guarantee (Pr[out-of-scope AND wrongly
allowed] <= alpha) instead of the current heuristic (keyword-overlap OR
LLM self-declaration, patched ad hoc after each adversarial finding)?

Data: data/processed/scm_results/g6_scope_gate_layerA_validation.csv --
61 prompts (50 legitimate RQ3 prompts + the 11-prompt polysemous-keyword
adversarial set) with a real ground-truth label (`true_should_refuse`) and
the two raw signals the deployed Scope Gate already computes
(`keyword_similarity_score` continuous in [0, ~1.67],
`llm_is_customer_facing` boolean). No new LLM calls -- this is a pure
re-analysis of already-collected, already-labeled data.

Step 0 (diagnostic, done before writing any calibration code): check how
discriminative each raw signal already is. `llm_is_customer_facing==False`
alone already has PERFECT recall on the 11 true out-of-scope prompts (11/11)
but a 20% (10/50) false-positive rate on legitimate ones; `keyword_similarity
_score`'s class-conditional means (0.606 out-of-scope vs 0.453 legitimate)
barely separate and are in the "wrong" direction (out-of-scope prompts have
HIGHER mean keyword overlap) -- consistent with the paper's own finding
that keyword-overlap alone is a weak signal. This matters: conformal
calibration cannot manufacture discriminative power a feature does not
have -- it can only give a FORMAL guarantee about whatever score you feed
it. We report this ceiling honestly rather than only the final numbers.

Method: split-conformal calibration for one-sided risk control (Bates et
al.-style "Learn then Test" / conformal risk control, simplified to a
single scalar score and threshold). A logistic-regression score, fit on
[keyword_similarity_score, llm_pick_similarity, rb_similarity,
llm_is_customer_facing] via 5-fold cross-validation (to avoid evaluating
the score on the same rows it was fit on, given n=61 is small), serves as
the nonconformity score s(x) = 1 - P(in-scope | x). For a target miss-rate
alpha, the calibrated threshold is the ceil((n_pos+1)(1-alpha))-th smallest
score among the TRUE out-of-scope calibration examples (the standard
split-conformal quantile with finite-sample correction) -- any point with
s(x) >= threshold is refused.

No LLM API calls -- pure re-analysis of already-collected data.
"""

import math
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results',
                          'g6_scope_gate_layerA_validation.csv')


def conformal_threshold(cal_scores_pos: np.ndarray, alpha: float) -> float:
    """Standard split-conformal quantile with finite-sample correction:
    guarantees Pr[new true-positive's score < threshold] <= alpha, i.e.
    Pr[miss] <= alpha, under exchangeability with the calibration set.

    We REFUSE when score >= threshold, so to bound the miss probability
    (score falls BELOW threshold) at alpha, the threshold must sit near
    the LOW end of the true-positive score distribution -- the k-th
    smallest value with k = ceil((n+1) * alpha), not the (1-alpha)
    quantile (an earlier version of this script had this backwards,
    which silently produced a threshold near the top of the positive
    distribution -- the opposite of a miss-rate guarantee -- caught by
    the resulting recall collapsing to 1/11 on a sanity check before any
    result was reported)."""
    n = len(cal_scores_pos)
    k = max(1, math.ceil((n + 1) * alpha))
    k = min(k, n)
    sorted_scores = np.sort(cal_scores_pos)
    return sorted_scores[k - 1]


def main():
    print("=" * 80)
    print("  CONFORMAL CALIBRATION FOR THE SCOPE GATE -- a real trial on real data")
    print(f"  Data: {DATA_PATH}")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    y = df['true_should_refuse'].astype(bool).to_numpy()
    n_pos, n_neg = y.sum(), (~y).sum()
    print(f"\n[Data] n={len(df)} ({n_pos} true out-of-scope, {n_neg} legitimate)")

    # ---- Step 0: raw signal discrimination (done first, on purpose) ----
    print("\n[Step 0] Raw signal discrimination check:")
    ctab = pd.crosstab(df['llm_is_customer_facing'], df['true_should_refuse'])
    print(ctab.to_string())
    recall_bool = ((df['llm_is_customer_facing'] == False) & y).sum() / n_pos
    fpr_bool = ((df['llm_is_customer_facing'] == False) & (~y)).sum() / n_neg
    print(f"  'is_customer_facing==False' alone: recall={recall_bool:.1%}, false-positive-rate={fpr_bool:.1%}")
    means = df.groupby('true_should_refuse')['keyword_similarity_score'].mean()
    print(f"  keyword_similarity_score class means: legitimate={means[False]:.3f}, out-of-scope={means[True]:.3f}"
          f" ({'WRONG direction' if means[True] < means[False] else 'right direction, weak' if means[True]-means[False]<0.3 else 'informative'})")

    # ---- Fit a combined score via 5-fold CV (avoid evaluating on training rows) ----
    features = ['keyword_similarity_score', 'llm_pick_similarity', 'rb_similarity']
    X = df[features].fillna(0.0).to_numpy()
    X = np.hstack([X, df[['llm_is_customer_facing']].astype(float).to_numpy()])
    y_int = y.astype(int)

    cv_scores = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X, y_int):
        if y_int[train_idx].sum() == 0 or y_int[train_idx].sum() == len(train_idx):
            cv_scores[test_idx] = y_int[train_idx].mean()  # degenerate fold fallback
            continue
        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(X[train_idx], y_int[train_idx])
        cv_scores[test_idx] = clf.predict_proba(X[test_idx])[:, 1]  # P(out-of-scope)
    df['nonconformity_score'] = cv_scores  # higher = more suspicious

    print("\n[Fitted score] class-conditional distribution (out-of-fold predictions):")
    print(df.groupby('true_should_refuse')['nonconformity_score'].describe()[['mean', 'std', 'min', 'max']].to_string())

    # ---- Split-conformal calibration at several target miss-rates alpha ----
    print("\n[Conformal calibration] Using ALL 11 true-positive scores as the calibration")
    print("set (n is too small to further split cal/test -- we flag this explicitly):")
    pos_scores = df.loc[y, 'nonconformity_score'].to_numpy()
    neg_scores = df.loc[~y, 'nonconformity_score'].to_numpy()

    results = []
    for alpha in [0.30, 0.20, 0.10, 0.05, 0.01]:
        tau = conformal_threshold(pos_scores, alpha)
        refused_pos = (pos_scores >= tau).sum()
        refused_neg = (neg_scores >= tau).sum()
        results.append({
            'target_alpha_miss_rate': alpha, 'threshold': round(tau, 4),
            'actual_recall_on_calibration': f"{refused_pos}/{n_pos}",
            'legitimate_false_refusal_rate': f"{refused_neg}/{n_neg} ({100*refused_neg/n_neg:.1f}%)",
        })
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results',
                             'scope_gate_conformal_calibration.csv')
    df[['prompt_id', 'true_should_refuse', 'keyword_similarity_score',
        'llm_is_customer_facing', 'nonconformity_score']].to_csv(csv_path, index=False)
    res_df.to_csv(csv_path.replace('.csv', '_thresholds.csv'), index=False)
    print(f"\n[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {csv_path.replace('.csv', '_thresholds.csv')}")

    print("\n[Comparison] Current deployed heuristic (from this same validation set):")
    current_fpr = (df['predicted_refuse'] & ~y).sum() / n_neg
    current_recall = (df['predicted_refuse'] & y).sum() / n_pos
    print(f"  Deployed rule: recall={current_recall:.1%}, legitimate false-refusal-rate={current_fpr:.1%}")
    print("=" * 80)
    return df, res_df


if __name__ == '__main__':
    main()
