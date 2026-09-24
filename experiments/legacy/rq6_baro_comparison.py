# -*- coding: utf-8 -*-
"""
RQ6 diagnostic follow-up #4 — Does BARO (FSE'24), a non-causal statistical
RCA baseline, also fail on the same hard node (ts-travel-service_cpu) where
interventional Shapley failed, or does it succeed where Shapley did not?
============================================================================
This isolates whether the ts-travel-service_cpu weakness found in RQ6 Part B
is a property of the DATA/node (any RCA method would struggle) or a
property specific to interventional-Shapley attribution.

BARO's core statistic (Nguyen et al., FSE'24; RCAEval's own e2e/baro.py) is
reused VERBATIM: fit a RobustScaler (median/IQR) on the normal/baseline
distribution of each candidate column, transform the anomalous observations
through it, and rank columns by max absolute robust z-score. We skip
RCAEval's dataset-specific `preprocess()` step (column selection/renaming
for their raw CSV schema) since our data is already a clean DataFrame of
named DAG columns -- but the ranking rule itself (RobustScaler + max
z-score) is copied exactly from RCAEval's e2e/baro.py, not reimplemented
from a paper description.

Same controlled ground-truth-by-construction scenarios as
rq6_tier_decomposition_v2.py:
  tier1_driven: do(gateway_workload = 2.5x baseline) -- true cause is
    upstream, i.e. NOT the target's own column.
  tier2_driven: gateway held at baseline, target's own value manually
    perturbed +5 std -- true cause IS the target's own column.

BARO doesn't take a "target" as input (it globally ranks all columns), so
its analog of "correct" is: for tier1_driven, the target's own column
should NOT be top-1; for tier2_driven, the target's own column SHOULD be
top-1 (BARO is directly shown the perturbed column, so this direction is
expected to be near-trivial for it -- included as a sanity check).

No LLM API calls -- pure SCM/DoWhy computation + BARO's own statistic.
"""

import os
import sys
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from dowhy import gcm

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from capacity_agent import CapacityAgent

INJECTION_NODE = 'ts-preserve-service_workload'
TARGETS = ['ts-travel-service_cpu', 'ts-user-service_cpu']  # hard case, easy case (per Shapley)

# Known Shapley results for these same 2 targets, tier1_driven scenario
# (from rq6_tier_decomposition_v2.csv / rq6_hop_distance_diagnostic.csv),
# for a direct side-by-side print -- not recomputed here.
SHAPLEY_TIER1_DRIVEN_ACC = {
    'ts-travel-service_cpu': {'strict': '0-10%', 'upstream': '0-10%'},
    'ts-user-service_cpu':   {'strict': '80%',   'upstream': '60%'},
}


def baro_core(normal_df: pd.DataFrame, anomal_df: pd.DataFrame):
    """RCAEval's e2e/baro.py core ranking rule, copied verbatim (no dataset-
    specific preprocessing -- our columns are already clean DAG node names)."""
    ranks = []
    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()
        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    return ranks  # list of (col, score), descending


def main(n_repeats=10, dose=2.5, perturb_std=5.0):
    print("=" * 80)
    print("  RQ6 diagnostic #4: BARO (non-causal statistical RCA) on the same nodes")
    print(f"  dose={dose}x | perturb_std={perturb_std} | repeats={n_repeats} | No LLM calls.")
    print("=" * 80)

    print("\n[Train Ticket] Training accurate-path DAG (CapacityAgent)...")
    cap = CapacityAgent(system_type='trainticket', auto_train=False)
    cap.train_accurate_path()
    model, df_baseline, g = cap.global_dag_model, cap.global_df_baseline, cap.dag_graph

    normal_df = df_baseline  # BARO's "before injection" window: the training baseline
    base_val = df_baseline[INJECTION_NODE].mean()
    std_map = df_baseline.std()
    mean_map = df_baseline.mean()

    all_rows = []
    for target in TARGETS:
        # Fair, apples-to-apples candidate set: restrict BARO to the SAME
        # causal neighborhood gcm.attribute_anomalies is confined to (the
        # target plus its causal ancestors) -- not all 112 system-wide
        # columns, which is a different (harder, unrelated) global-scan task.
        ancestors = sorted(nx.ancestors(g, target))
        candidate_cols = [target] + ancestors
        candidate_cols = [c for c in candidate_cols if c in df_baseline.columns]
        print(f"\n{'='*60}\n[{target}]  candidate set = target + {len(ancestors)} causal ancestors "
              f"(Shapley tier1_driven acc: strict={SHAPLEY_TIER1_DRIVEN_ACC[target]['strict']}, "
              f"upstream={SHAPLEY_TIER1_DRIVEN_ACC[target]['upstream']})\n{'='*60}")

        # ---- Scenario A: tier1_driven (true cause = upstream, NOT target) ----
        target_rank_a, best_ancestor_beats_target_a, inj_score_beats_target_a = [], [], []
        for i in range(n_repeats):
            anomal_df = gcm.interventional_samples(
                model, interventions={INJECTION_NODE: lambda x, w=base_val * dose: w},
                num_samples_to_draw=200)
            common_cols = [c for c in candidate_cols if c in anomal_df.columns]
            ranks = baro_core(normal_df[common_cols], anomal_df[common_cols])
            score_map = dict(ranks)
            rank_order = [c for c, _ in ranks]
            target_rank_a.append(rank_order.index(target) + 1)
            target_score = score_map[target]
            ancestor_scores = {c: s for c, s in ranks if c != target}
            best_ancestor = max(ancestor_scores, key=ancestor_scores.get) if ancestor_scores else None
            best_ancestor_beats_target_a.append(
                ancestor_scores.get(best_ancestor, -np.inf) > target_score if best_ancestor else False)
            if INJECTION_NODE in score_map:
                inj_score_beats_target_a.append(score_map[INJECTION_NODE] > target_score)

        baro_correct_a = 100.0 * np.mean(best_ancestor_beats_target_a)
        print(f"  [tier1_driven] Among {len(candidate_cols)} candidates (target + its own causal "
              f"ancestors): best ancestor outscores target in {baro_correct_a:.0f}% of {n_repeats} "
              f"repeats (this is BARO's analog of 'upstream correct')")
        print(f"    mean rank of target among its own candidate set: {np.mean(target_rank_a):.1f} "
              f"/ {len(candidate_cols)}")
        if inj_score_beats_target_a:
            print(f"    TRUE injected node ({INJECTION_NODE}) itself outscores target in "
                  f"{100*np.mean(inj_score_beats_target_a):.0f}% of repeats")

        # ---- Scenario B: tier2_driven (true cause = target's own column) ----
        top1_is_target_b = []
        for i in range(n_repeats):
            anomal_row = mean_map.copy()
            anomal_row[INJECTION_NODE] = base_val
            anomal_row[target] = mean_map[target] + perturb_std * std_map[target]
            anomal_df = pd.DataFrame([anomal_row] * 50)
            common_cols = [c for c in candidate_cols if c in anomal_df.columns]
            ranks = baro_core(normal_df[common_cols], anomal_df[common_cols])
            rank_order = [c for c, _ in ranks]
            top1_is_target_b.append(rank_order[0] == target)

        baro_correct_b = 100.0 * np.mean(top1_is_target_b)
        print(f"  [tier2_driven] BARO top-1 (within candidate set) == target rate "
              f"(CORRECT if so): {baro_correct_b:.0f}%")

        all_rows.append({
            'target': target, 'scenario': 'tier1_driven', 'n_candidates': len(candidate_cols),
            'baro_correct_pct': baro_correct_a,
            'shapley_strict_acc': SHAPLEY_TIER1_DRIVEN_ACC[target]['strict'],
            'shapley_upstream_acc': SHAPLEY_TIER1_DRIVEN_ACC[target]['upstream'],
        })
        all_rows.append({
            'target': target, 'scenario': 'tier2_driven', 'n_candidates': len(candidate_cols),
            'baro_correct_pct': baro_correct_b,
            'shapley_strict_acc': '100%', 'shapley_upstream_acc': '100%',
        })

    result = pd.DataFrame(all_rows)
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results', 'rq6_baro_comparison.csv')
    result.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Saved: {csv_path}")
    print("\n[Summary] BARO vs Shapley side by side:")
    print(result.to_string(index=False))
    print("=" * 80)
    return result


if __name__ == '__main__':
    main()
