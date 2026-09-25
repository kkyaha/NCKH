# -*- coding: utf-8 -*-
"""
RQ6 (Part B) — Does Shapley Attribution Correctly Split Tier 1 vs Tier 2?
============================================================================
The claim under test (stated in paper_draft.tex §III-C / RQ2): interventional
Shapley attribution can decompose a projected change at a downstream target
into "how much comes from Tier-1 network workload propagation vs. Tier-2
local hardware conversion". Part A (rq6_attribution_validity.py,
rq6_topology_check.py) never actually tested this — it tested (i) dose-
response of a single node with NO Tier-1 ancestors (the injection gateway
itself) and (ii) cross-SERVICE ranking by topology, neither of which
exercises the tier-level split within one target node.

This script tests it directly with a controlled, ground-truth-by-construction
design: for a downstream target R_j (e.g. catalogue_cpu), we construct two
kinds of anomaly, each with a KNOWN true cause:

  Tier-1-driven: do(gateway_workload = baseline * dose). The target's own
    resource value is left to the fitted mechanism (no manual perturbation)
    — any anomaly in R_j is, by construction, explained by workload
    propagation reaching it, not by a locally-injected fault.

  Tier-2-driven: do(gateway_workload = baseline), i.e. NO real workload
    change, but the target's OWN resource value in the anomaly sample is
    manually perturbed by +k std (simulating an inefficiency/local fault
    independent of traffic) — any anomaly here is, by construction, NOT
    explained by workload.

gcm.attribute_anomalies() returns a per-ancestor contribution dict that
includes the target's own key (its own noise/residual) alongside every
`*_workload` ancestor. We classify contributors as Tier 1 (any `_workload`
node) vs Tier 2 (the target's own key) and check whether the dominant tier
matches the known ground truth in each scenario.

Runs on both Sock Shop (evaluation_suite's monotonicity-constrained DAG) and
Train Ticket (CapacityAgent's accurate path, now synchronized to the same
constraint). No LLM API calls.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from dowhy import gcm

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
import _paths  # noqa: F401  -- dua cac nhom con khac vao sys.path
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from evaluation_suite import build_and_train_global_dag
from capacity_agent import CapacityAgent

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def classify_contribs(contribs: dict, target_node: str):
    """Split a per-ancestor contribution dict into (tier1_sum, tier2_own)."""
    tier1 = 0.0
    tier2 = 0.0
    for node, arr in contribs.items():
        val = float(arr[0]) if len(arr) > 0 else 0.0
        if node == target_node:
            tier2 += val
        elif node.endswith('_workload'):
            tier1 += val
        # (non-workload, non-target ancestors would be a modeling error in
        # this two-tier DAG — none expected; silently ignored if present.)
    return tier1, tier2


def run_tier_test(model, df_baseline, gateway_workload_node, target_nodes,
                   n_repeats=10, dose=2.5, perturb_std=5.0, label=''):
    rows = []
    base_val = df_baseline[gateway_workload_node].mean()
    std_map = df_baseline.std()
    mean_map = df_baseline.mean()

    for target in target_nodes:
        if target not in df_baseline.columns:
            print(f"  [skip] {target} not in fitted DAG's columns")
            continue
        for i in range(n_repeats):
            # ---- Scenario A: Tier-1-driven (real workload intervention) ----
            samples = gcm.interventional_samples(
                model, interventions={gateway_workload_node: lambda x, w=base_val * dose: w},
                num_samples_to_draw=200)
            anomaly_row_a = mean_map.copy()
            for col in samples.columns:
                anomaly_row_a[col] = samples[col].mean()
            anomaly_df_a = pd.DataFrame([anomaly_row_a])
            try:
                contribs_a = gcm.attribute_anomalies(
                    model, target_node=target, anomaly_samples=anomaly_df_a,
                    attribute_mean_deviation=True, num_distribution_samples=500)
                t1_a, t2_a = classify_contribs(contribs_a, target)
                predicted_a = 'tier1' if abs(t1_a) > abs(t2_a) else 'tier2'
            except Exception as e:
                t1_a, t2_a, predicted_a = np.nan, np.nan, f'ERROR:{str(e)[:40]}'

            rows.append({'label': label, 'target': target, 'repeat': i, 'scenario': 'tier1_driven',
                         'true_tier': 'tier1', 'tier1_contrib': t1_a, 'tier2_contrib': t2_a,
                         'predicted_tier': predicted_a, 'correct': predicted_a == 'tier1'})

            # ---- Scenario B: Tier-2-driven (no real workload change, own value perturbed) ----
            anomaly_row_b = mean_map.copy()
            anomaly_row_b[gateway_workload_node] = base_val  # explicit no-op
            anomaly_row_b[target] = mean_map[target] + perturb_std * std_map[target]
            anomaly_df_b = pd.DataFrame([anomaly_row_b])
            try:
                contribs_b = gcm.attribute_anomalies(
                    model, target_node=target, anomaly_samples=anomaly_df_b,
                    attribute_mean_deviation=True, num_distribution_samples=500)
                t1_b, t2_b = classify_contribs(contribs_b, target)
                predicted_b = 'tier1' if abs(t1_b) > abs(t2_b) else 'tier2'
            except Exception as e:
                t1_b, t2_b, predicted_b = np.nan, np.nan, f'ERROR:{str(e)[:40]}'

            rows.append({'label': label, 'target': target, 'repeat': i, 'scenario': 'tier2_driven',
                         'true_tier': 'tier2', 'tier1_contrib': t1_b, 'tier2_contrib': t2_b,
                         'predicted_tier': predicted_b, 'correct': predicted_b == 'tier2'})
        print(f"  [{label}] {target}: done ({n_repeats} repeats x 2 scenarios)")
    return pd.DataFrame(rows)


def main(n_repeats=10):
    for arg in sys.argv:
        if arg.startswith('--repeats='):
            n_repeats = int(arg.split('=', 1)[1])

    print("=" * 80)
    print("  RQ6 (Part B): TIER-1 vs TIER-2 SHAPLEY ATTRIBUTION — CONTROLLED TEST")
    print(f"  Repeats/target/scenario: {n_repeats} | No LLM API calls.")
    print("=" * 80)

    all_dfs = []

    # ---------------- Sock Shop ----------------
    print("\n[Sock Shop] Training 28-node DAG (evaluation_suite, monotonicity-constrained)...")
    ss_model, ss_df, ss_g = build_and_train_global_dag()
    ss_targets = ['catalogue_cpu', 'orders_cpu', 'payment_cpu', 'shipping_cpu', 'user_cpu', 'carts_cpu']
    ss_df_res = run_tier_test(ss_model, ss_df, 'front-end_workload', ss_targets,
                               n_repeats=n_repeats, dose=2.5, perturb_std=5.0, label='SockShop')
    all_dfs.append(ss_df_res)

    # ---------------- Train Ticket ----------------
    print("\n[Train Ticket] Training accurate-path DAG (CapacityAgent, now monotonicity-constrained)...")
    cap = CapacityAgent(system_type='trainticket', auto_train=False)
    cap.train_accurate_path()
    tt_model, tt_df = cap.global_dag_model, cap.global_df_baseline
    tt_targets = ['ts-order-service_cpu', 'ts-user-service_cpu', 'ts-travel-service_cpu', 'ts-seat-service_cpu']
    tt_df_res = run_tier_test(tt_model, tt_df, 'ts-preserve-service_workload', tt_targets,
                               n_repeats=n_repeats, dose=2.5, perturb_std=5.0, label='TrainTicket')
    all_dfs.append(tt_df_res)

    df = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, 'rq6_tier_decomposition.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Raw results saved to: {csv_path}")

    summary = df.groupby(['label', 'scenario'])['correct'].agg(n='count', accuracy_pct=lambda s: 100.0 * s.mean())
    print("\n[Result] Tier-attribution accuracy (does the dominant contributor match the known cause?):")
    print(summary.to_string())

    overall = df.groupby('label')['correct'].agg(n='count', accuracy_pct=lambda s: 100.0 * s.mean())
    print("\n[Result] Overall accuracy per system:")
    print(overall.to_string())

    summary_path = os.path.join(OUTPUT_DIR, 'rq6_tier_decomposition_summary.csv')
    summary.to_csv(summary_path, encoding='utf-8')
    print(f"[OK] Summary saved to: {summary_path}")
    print("=" * 80)
    return df, summary


if __name__ == '__main__':
    main()
