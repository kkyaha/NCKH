# -*- coding: utf-8 -*-
"""
RQ6 (Part B, v2) — Corrected Tier-1 vs Tier-2 Shapley Decomposition
========================================================================
Follow-up to rq6_tier_decomposition.py (v1). v1's classify_contribs() only
recognized two contributor kinds: any `*_workload` node (Tier 1) and the
target's own key (Tier 2). It silently dropped a THIRD kind that now exists
in the trained DAG: backpressure R->R edges (another service's own CPU as
a second parent of the target's CPU, added per Section "backpressure-edge"
of the paper). Cross-checking the backpressure-edge list against v1's test
targets showed this affects 3/6 Sock Shop targets (shipping_cpu, carts_cpu,
user_cpu) and 4/4 Train Ticket targets (all of them) -- i.e. most of v1's
"tier1_driven" trials had part of their true contribution mass silently
excluded from both tier sums.

v1's Sock Shop tier1_driven accuracy was 78.3% overall but ranged from 0%
(payment_cpu, which has NO backpressure edge) to 100% (three backpressure
targets); Train Ticket's tier1_driven accuracy was only 17.5% overall. Since
the worst Sock Shop node has no backpressure edge at all, the backpressure
gap cannot be the *only* explanation -- this script isolates how much of the
gap it actually explains versus what remains a genuine mechanism weakness.

This version classifies every ancestor contributor into THREE buckets:
  - tier1        : any `*_workload` node (Tier-1 network propagation)
  - backpressure : any other `*_cpu`/`*_mem`/`*_latency-50` node that is NOT
                    the target itself (a caller's own resource metric,
                    reaching the target only via a backpressure edge)
  - tier2         : the target's own key (local noise/residual)

and reports TWO accuracy numbers per trial:
  (a) strict_correct   : does tier1 alone dominate (tier1_driven) or tier2
                          alone dominate (tier2_driven)? -- the same standard
                          v1 used, now with backpressure mass no longer
                          silently merged into either tier.
  (b) upstream_correct : does (tier1 + backpressure), i.e. "anything NOT the
                          target's own local noise", dominate for
                          tier1_driven, and tier2 alone dominate for
                          tier2_driven? -- the practically relevant question
                          for a bottleneck-ranking heuristic ("look upstream
                          or look here"), and the fairer test given the paper
                          itself (Section "workload-propagation") already
                          flags that backpressure nodes narrow the clean
                          two-way interpretability on exactly these targets.

Same controlled, ground-truth-by-construction design as v1 (tier1_driven:
do(gateway_workload=dose); tier2_driven: gateway held at baseline, target's
own value manually perturbed +k std). Runs on the SAME targets as v1 for a
direct comparison. No LLM API calls.
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from evaluation_suite import build_and_train_global_dag
from capacity_agent import CapacityAgent

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

RESOURCE_SUFFIXES = ('_cpu', '_mem', '_latency-50')


def classify_contribs_v2(contribs: dict, target_node: str):
    """Split a per-ancestor contribution dict into (tier1, backpressure, tier2)."""
    tier1 = 0.0
    backpressure = 0.0
    tier2 = 0.0
    for node, arr in contribs.items():
        val = float(arr[0]) if len(arr) > 0 else 0.0
        if node == target_node:
            tier2 += val
        elif node.endswith('_workload'):
            tier1 += val
        elif node.endswith(RESOURCE_SUFFIXES):
            # another service's own resource metric reaching the target only
            # via a backpressure (R->R) edge
            backpressure += val
        # (anything else would be an unexpected node kind in this two-tier
        # DAG; none expected -- silently ignored if it ever appears.)
    return tier1, backpressure, tier2


def run_tier_test_v2(model, df_baseline, gateway_workload_node, target_nodes,
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
                t1_a, bp_a, t2_a = classify_contribs_v2(contribs_a, target)
                strict_pred_a = 'tier1' if abs(t1_a) > abs(t2_a) else 'tier2'
                upstream_pred_a = 'upstream' if abs(t1_a + bp_a) > abs(t2_a) else 'local'
            except Exception as e:
                t1_a, bp_a, t2_a = np.nan, np.nan, np.nan
                strict_pred_a = upstream_pred_a = f'ERROR:{str(e)[:40]}'

            rows.append({
                'label': label, 'target': target, 'repeat': i, 'scenario': 'tier1_driven',
                'true_tier': 'tier1', 'true_upstream': 'upstream',
                'tier1_contrib': t1_a, 'backpressure_contrib': bp_a, 'tier2_contrib': t2_a,
                'strict_pred': strict_pred_a, 'strict_correct': strict_pred_a == 'tier1',
                'upstream_pred': upstream_pred_a, 'upstream_correct': upstream_pred_a == 'upstream',
            })

            # ---- Scenario B: Tier-2-driven (no real workload change, own value perturbed) ----
            anomaly_row_b = mean_map.copy()
            anomaly_row_b[gateway_workload_node] = base_val  # explicit no-op
            anomaly_row_b[target] = mean_map[target] + perturb_std * std_map[target]
            anomaly_df_b = pd.DataFrame([anomaly_row_b])
            try:
                contribs_b = gcm.attribute_anomalies(
                    model, target_node=target, anomaly_samples=anomaly_df_b,
                    attribute_mean_deviation=True, num_distribution_samples=500)
                t1_b, bp_b, t2_b = classify_contribs_v2(contribs_b, target)
                strict_pred_b = 'tier1' if abs(t1_b) > abs(t2_b) else 'tier2'
                upstream_pred_b = 'upstream' if abs(t1_b + bp_b) > abs(t2_b) else 'local'
            except Exception as e:
                t1_b, bp_b, t2_b = np.nan, np.nan, np.nan
                strict_pred_b = upstream_pred_b = f'ERROR:{str(e)[:40]}'

            rows.append({
                'label': label, 'target': target, 'repeat': i, 'scenario': 'tier2_driven',
                'true_tier': 'tier2', 'true_upstream': 'local',
                'tier1_contrib': t1_b, 'backpressure_contrib': bp_b, 'tier2_contrib': t2_b,
                'strict_pred': strict_pred_b, 'strict_correct': strict_pred_b == 'tier2',
                'upstream_pred': upstream_pred_b, 'upstream_correct': upstream_pred_b == 'local',
            })
        print(f"  [{label}] {target}: done ({n_repeats} repeats x 2 scenarios)")
    return pd.DataFrame(rows)


def main(n_repeats=10):
    for arg in sys.argv:
        if arg.startswith('--repeats='):
            n_repeats = int(arg.split('=', 1)[1])

    print("=" * 80)
    print("  RQ6 (Part B, v2): CORRECTED 3-WAY TIER-1 / BACKPRESSURE / TIER-2 TEST")
    print(f"  Repeats/target/scenario: {n_repeats} | No LLM API calls.")
    print("=" * 80)

    all_dfs = []

    # ---------------- Sock Shop ----------------
    print("\n[Sock Shop] Training 28-node DAG (evaluation_suite, monotonicity-constrained)...")
    ss_model, ss_df, ss_g = build_and_train_global_dag()
    ss_targets = ['catalogue_cpu', 'orders_cpu', 'payment_cpu', 'shipping_cpu', 'user_cpu', 'carts_cpu']
    ss_df_res = run_tier_test_v2(ss_model, ss_df, 'front-end_workload', ss_targets,
                                  n_repeats=n_repeats, dose=2.5, perturb_std=5.0, label='SockShop')
    all_dfs.append(ss_df_res)

    # ---------------- Train Ticket ----------------
    print("\n[Train Ticket] Training accurate-path DAG (CapacityAgent, now monotonicity-constrained)...")
    cap = CapacityAgent(system_type='trainticket', auto_train=False)
    cap.train_accurate_path()
    tt_model, tt_df = cap.global_dag_model, cap.global_df_baseline
    tt_targets = ['ts-order-service_cpu', 'ts-user-service_cpu', 'ts-travel-service_cpu', 'ts-seat-service_cpu']
    tt_df_res = run_tier_test_v2(tt_model, tt_df, 'ts-preserve-service_workload', tt_targets,
                                  n_repeats=n_repeats, dose=2.5, perturb_std=5.0, label='TrainTicket')
    all_dfs.append(tt_df_res)

    df = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, 'rq6_tier_decomposition_v2.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Raw results saved to: {csv_path}")

    print("\n[Result] STRICT accuracy (tier1 alone vs tier2 alone -- same standard as v1, "
          "backpressure mass no longer silently merged in):")
    strict_by_scenario = df.groupby(['label', 'scenario'])['strict_correct'].agg(n='count', accuracy_pct=lambda s: 100.0 * s.mean())
    print(strict_by_scenario.to_string())
    strict_overall = df.groupby('label')['strict_correct'].agg(n='count', accuracy_pct=lambda s: 100.0 * s.mean())
    print("\n[Result] STRICT overall accuracy per system:")
    print(strict_overall.to_string())

    print("\n[Result] UPSTREAM-vs-LOCAL accuracy ((tier1+backpressure) vs tier2 -- "
          "the fairer, practically-relevant test):")
    upstream_by_scenario = df.groupby(['label', 'scenario'])['upstream_correct'].agg(n='count', accuracy_pct=lambda s: 100.0 * s.mean())
    print(upstream_by_scenario.to_string())
    upstream_overall = df.groupby('label')['upstream_correct'].agg(n='count', accuracy_pct=lambda s: 100.0 * s.mean())
    print("\n[Result] UPSTREAM-vs-LOCAL overall accuracy per system:")
    print(upstream_overall.to_string())

    print("\n[Result] Per-target breakdown (strict vs upstream, tier1_driven scenario only "
          "-- this is where v1 broke down):")
    t1_only = df[df['scenario'] == 'tier1_driven']
    per_target = t1_only.groupby(['label', 'target']).agg(
        n=('strict_correct', 'count'),
        strict_acc_pct=('strict_correct', lambda s: 100.0 * s.mean()),
        upstream_acc_pct=('upstream_correct', lambda s: 100.0 * s.mean()),
        mean_tier1=('tier1_contrib', 'mean'),
        mean_backpressure=('backpressure_contrib', 'mean'),
        mean_tier2=('tier2_contrib', 'mean'),
    ).round(4)
    print(per_target.to_string())

    summary_path = os.path.join(OUTPUT_DIR, 'rq6_tier_decomposition_v2_summary.csv')
    per_target.to_csv(summary_path, encoding='utf-8')
    print(f"[OK] Per-target summary saved to: {summary_path}")
    print("=" * 80)
    return df, strict_overall, upstream_overall, per_target


if __name__ == '__main__':
    main()
