# -*- coding: utf-8 -*-
"""
RQ6 diagnostic follow-up — Does tier1_driven attribution accuracy degrade
with hop-distance from the injection point?
============================================================================
v2's 4 Train Ticket targets (ts-order/seat/travel/user-service_cpu) were all
accidentally at hops=1 from the injection gateway (ts-preserve-service) --
no variance in hop-distance, so the "signal dilutes over hops" hypothesis
could not be tested against that data. This script extends the same
controlled tier1_driven/tier2_driven test (identical protocol to
rq6_tier_decomposition_v2.py) to additional Train Ticket targets spanning
hops 1, 2 and 3 (from rq6_topology_check.csv's already-computed hop map),
to see whether accuracy tracks distance or something else.

No LLM API calls -- pure SCM/DoWhy computation.
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from capacity_agent import CapacityAgent

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESOURCE_SUFFIXES = ('_cpu', '_mem', '_latency-50')
INJECTION_SERVICE = 'ts-preserve-service'
INJECTION_NODE = f'{INJECTION_SERVICE}_workload'

# Targets spanning hops 1 (already tested in v2, included here for a
# single consistent table), 2 and 3, per rq6_topology_check.csv's hop map.
TARGETS_BY_HOP = {
    1: ['ts-order-service_cpu', 'ts-seat-service_cpu', 'ts-travel-service_cpu', 'ts-user-service_cpu'],
    2: ['ts-config-service_cpu', 'ts-basic-service_cpu', 'ts-train-service_cpu', 'ts-route-service_cpu'],
    3: ['ts-price-service_cpu'],
}


def classify_contribs_v2(contribs: dict, target_node: str):
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
            backpressure += val
    return tier1, backpressure, tier2


def run_tier_test(model, df_baseline, target_nodes, hop, n_repeats=10, dose=2.5, perturb_std=5.0):
    rows = []
    base_val = df_baseline[INJECTION_NODE].mean()
    std_map = df_baseline.std()
    mean_map = df_baseline.mean()

    for target in target_nodes:
        if target not in df_baseline.columns:
            print(f"  [skip] {target} not in fitted DAG's columns")
            continue
        for i in range(n_repeats):
            samples = gcm.interventional_samples(
                model, interventions={INJECTION_NODE: lambda x, w=base_val * dose: w},
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
                strict_a = 'tier1' if abs(t1_a) > abs(t2_a) else 'tier2'
                upstream_a = 'upstream' if abs(t1_a + bp_a) > abs(t2_a) else 'local'
            except Exception as e:
                t1_a, bp_a, t2_a = np.nan, np.nan, np.nan
                strict_a = upstream_a = f'ERROR:{str(e)[:40]}'

            rows.append({
                'target': target, 'hops': hop, 'repeat': i, 'scenario': 'tier1_driven',
                'tier1_contrib': t1_a, 'backpressure_contrib': bp_a, 'tier2_contrib': t2_a,
                'strict_correct': strict_a == 'tier1', 'upstream_correct': upstream_a == 'upstream',
            })

            anomaly_row_b = mean_map.copy()
            anomaly_row_b[INJECTION_NODE] = base_val
            anomaly_row_b[target] = mean_map[target] + perturb_std * std_map[target]
            anomaly_df_b = pd.DataFrame([anomaly_row_b])
            try:
                contribs_b = gcm.attribute_anomalies(
                    model, target_node=target, anomaly_samples=anomaly_df_b,
                    attribute_mean_deviation=True, num_distribution_samples=500)
                t1_b, bp_b, t2_b = classify_contribs_v2(contribs_b, target)
                strict_b = 'tier1' if abs(t1_b) > abs(t2_b) else 'tier2'
                upstream_b = 'upstream' if abs(t1_b + bp_b) > abs(t2_b) else 'local'
            except Exception as e:
                t1_b, bp_b, t2_b = np.nan, np.nan, np.nan
                strict_b = upstream_b = f'ERROR:{str(e)[:40]}'

            rows.append({
                'target': target, 'hops': hop, 'repeat': i, 'scenario': 'tier2_driven',
                'tier1_contrib': t1_b, 'backpressure_contrib': bp_b, 'tier2_contrib': t2_b,
                'strict_correct': strict_b == 'tier2', 'upstream_correct': upstream_b == 'local',
            })
        print(f"  [hops={hop}] {target}: done ({n_repeats} repeats x 2 scenarios)")
    return pd.DataFrame(rows)


def main(n_repeats=10):
    print("=" * 80)
    print("  RQ6 diagnostic: TIER1_DRIVEN ACCURACY vs HOP-DISTANCE (Train Ticket)")
    print(f"  Repeats/target/scenario: {n_repeats} | No LLM API calls.")
    print("=" * 80)

    print("\n[Train Ticket] Training accurate-path DAG (CapacityAgent)...")
    cap = CapacityAgent(system_type='trainticket', auto_train=False)
    cap.train_accurate_path()
    model, df_baseline = cap.global_dag_model, cap.global_df_baseline

    all_dfs = []
    for hop, targets in TARGETS_BY_HOP.items():
        df_res = run_tier_test(model, df_baseline, targets, hop, n_repeats=n_repeats)
        all_dfs.append(df_res)

    df = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, 'rq6_hop_distance_diagnostic.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Raw results saved to: {csv_path}")

    t1 = df[df['scenario'] == 'tier1_driven']
    by_hop = t1.groupby('hops').agg(
        n=('strict_correct', 'count'),
        strict_acc_pct=('strict_correct', lambda s: 100.0 * s.mean()),
        upstream_acc_pct=('upstream_correct', lambda s: 100.0 * s.mean()),
        mean_abs_tier1=('tier1_contrib', lambda s: s.abs().mean()),
        mean_abs_backpressure=('backpressure_contrib', lambda s: s.abs().mean()),
        mean_abs_tier2=('tier2_contrib', lambda s: s.abs().mean()),
    ).round(4)
    print("\n[Result] tier1_driven accuracy by hop-distance:")
    print(by_hop.to_string())

    by_target = t1.groupby(['hops', 'target']).agg(
        n=('strict_correct', 'count'),
        strict_acc_pct=('strict_correct', lambda s: 100.0 * s.mean()),
        upstream_acc_pct=('upstream_correct', lambda s: 100.0 * s.mean()),
        mean_tier1=('tier1_contrib', 'mean'),
        mean_tier2=('tier2_contrib', 'mean'),
    ).round(4)
    print("\n[Result] Per-target breakdown:")
    print(by_target.to_string())

    spearman_strict = by_hop.index.to_series().corr(by_hop['strict_acc_pct'], method='spearman')
    spearman_upstream = by_hop.index.to_series().corr(by_hop['upstream_acc_pct'], method='spearman')
    print(f"\nSpearman(hops, strict_acc_pct)   = {spearman_strict}")
    print(f"Spearman(hops, upstream_acc_pct) = {spearman_upstream}")
    print("=" * 80)
    return df, by_hop, by_target


if __name__ == '__main__':
    main()
