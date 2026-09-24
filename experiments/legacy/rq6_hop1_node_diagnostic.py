# -*- coding: utf-8 -*-
"""
RQ6 diagnostic follow-up #2 — Why do ts-order-service_cpu and
ts-travel-service_cpu specifically fail tier1_driven attribution while
ts-user-service_cpu (same hop distance) does not?
============================================================================
Hypothesis: the fitted NNLS (LinearRegression(positive=True)) mechanism for
each target's own resource node has multiple parents -- its own `_workload`
node (Tier 1) AND every backpressure caller's `_cpu` (added per
Section "backpressure-edge"). With more competing parents, NNLS may assign
the direct `_workload` parent a small coefficient relative to the
backpressure parents (or relative to residual noise variance), starving
the "true" Tier-1 signal even when a real workload intervention is the
actual cause -- independent of hop-distance per se.

This script inspects, for each of the 4 hops=1 targets already tested:
  - how many backpressure-edge parents each target's _cpu node has
    (vs. exactly one `_workload` parent, always present)
  - the fitted NNLS coefficient on each parent (LinearRegression(positive=True)
    -> .coef_, aligned to predecessor order from the causal graph)
  - the fitted model's in-sample R^2 (fit quality -- does a poorly-fit node
    just have more unexplained residual variance, independent of coefficient
    dilution?)

No LLM API calls -- pure SCM/DoWhy computation, reuses the already-trained
Train Ticket accurate-path DAG (same CapacityAgent code path as prior RQ6
scripts).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from capacity_agent import CapacityAgent

TARGETS = ['ts-order-service_cpu', 'ts-travel-service_cpu', 'ts-seat-service_cpu', 'ts-user-service_cpu']


def main():
    print("=" * 80)
    print("  RQ6 diagnostic #2: WHY do ts-order/ts-travel fail while ts-user succeeds?")
    print("  Inspecting fitted NNLS coefficients + fit quality per target.")
    print("=" * 80)

    print("\n[Train Ticket] Training accurate-path DAG (CapacityAgent)...")
    cap = CapacityAgent(system_type='trainticket', auto_train=False)
    cap.train_accurate_path()
    model, df, g = cap.global_dag_model, cap.global_df_baseline, cap.dag_graph

    rows = []
    for target in TARGETS:
        # IMPORTANT: DoWhy's gcm.fit uses sorted(predecessors) (see
        # dowhy.gcm.fitting_sampling.get_ordered_predecessors), NOT
        # networkx's raw insertion-order iteration -- must match exactly or
        # every coefficient below is silently mislabeled.
        parents = sorted(g.predecessors(target))
        workload_parents = [p for p in parents if p.endswith('_workload')]
        backpressure_parents = [p for p in parents if not p.endswith('_workload')]

        mechanism = model.causal_mechanism(target)
        sk_model = mechanism.prediction_model.sklearn_model
        coefs = sk_model.coef_.flatten() if hasattr(sk_model, 'coef_') else None

        X = df[parents].values
        y_true = df[target].values
        y_pred = sk_model.predict(X).flatten()
        r2 = r2_score(y_true, y_pred)

        print(f"\n[{target}]")
        print(f"  Parents ({len(parents)} total: {len(workload_parents)} workload, "
              f"{len(backpressure_parents)} backpressure):")
        for p, c in zip(parents, coefs):
            kind = 'Tier1(workload)' if p.endswith('_workload') else 'backpressure'
            print(f"    {p:35s} [{kind:16s}] coef={c:.5f}  parent_std={df[p].std():.4f}  "
                  f"coef*std={c*df[p].std():.5f}")
        print(f"  In-sample R^2 = {r2:.4f} | target_std = {df[target].std():.4f} "
              f"| target_mean = {df[target].mean():.4f}")

        for p, c in zip(parents, coefs):
            kind = 'workload' if p.endswith('_workload') else 'backpressure'
            rows.append({
                'target': target, 'parent': p, 'kind': kind, 'coef': c,
                'parent_std': df[p].std(), 'effective_scale': c * df[p].std(),
                'n_parents': len(parents), 'n_backpressure': len(backpressure_parents),
                'r2': r2, 'target_std': df[target].std(),
            })

    result = pd.DataFrame(rows)
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results', 'rq6_hop1_node_diagnostic.csv')
    result.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Saved: {csv_path}")

    # Summary: for each target, what fraction of total "effective scale" (coef * parent_std,
    # a rough proxy for how much each parent's typical variation moves the target) comes
    # from the workload parent vs backpressure parents?
    print("\n[Summary] Share of effective scale (coef * parent_std) by kind, per target:")
    summary = result.groupby(['target', 'kind'])['effective_scale'].apply(lambda s: s.abs().sum()).unstack(fill_value=0)
    summary['workload_share_pct'] = 100 * summary.get('workload', 0) / (summary.get('workload', 0) + summary.get('backpressure', 0)).replace(0, np.nan)
    print(summary.to_string())
    print("=" * 80)
    return result, summary


if __name__ == '__main__':
    main()
