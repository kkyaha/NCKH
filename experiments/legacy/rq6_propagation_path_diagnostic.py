# -*- coding: utf-8 -*-
"""
RQ6 diagnostic follow-up #3 — Does the TRUE PROPAGATED signal (how much a
target's own `_workload` node actually moves under the real intervention)
explain the ts-order/ts-travel vs ts-seat/ts-user split, where diagnostic #2
(static coefficient share at each target's own Tier-2 mechanism) did not?
============================================================================
Diagnostic #2 found that ALL FOUR hops=1 targets have a tiny workload-share
(1.5%-10.8%) in their OWN fitted mechanism -- uniform across all four, so it
cannot explain why ts-user/ts-seat succeed at intervention-based attribution
while ts-order/ts-travel fail. This script checks a layer upstream of that:
under do(ts-preserve-service_workload = 2.5x baseline), how far does the
TRUE signal actually travel through the pure Tier-1 (workload->workload)
chain before reaching each target's own `_workload` node -- in (a) graph
hops along the Tier-1-only subgraph, and (b) the actual sigma-shift
(how many of that workload node's own std the intervention moves its mean),
measured directly via gcm.interventional_samples rather than inferred from
static coefficients.

No LLM API calls -- pure SCM/DoWhy computation.
"""

import os
import sys
import warnings

import networkx as nx
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

INJECTION_NODE = 'ts-preserve-service_workload'
TARGETS = {
    'ts-order-service_cpu': 'ts-order-service_workload',
    'ts-travel-service_cpu': 'ts-travel-service_workload',
    'ts-seat-service_cpu': 'ts-seat-service_workload',
    'ts-user-service_cpu': 'ts-user-service_workload',
}


def main(n_repeats=10, dose=2.5):
    print("=" * 80)
    print("  RQ6 diagnostic #3: TRUE PROPAGATED Tier-1 SIGNAL vs attribution outcome")
    print(f"  dose={dose}x baseline | repeats={n_repeats} | No LLM API calls.")
    print("=" * 80)

    print("\n[Train Ticket] Training accurate-path DAG (CapacityAgent)...")
    cap = CapacityAgent(system_type='trainticket', auto_train=False)
    cap.train_accurate_path()
    model, df, g = cap.global_dag_model, cap.global_df_baseline, cap.dag_graph

    # ---- (a) Pure Tier-1 (workload->workload) subgraph path length ----
    tier1_edges = [(u, v) for u, v in g.edges() if u.endswith('_workload') and v.endswith('_workload')]
    g_tier1 = nx.DiGraph(tier1_edges)
    print(f"\n[Tier-1 subgraph] {g_tier1.number_of_nodes()} workload nodes, "
          f"{g_tier1.number_of_edges()} workload->workload edges.")

    base_val = df[INJECTION_NODE].mean()
    intervened_val = base_val * dose
    print(f"[Setup] do({INJECTION_NODE} = {base_val:.4f} -> {intervened_val:.4f}, +{(dose-1)*100:.0f}%)")

    rows = []
    for target_cpu, wl_node in TARGETS.items():
        try:
            tier1_hops = nx.shortest_path_length(g_tier1, INJECTION_NODE, wl_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            tier1_hops = None
        try:
            tier1_path = nx.shortest_path(g_tier1, INJECTION_NODE, wl_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            tier1_path = None

        wl_base_mean = df[wl_node].mean()
        wl_base_std = df[wl_node].std()

        intervened_means = []
        for i in range(n_repeats):
            samples = gcm.interventional_samples(
                model, interventions={INJECTION_NODE: lambda x, w=intervened_val: w},
                num_samples_to_draw=200)
            intervened_means.append(samples[wl_node].mean())
        intervened_mean = float(np.mean(intervened_means))
        intervened_std_of_mean = float(np.std(intervened_means))
        sigma_shift = (intervened_mean - wl_base_mean) / wl_base_std if wl_base_std > 0 else np.nan
        pct_shift = 100.0 * (intervened_mean - wl_base_mean) / wl_base_mean if wl_base_mean != 0 else np.nan

        print(f"\n[{target_cpu}]  own workload node: {wl_node}")
        print(f"  Tier-1-only hops from injection: {tier1_hops}  | path: {tier1_path}")
        print(f"  baseline: mean={wl_base_mean:.4f} std={wl_base_std:.4f}")
        print(f"  intervened mean (over {n_repeats} draws): {intervened_mean:.4f} +/- {intervened_std_of_mean:.4f}")
        print(f"  => sigma-shift = {sigma_shift:.3f} std  |  pct-shift = {pct_shift:.1f}%")

        rows.append({
            'target_cpu': target_cpu, 'workload_node': wl_node, 'tier1_hops': tier1_hops,
            'wl_base_mean': wl_base_mean, 'wl_base_std': wl_base_std,
            'intervened_mean': intervened_mean, 'sigma_shift': sigma_shift, 'pct_shift': pct_shift,
        })

    result = pd.DataFrame(rows)
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results', 'rq6_propagation_path_diagnostic.csv')
    result.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Saved: {csv_path}")
    print("\n[Summary]")
    print(result[['target_cpu', 'tier1_hops', 'sigma_shift', 'pct_shift']].to_string(index=False))
    print("=" * 80)
    return result


if __name__ == '__main__':
    main()
