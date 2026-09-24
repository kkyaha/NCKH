# -*- coding: utf-8 -*-
"""
RQ6 (Part A.3) — Does Shapley Attribution Respect Graph Topology?
====================================================================
The deferred check from RQ6 Part A (docs/HE_THONG.md (muc 6)):
Sock Shop's 7-service graph is too small and too densely connected from its
single gateway (front-end) for a clean "this node is genuinely NOT
downstream of the injection point" test — nearly every node is reachable.
Train Ticket's 28-service graph is not: from the natural gateway
ts-preserve-service, the trained accurate-path DAG has 17 reachable
downstream services and 10 genuinely unreachable ones (no path exists at
all in the fitted causal graph).

This gives a clean, principled validity test: intervening on
ts-preserve-service_workload should not change the fitted distribution of
an unreachable service's CPU at all (it isn't even an ancestor in the
graph), so a well-behaved attribution mechanism should flag it "normal" at
close to the same rate as the pure null condition (RQ6 Part A.2) — not
because we are telling it to, but because nothing in that node's causal
ancestry changed. A mechanism that fires anyway on unreachable nodes would
be attributing along something other than the graph structure the paper's
whole argument rests on.

No LLM API calls — pure SCM/DoWhy computation.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from capacity_agent import CapacityAgent
from future_rca import FutureRCAEngine

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

INJECTION_SERVICE = 'ts-preserve-service'
INJECTION_NODE = f'{INJECTION_SERVICE}_workload'


def run_rq6_part_a3(n_repeats=10, intervention_pct=150.0):
    for arg in sys.argv:
        if arg.startswith('--repeats='):
            n_repeats = int(arg.split('=', 1)[1])
        if arg.startswith('--intervention-pct='):
            intervention_pct = float(arg.split('=', 1)[1])

    print("=" * 80)
    print("   RQ6 (Part A.3): DOES SHAPLEY ATTRIBUTION RESPECT GRAPH TOPOLOGY?")
    print(f"   Testbed: Train Ticket | Injection: {INJECTION_SERVICE} "
          f"| Intervention: +{intervention_pct}% | Repeats: {n_repeats}")
    print("   No LLM API calls — pure SCM/DoWhy computation.")
    print("=" * 80)

    print(f"\n[Setup] Training Train Ticket accurate-path DAG "
          f"(CapacityAgent, accurate path only — fast path not needed here)...")
    cap = CapacityAgent(system_type='trainticket', auto_train=False)
    cap.train_accurate_path()
    if cap.global_dag_model is None:
        raise RuntimeError("Train Ticket accurate-path DAG failed to train — check data/raw/trainticket.")

    hops = cap._compute_hops(INJECTION_SERVICE)
    reachable = sorted([s for s, h in hops.items() if h >= 1])       # excludes injection itself (h=0)
    unreachable = sorted([s for s, h in hops.items() if h == -1])
    print(f"[Setup] {len(reachable)} reachable services (hops >=1): {reachable}")
    print(f"[Setup] {len(unreachable)} unreachable services (no path in fitted DAG): {unreachable}")

    engine = FutureRCAEngine(cap.global_dag_model, cap.global_df_baseline, cap.dag_graph)
    base_wl = cap.global_df_baseline[INJECTION_NODE].mean()
    wl = base_wl * (1.0 + intervention_pct / 100.0)
    print(f"[Setup] Baseline {INJECTION_NODE}={base_wl:.3f} -> intervened {wl:.3f} "
          f"(+{intervention_pct}%)")

    print(f"\nRunning do({INJECTION_NODE}={wl:.2f}) x {n_repeats} independent repeats...")
    rows = []
    for i in range(n_repeats):
        result = engine.analyze(f"TOPO-{i}", {INJECTION_NODE: wl})
        for nr in result.node_risks:
            svc = nr.node.rsplit('_cpu', 1)[0]
            h = hops.get(svc, None)
            group = (
                'injection_self' if h == 0 else
                'reachable' if (h is not None and h >= 1) else
                'unreachable' if h == -1 else
                'unknown'
            )
            rows.append({
                'repeat': i, 'service': svc, 'hops': h, 'group': group,
                'change_pct': nr.change_pct, 'z_score': nr.z_score,
                'anomaly_score': nr.anomaly_score,
                'shapley_contribution': nr.shapley_contribution,
                'risk_level': nr.risk_level,
            })
        print(f"  [repeat {i+1}/{n_repeats}] done "
              f"({sum(1 for r in rows if r['repeat']==i)} CPU nodes scored)")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'rq6_topology_check.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Raw results saved to: {csv_path}")

    # ================================================================
    # ANALYSIS
    # ================================================================
    group_summary = df.groupby('group').agg(
        n_service_repeats=('service', 'count'),
        mean_shapley=('shapley_contribution', 'mean'),
        mean_anomaly=('anomaly_score', 'mean'),
        mean_change_pct=('change_pct', lambda s: s.abs().mean()),
        flagged_rate_pct=('risk_level', lambda s: 100.0 * (s != 'normal').mean()),
    ).round(4)
    print("\n[Result] Summary by topological group:")
    print(group_summary.to_string())

    # Precision@K: rank ALL (service, repeat) rows by shapley_contribution
    # descending; among the top-K where K = number of reachable services,
    # what fraction are actually reachable (vs unreachable / self)?
    k = len(reachable)
    precisions = []
    for i in range(n_repeats):
        rep_df = df[df['repeat'] == i].copy()
        rep_df = rep_df[rep_df['group'].isin(['reachable', 'unreachable'])]  # exclude injection's own cpu
        top_k = rep_df.nlargest(k, 'shapley_contribution')
        prec = (top_k['group'] == 'reachable').mean()
        precisions.append(prec)
    precision_at_k = 100.0 * np.mean(precisions)
    print(f"\n[Result] Precision@{k} (of the top-{k} Shapley-ranked services per repeat, "
          f"% that are genuinely reachable): {precision_at_k:.1f}% "
          f"(mean over {n_repeats} repeats; chance level given "
          f"{len(reachable)} reachable / {len(reachable)+len(unreachable)} total = "
          f"{100.0*k/(len(reachable)+len(unreachable)):.1f}%)")

    summary_csv = os.path.join(OUTPUT_DIR, 'rq6_topology_check_summary.csv')
    group_summary.to_csv(summary_csv, encoding='utf-8')

    # ================================================================
    # APPEND TO THE RQ6 MARKDOWN REPORT
    # ================================================================
    addendum = f"""

---

# RQ6 Part A.3 — Does Shapley Attribution Respect Graph Topology? (Train Ticket)

Testbed: Train Ticket accurate-path DAG (112 nodes, 150 edges after filtering
to the 28 named services). Injection: `{INJECTION_SERVICE}` (+{intervention_pct}%),
{n_repeats} independent repeats. From this gateway, the fitted DAG has
**{len(reachable)} reachable services** (hops 1-3: {reachable}) and
**{len(unreachable)} genuinely unreachable services** (no path at all:
{unreachable}) — Sock Shop's 7-node graph is too small/densely connected for
this test, which is why it was deferred there.

## Summary by topological group

```
{group_summary.to_string()}
```

## Precision@{k}

Of the top-{k} Shapley-ranked service CPU nodes per repeat (excluding
`{INJECTION_SERVICE}`'s own CPU), the fraction that are genuinely reachable
from the injection point: **{precision_at_k:.1f}%** (mean over {n_repeats}
repeats), against a **{100.0*k/(len(reachable)+len(unreachable)):.1f}%**
chance baseline if Shapley ranked services at random with respect to
topology.

## Data
* `data/processed/scm_results/rq6_topology_check.csv` — {len(df)} rows ({n_repeats} repeats x {len(reachable)+len(unreachable)+1} CPU nodes).
* `data/processed/scm_results/rq6_topology_check_summary.csv` — grouped summary.
"""
    md_path = os.path.join(DOCS_DIR, 'docs/HE_THONG.md (muc 6)')
    with open(md_path, 'a', encoding='utf-8') as f:
        f.write(addendum)
    print(f"\n[OK] Appended Part A.3 results to: {md_path}")
    print("=" * 80)
    return df, group_summary, precision_at_k


if __name__ == '__main__':
    run_rq6_part_a3()
