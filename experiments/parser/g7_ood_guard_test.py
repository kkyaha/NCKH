# -*- coding: utf-8 -*-
"""
G7 (Proposed Guard) — Does an OOD-Confidence Check Add Value Within G2's Legal Range?
========================================================================================
Question this script answers BEFORE writing any product code: G2 already hard-clamps
the Parser Agent's injection_delta_pct to [5%, 50%] (src/agents/parser_agent.py,
MIN_DELTA_PCT/MAX_DELTA_PCT) regardless of what the LLM proposes. So before wiring an
OOD-confidence guard (G7) into orchestrator.py, we need to know whether extrapolation
risk is actually reachable INSIDE that already-guarded [5,50]% range at all -- if it is
not (e.g. because every downstream node's training distribution comfortably covers a
50% gateway-workload increase), G7 would rarely or never fire and adds no real value.
If it IS reachable -- e.g. because downstream (Tier-1 propagated or Tier-2 converted)
nodes can land in OOD territory even when the injection point itself is a legal,
guard-approved delta -- that is a genuine, previously-undocumented extension of the
"Extrapolation-Sign Failure Mode" already in the paper's Discussion section, and
justifies wiring G7 for real.

Method: sweep the gateway injection delta over the FULL legal range G2 already allows
(5% to 50%, step 5), run a real do(gateway_workload = baseline*(1+delta/100))
intervention through the same fitted DAGs used elsewhere in this codebase, and for
every downstream node compare its projected value against ITS OWN training-distribution
percentiles (P75/P90/P95/P99, 2x max) -- the same logic as experiments/future_rca.py's
OODGuard, reimplemented here without that module's inline literature-citation strings
(those are not something we can vouch for in a product-facing report).

Runs on both Sock Shop (evaluation_suite's 28-node DAG) and Train Ticket (CapacityAgent's
accurate path) for the same cross-topology discipline as the rest of this codebase --
though note only Sock Shop's Parser Agent is wired with a single hardcoded gateway
('front-end'); Train Ticket is tested here for DAG-level generalization only, not
because G7 as scoped would currently apply to it in the live product.

No LLM API calls.
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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# G2's own legal range (src/agents/parser_agent.py MIN_DELTA_PCT/MAX_DELTA_PCT) --
# a delta outside [5,50] can never reach the SCM in the live product, so we never
# test outside it: that would misrepresent what G7 could ever actually see.
LEGAL_DELTAS = list(range(5, 51, 5))


def compute_train_stats(df_train: pd.DataFrame) -> dict:
    """Per-node training-distribution percentiles -- same fields as future_rca.OODGuard,
    minus the literature-citation strings in its caveat text (unverifiable in this context)."""
    stats = {}
    for col in df_train.columns:
        vals = df_train[col].dropna().values
        if len(vals) == 0:
            continue
        stats[col] = {
            'max': np.max(vals),
            'p90': np.percentile(vals, 90),
            'p95': np.percentile(vals, 95),
            'p99': np.percentile(vals, 99),
        }
    return stats


def classify_ood(value: float, s: dict) -> str:
    if value <= s['p90']:
        return 'high'
    elif value <= s['p95']:
        return 'medium'
    elif value <= s['p99']:
        return 'low'
    elif value <= 2 * s['max']:
        return 'low'
    else:
        return 'very_low'


def run_sweep(model, df_train, gateway_node, label, n_samples=300):
    stats = compute_train_stats(df_train)
    base_val = df_train[gateway_node].mean()
    downstream_nodes = [n for n in df_train.columns if n != gateway_node and n in stats]

    rows = []
    for delta in LEGAL_DELTAS:
        target_val = base_val * (1 + delta / 100.0)
        samples = gcm.interventional_samples(
            model, interventions={gateway_node: lambda x, w=target_val: w},
            num_samples_to_draw=n_samples)
        for node in downstream_nodes:
            if node not in samples.columns:
                continue
            proj_mean = samples[node].mean()
            conf = classify_ood(proj_mean, stats[node])
            rows.append({
                'label': label, 'delta_pct': delta, 'node': node,
                'tier': 'tier1' if node.endswith('_workload') else 'tier2',
                'projected_mean': proj_mean, 'train_p95': stats[node]['p95'],
                'train_max': stats[node]['max'], 'ood_confidence': conf,
            })
        print(f"  [{label}] delta={delta:>3}% done ({len(downstream_nodes)} downstream nodes)")
    return pd.DataFrame(rows)


def main():
    print("=" * 90)
    print("  G7 FEASIBILITY TEST: OOD risk WITHIN G2's already-guarded [5,50]% range")
    print("=" * 90)

    all_dfs = []

    print("\n[Sock Shop] Training 28-node DAG...")
    ss_model, ss_df, _ = build_and_train_global_dag()
    ss_res = run_sweep(ss_model, ss_df, 'front-end_workload', 'SockShop')
    all_dfs.append(ss_res)

    print("\n[Train Ticket] Training accurate-path DAG...")
    cap = CapacityAgent(system_type='trainticket', auto_train=False)
    cap.train_accurate_path()
    tt_res = run_sweep(cap.global_dag_model, cap.global_df_baseline,
                        'ts-preserve-service_workload', 'TrainTicket')
    all_dfs.append(tt_res)

    df = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, 'g7_ood_feasibility.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Raw results saved: {csv_path}")

    # Summary: fraction of downstream nodes at each confidence level, per delta, per system
    summary = (df.groupby(['label', 'delta_pct', 'ood_confidence'])
               .size().unstack(fill_value=0))
    summary_pct = summary.div(summary.sum(axis=1), axis=0) * 100.0
    print("\n[Result] % of downstream nodes at each OOD-confidence level, per legal delta:")
    print(summary_pct.round(1).to_string())

    # Headline: does ANY node ever leave 'high' confidence within the legal range?
    any_non_high = df[df['ood_confidence'] != 'high']
    print(f"\n[Headline] Non-'high'-confidence (node, delta) pairs within G2's legal "
          f"range: {len(any_non_high)} / {len(df)} ({100*len(any_non_high)/len(df):.1f}%)")
    if len(any_non_high) > 0:
        print("First few such cases:")
        print(any_non_high.sort_values('delta_pct').head(10).to_string(index=False))

    summary_path = os.path.join(OUTPUT_DIR, 'g7_ood_feasibility_summary.csv')
    summary_pct.round(2).to_csv(summary_path)
    print(f"[OK] Summary saved: {summary_path}")
    print("=" * 90)
    return df, summary_pct


if __name__ == '__main__':
    main()
