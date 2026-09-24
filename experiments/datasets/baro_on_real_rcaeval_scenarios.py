# -*- coding: utf-8 -*-
"""
BARO on RCAEval's real Sock Shop fault-injection scenarios (RE2-SS)
========================================================================
Neither paper_draft.tex's RQ2 (SCM vs. non-causal regression, point-forecast
MAPE) nor the companion xai_attribution_paper_draft.tex's RQ6 (Shapley
attribution faithfulness, synthetic ground-truth-by-construction scenarios)
anchors against any of the 15 baselines RCAEval itself ships. This script
fills that specific gap using BARO's *native* usage pattern -- unlike
experiments/rq6_baro_comparison.py (which fed BARO synthetic single-snapshot
interventional samples to match the companion paper's protocol), here BARO
gets what it was actually designed for: real, timestamped pre/post-injection
telemetry with a real inject_time, from RCAEval's own raw data
(data/raw/RE2-SS/<service>_<faulttype>/<run>/{metrics.csv,inject_time.txt}).

This is the exact same 90-scenario corpus (30 fault types x 3 runs) already
used for RQ1/RQ2's point-forecast accuracy -- so the number below is
directly comparable in *scope* (same benchmark, same scenarios) even though
it answers a different question (service-level root-cause localization,
not point-forecast error).

BARO's core ranking rule is reused verbatim (RCAEval's own e2e/baro.py):
fit a RobustScaler on each column's pre-injection (normal) values, transform
the post-injection (anomalous) values through it, rank columns by maximum
absolute robust z-score. We skip RCAEval's dataset-specific preprocess()
(raw-CSV column renaming for their own pipeline) since we only need the
column's service prefix (text before the first '_', which is safe here --
none of Sock Shop's 5 injected services contain an underscore, only some
sidecar-node column names do, e.g. 'orders-db_...').

Ground truth: the scenario folder name is "<service>_<fault_type>" --
the true injected service is the part before the first underscore.

No LLM API calls, no SCM training -- pure I/O + BARO's own statistic.
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'RE2-SS')


def baro_core(normal_df: pd.DataFrame, anomal_df: pd.DataFrame):
    """RCAEval's e2e/baro.py core ranking rule, copied verbatim."""
    ranks = []
    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()
        if np.all(a == a[0]):  # constant column -> RobustScaler IQR=0, skip
            continue
        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = np.abs(scaler.transform(b.reshape(-1, 1))[:, 0])
        ranks.append((col, float(np.max(zscores))))
    ranks.sort(key=lambda x: x[1], reverse=True)
    return ranks


def run_one_scenario(scenario_dir: str, true_service: str):
    inject_time_path = os.path.join(scenario_dir, 'inject_time.txt')
    metrics_path = os.path.join(scenario_dir, 'metrics.csv')
    if not (os.path.exists(inject_time_path) and os.path.exists(metrics_path)):
        return None

    with open(inject_time_path) as f:
        inject_time = int(f.read().strip())

    df = pd.read_csv(metrics_path)
    if 'time' not in df.columns:
        return None

    normal_df = df[df['time'] < inject_time].drop(columns=['time'])
    anomal_df = df[df['time'] >= inject_time].drop(columns=['time'])
    if len(normal_df) < 5 or len(anomal_df) < 5:
        return None

    ranks = baro_core(normal_df, anomal_df)
    if not ranks:
        return None

    # column -> service prefix (text before first '_')
    ranked_services = []
    for col, _ in ranks:
        svc = col.split('_', 1)[0]
        if svc not in ranked_services:  # first (highest-ranked) occurrence per service
            ranked_services.append(svc)

    top1 = ranked_services[0] == true_service
    top3 = true_service in ranked_services[:3]
    rank_of_true = ranked_services.index(true_service) + 1 if true_service in ranked_services else None

    return {
        'top1': top1, 'top3': top3, 'rank_of_true_service': rank_of_true,
        'n_services_ranked': len(ranked_services), 'top1_predicted': ranked_services[0],
    }


def main():
    print("=" * 80)
    print("  BARO on RCAEval's REAL Sock Shop fault-injection scenarios (RE2-SS)")
    print("  Native usage: real inject_time split, no synthetic samples.")
    print("=" * 80)

    scenario_types = sorted(os.listdir(DATA_DIR))
    rows = []
    for scenario_type in scenario_types:
        type_dir = os.path.join(DATA_DIR, scenario_type)
        if not os.path.isdir(type_dir):
            continue
        true_service, _, fault_type = scenario_type.rpartition('_')
        run_dirs = sorted(glob.glob(os.path.join(type_dir, '*')))
        for run_dir in run_dirs:
            if not os.path.isdir(run_dir):
                continue
            result = run_one_scenario(run_dir, true_service)
            if result is None:
                print(f"  [skip] {scenario_type}/{os.path.basename(run_dir)} -- missing/unusable data")
                continue
            result.update({'scenario': scenario_type, 'true_service': true_service,
                            'fault_type': fault_type, 'run': os.path.basename(run_dir)})
            rows.append(result)
            print(f"  [{scenario_type}/{os.path.basename(run_dir)}] true={true_service:10s} "
                  f"top1_pred={result['top1_predicted']:15s} top1={result['top1']} "
                  f"top3={result['top3']} rank={result['rank_of_true_service']}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results', 'baro_on_real_rcaeval_scenarios.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Saved: {csv_path}  ({len(df)} scenario-runs)")

    print(f"\n[Overall] Top-1 accuracy: {100*df['top1'].mean():.1f}%  |  "
          f"Top-3 accuracy: {100*df['top3'].mean():.1f}%  (n={len(df)})")

    print("\n[By fault type]")
    print(df.groupby('fault_type').agg(
        n=('top1', 'count'),
        top1_pct=('top1', lambda s: 100 * s.mean()),
        top3_pct=('top3', lambda s: 100 * s.mean()),
    ).round(1).to_string())

    print("\n[By true service]")
    print(df.groupby('true_service').agg(
        n=('top1', 'count'),
        top1_pct=('top1', lambda s: 100 * s.mean()),
        top3_pct=('top3', lambda s: 100 * s.mean()),
    ).round(1).to_string())
    print("=" * 80)
    return df


if __name__ == '__main__':
    main()
