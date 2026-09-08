# -*- coding: utf-8 -*-
"""
Direct Ground-Truth Matching with RE2-SS Raw Data (FULL AGGREGATED VERSION)
============================================================================
So khop truc tiep du bao cua SCM voi CAC DIEM DO DAC THAT trong TOAN BO
90 (scenario x run) cua bo du lieu RE2-SS, khong chi 2 vi du minh hoa nhu
ban cu. Day la bang chung "ground truth" manh nhat cho bao cao/bai bao vi:

  - Khong bucket-averaging: du bao duoc tinh CHINH XAC (khong Monte Carlo)
    tren TUNG diem workload thuc trong tap held-out 30% cuoi cua moi run,
    dung cong thuc E[Target | do(Workload=w)] = prediction_model.predict(w)
    (dung cho AdditiveNoiseModel, xem giai thich trong evaluation_suite.py).
  - Train/test tach theo THOI GIAN trong CUNG MOT run (70% dau -> 30% cuoi),
    khac voi benchmark chinh (gop nhieu run/scenario roi tach theo quantile
    workload toan cuc) -> day la mot phep kiem tra tinh khai quat DOC LAP,
    khong dung lai tap huan luyen/test cua benchmark chinh.

Output: data/processed/scm_results/ground_truth_direct_match.csv
  (moi dong = 1 diem workload thuc te so khop voi 1 du bao SCM)
Va ban tong hop: data/processed/scm_results/ground_truth_direct_match_summary.csv
  (moi dong = 1 (scenario, service, metric), voi mean/median/p90 % loi)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'RE2-SS')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from evaluation_suite import QueueingLatencyRegressor  # cung mechanism voi Global DAG chinh

SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
METRICS = [
    ('CPU', 'cpu', LinearRegression),
    ('Memory', 'mem', LinearRegression),
    ('Socket', 'socket', LinearRegression),
    ('Latency_p50', 'latency-50', QueueingLatencyRegressor),
]
MIN_ROWS = 60  # can du du lieu de tach 70/30 co y nghia


def _fit_mechanism(df_train, target_col, regressor_cls):
    """Fit 1 SCM bivariate Workload -> target_col, dung DUNG mechanism nhu Global DAG chinh
    (LinearRegression cho CPU/Memory, QueueingLatencyRegressor cho Latency) thay vi
    gcm.auto (vua cham vua co the chon mechanism khac nhau giua cac lan chay)."""
    g = nx.DiGraph()
    g.add_edge('Workload', target_col)
    m = gcm.InvertibleStructuralCausalModel(g)
    m.set_causal_mechanism('Workload', gcm.EmpiricalDistribution())
    m.set_causal_mechanism(target_col, AdditiveNoiseModel(SklearnRegressionModel(regressor_cls())))
    gcm.fit(m, df_train[['Workload', target_col]])
    return m


def run_ground_truth_comparison_all():
    print("=" * 95)
    print("  SO KHOP TRUC TIEP TOAN BO SCM VOI GROUND-TRUTH THUC DO (RE2-SS, khong bucket)")
    print("=" * 95)

    if not os.path.isdir(RAW_DIR):
        print(f"  Loi: khong tim thay {RAW_DIR}")
        return pd.DataFrame(), pd.DataFrame()

    point_rows = []
    n_fits_ok, n_fits_skipped = 0, 0

    scenarios = sorted(d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d)))
    for scenario in scenarios:
        scenario_path = os.path.join(RAW_DIR, scenario)
        run_ids = sorted(d for d in os.listdir(scenario_path) if os.path.isdir(os.path.join(scenario_path, d)))
        for run_id in run_ids:
            run_path = os.path.join(scenario_path, run_id)
            csv_path = os.path.join(run_path, 'simple_metrics.csv')
            inject_path = os.path.join(run_path, 'inject_time.txt')
            if not (os.path.exists(csv_path) and os.path.exists(inject_path)):
                continue
            try:
                with open(inject_path) as f:
                    inject_time = int(f.read().strip())
                df_raw = pd.read_csv(csv_path)
                tc = 'imte' if 'imte' in df_raw.columns else ('time' if 'time' in df_raw.columns else None)
                if tc is None:
                    continue
                df_normal_all = df_raw[df_raw[tc] < inject_time]
            except Exception as e:
                print(f"  [SKIP] {scenario}/{run_id}: khong doc duoc file ({e})")
                continue

            for service in SERVICES:
                wl_col = f'{service}_workload'
                if wl_col not in df_normal_all.columns:
                    continue

                for metric_name, metric_suffix, regressor_cls in METRICS:
                    target_col = f'{service}_{metric_suffix}'
                    if target_col not in df_normal_all.columns:
                        continue

                    df_pair = df_normal_all[[wl_col, target_col]].dropna().copy()
                    df_pair.columns = ['Workload', target_col]
                    if len(df_pair) < MIN_ROWS:
                        n_fits_skipped += 1
                        continue

                    split_idx = int(len(df_pair) * 0.7)
                    df_train = df_pair.iloc[:split_idx]
                    df_test = df_pair.iloc[split_idx:]
                    if len(df_test) < 5 or df_train['Workload'].nunique() < 3:
                        n_fits_skipped += 1
                        continue

                    try:
                        model = _fit_mechanism(df_train, target_col, regressor_cls)
                        mech = model.causal_mechanism(target_col)
                        wl_test = df_test['Workload'].values.reshape(-1, 1)
                        y_true = df_test[target_col].values
                        y_pred = mech.prediction_model.predict(wl_test).ravel()
                    except Exception as e:
                        n_fits_skipped += 1
                        continue

                    n_fits_ok += 1
                    base_wl = df_train['Workload'].mean()
                    with np.errstate(divide='ignore', invalid='ignore'):
                        abs_pct_err = np.where(
                            y_true != 0, np.abs(y_pred - y_true) / np.abs(y_true) * 100.0, np.nan
                        )

                    for wl_v, yt_v, yp_v, ape_v in zip(df_test['Workload'].values, y_true, y_pred, abs_pct_err):
                        point_rows.append({
                            'scenario': scenario,
                            'run_id': run_id,
                            'service': service,
                            'metric': metric_name,
                            'workload_delta_pct_vs_train_mean': round((wl_v - base_wl) / base_wl * 100.0, 2) if base_wl != 0 else np.nan,
                            'actual_workload': round(float(wl_v), 4),
                            'actual_value': round(float(yt_v), 6),
                            'scm_predicted_value': round(float(yp_v), 6),
                            'abs_pct_error': round(float(ape_v), 3) if np.isfinite(ape_v) else np.nan,
                        })

        print(f"  [{scenario}] xu ly xong.")

    if not point_rows:
        print("  Khong co diem du lieu nao duoc so khop — kiem tra lai duong dan du lieu.")
        return pd.DataFrame(), pd.DataFrame()

    df_points = pd.DataFrame(point_rows)
    points_path = os.path.join(OUT_DIR, 'ground_truth_direct_match.csv')
    df_points.to_csv(points_path, index=False)
    print(f"\n  [OK] {len(df_points):,} diem so khop that (khong bucket) da luu: {points_path}")
    print(f"  Fits thanh cong: {n_fits_ok} | Fits bo qua (thieu du lieu): {n_fits_skipped}")

    # ---- Bang tong hop ----
    summary = (
        df_points.dropna(subset=['abs_pct_error'])
        .groupby(['metric'])['abs_pct_error']
        .agg(n='count', mean_abs_pct_error='mean', median_abs_pct_error='median',
             p90_abs_pct_error=lambda s: np.percentile(s, 90))
        .reset_index()
        .round(3)
    )
    summary_by_service = (
        df_points.dropna(subset=['abs_pct_error'])
        .groupby(['service', 'metric'])['abs_pct_error']
        .agg(n='count', mean_abs_pct_error='mean', median_abs_pct_error='median',
             p90_abs_pct_error=lambda s: np.percentile(s, 90))
        .reset_index()
        .round(3)
    )
    summary_path = os.path.join(OUT_DIR, 'ground_truth_direct_match_summary.csv')
    summary_by_service.to_csv(summary_path, index=False)
    print(f"  [OK] Bang tong hop theo (service, metric) da luu: {summary_path}")

    print("\n" + "=" * 95)
    print("  TONG HOP SAI SO %ABS THEO METRIC (GOP TOAN BO SERVICE & SCENARIO)")
    print("=" * 95)
    print(f"  {'Metric':<14} | {'n điểm':>8} | {'Mean |err|%':>12} | {'Median |err|%':>14} | {'P90 |err|%':>10}")
    print("  " + "-" * 70)
    for _, r in summary.iterrows():
        print(f"  {r['metric']:<14} | {r['n']:>8.0f} | {r['mean_abs_pct_error']:>11.2f}% | "
              f"{r['median_abs_pct_error']:>13.2f}% | {r['p90_abs_pct_error']:>9.2f}%")
    print("=" * 95)

    return df_points, summary_by_service


if __name__ == '__main__':
    run_ground_truth_comparison_all()
