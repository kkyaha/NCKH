# -*- coding: utf-8 -*-
"""
ĐÁNH GIÁ SCM TRÊN HỆ THỐNG THỨ HAI: TRAIN TICKET (RQ6 — GENERALIZATION)
========================================================================
Ap dung DUNG methodology da sua/kiem chung cho SockShop (RQ1/RQ2) sang
Train Ticket (28 microservices, RCAEval), thay vi dung lai script cu trong
nhanh TrainTicket (von con loi bucket-averaging + thieu GaussianProcess +
khong co ground-truth direct match + khong kiem dinh thong ke — xem
docs/RQ6_TRAINTICKET_GENERALIZATION_REPORT.md muc gioi han).

Muc dich: RQ6 — SCM co tong quat hoa duoc sang mot he thong Microservices
khac (topology, quy mo khac SockShop) hay khong, dung CUNG mot chuan danh
gia (full-resolution metric, 4-model comparison, kiem dinh theo tung
metric, ground-truth direct match) de so sanh cong bang voi SockShop.

Output:
  data/processed/scm_results/trainticket_f1_rmse_evaluation.csv
  data/processed/scm_results/trainticket_model_comparison.csv
  data/processed/scm_results/trainticket_p_value_statistical_test.csv
  data/processed/scm_results/trainticket_ground_truth_direct_match_summary.csv
"""

import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TT_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'trainticket')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))               # experiments/ (sibling: evaluation_suite)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))    # core scm lib (data_processor)
from data_processor import TRAINTICKET_SERVICES, load_trainticket_data
from evaluation_suite import mape, smape  # dung LAI dung cong thuc da kiem chung, khong viet lai

TT_METRICS = [
    ('CPU', 'cpu', '%', 1.0),
    ('Memory', 'mem', 'MB', 1 / 1e6),
]
N_PROJ = 300
MIN_ROWS = 300


def _load_tt_pair(df_all, service, metric_col):
    wlc, tgc = f'{service}_workload', f'{service}_{metric_col}'
    if wlc not in df_all.columns or tgc not in df_all.columns:
        return None
    df = df_all[[wlc, tgc]].dropna().copy()
    df.columns = ['Workload', 'Target']
    return df if len(df) >= MIN_ROWS else None


def _fit_scm_bivariate(df_train):
    g = nx.DiGraph([('Workload', 'Target')])
    m = gcm.InvertibleStructuralCausalModel(g)
    gcm.auto.assign_causal_mechanisms(m, df_train)
    gcm.fit(m, df_train)
    return m


# ============================================================
# RQ1-tuong-duong: OOD accuracy tren Train Ticket (bucket + full-resolution)
# ============================================================
def run_tt_f1_rmse_benchmark(df_all=None):
    print("=" * 95)
    print("  [TRAIN TICKET] RQ1-tuong-duong: DO CHINH XAC OOD (bucket + full-resolution)")
    print("=" * 95)
    if df_all is None:
        df_all = load_trainticket_data(TT_DATA_DIR)
    if df_all is None or df_all.empty:
        print("  Loi: khong load duoc du lieu Train Ticket. Chay download_trainticket_data.py truoc.")
        return pd.DataFrame()

    rows = []
    for metric_name, metric_col, unit, scale in TT_METRICS:
        for svc in TRAINTICKET_SERVICES:
            df = _load_tt_pair(df_all, svc, metric_col)
            if df is None:
                continue
            df = df.sort_values('Workload').reset_index(drop=True)
            split = int(len(df) * 0.67)
            df_train, df_test = df.iloc[:split], df.iloc[split:]
            if len(df_test) < 30 or df_train['Workload'].nunique() < 3:
                continue

            model = _fit_scm_bivariate(df_train)
            mech = model.causal_mechanism('Target')

            # Full-resolution: cong thuc dong, khong Monte Carlo, tren TOAN BO diem test
            yt_full = df_test['Target'].values * scale
            yp_full = mech.prediction_model.predict(df_test[['Workload']].values).ravel() * scale

            # Bucket-average (de doi chieu voi SockShop — CACH CU, khong dung lam so lieu chinh)
            df_test2 = df_test.copy()
            n_bins = min(8, df_test2['Workload'].nunique())
            df_test2['bkt'] = pd.qcut(df_test2['Workload'], q=n_bins, duplicates='drop')
            bkts = df_test2.groupby('bkt', observed=True)[['Workload', 'Target']].mean()
            yt_bkt, yp_bkt = [], []
            for _, row in bkts.iterrows():
                yt_bkt.append(row['Target'])
                yp_bkt.append(mech.prediction_model.predict(np.array([[row['Workload']]]))[0])
            yt_bkt, yp_bkt = np.array(yt_bkt) * scale, np.array(yp_bkt) * scale

            rows.append({
                'system': 'TrainTicket', 'service': svc, 'metric': metric_name, 'unit': unit,
                'mape_pct': round(mape(yt_bkt, yp_bkt), 2),
                'mape_full_res_pct': round(mape(yt_full, yp_full), 2),
                'rmse': round(np.sqrt(mean_squared_error(yt_bkt, yp_bkt)), 4),
                'rmse_full_res': round(np.sqrt(mean_squared_error(yt_full, yp_full)), 4),
                'r2': round(r2_score(yt_bkt, yp_bkt), 3),
                'r2_full_res': round(r2_score(yt_full, yp_full), 3),
                'n_train': len(df_train), 'n_test_full_res': len(df_test),
                'wl_train_range': f"{df_train['Workload'].min():.1f}-{df_train['Workload'].max():.1f}",
                'wl_test_range': f"{df_test['Workload'].min():.1f}-{df_test['Workload'].max():.1f}",
            })
            print(f"  {svc:<32} | {metric_name:<7} | MAPE full-res={mape(yt_full,yp_full):>7.2f}% | n={len(df_test)}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'trainticket_f1_rmse_evaluation.csv'), index=False)
    print(f"\n  [OK] {len(out)} dong da luu: trainticket_f1_rmse_evaluation.csv")
    return out


# ============================================================
# RQ2-tuong-duong: SCM vs 3 baseline (FULL 4-model, giong SockShop)
# ============================================================
def get_tt_models():
    return {
        'LinearReg': Pipeline([('sc', StandardScaler()), ('reg', LinearRegression())]),
        'GradBoost': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
        'GaussianProcess': Pipeline([
            ('sc', StandardScaler()),
            ('gp', GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
                                             n_restarts_optimizer=2, alpha=1e-3, normalize_y=True))
        ]),
    }


def run_tt_model_comparison(df_all=None):
    print("\n" + "=" * 95)
    print("  [TRAIN TICKET] RQ2-tuong-duong: SCM vs LinearReg vs GradBoost vs GaussianProcess")
    print("=" * 95)
    if df_all is None:
        df_all = load_trainticket_data(TT_DATA_DIR)
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    records = []
    for metric_name, metric_col, unit, scale in TT_METRICS:
        for svc in TRAINTICKET_SERVICES:
            df = _load_tt_pair(df_all, svc, metric_col)
            if df is None:
                continue
            df = df.sort_values('Workload').reset_index(drop=True)
            split = int(len(df) * 0.67)
            df_train, df_test = df.iloc[:split], df.iloc[split:]
            if len(df_test) < 30 or df_train['Workload'].nunique() < 3:
                continue

            X_train = df_train[['Workload']].values
            y_train = df_train['Target'].values
            X_test_full = df_test[['Workload']].values
            y_test_full = df_test['Target'].values * scale

            for model_name, model in get_tt_models().items():
                if model_name == 'GaussianProcess' and len(X_train) > 400:
                    idx = np.random.RandomState(42).choice(len(X_train), 400, replace=False)
                    model.fit(X_train[idx], y_train[idx])
                else:
                    model.fit(X_train, y_train)
                preds = model.predict(X_test_full) * scale
                records.append({
                    'system': 'TrainTicket', 'service': svc, 'metric': metric_name, 'model': model_name,
                    'mape_full_res_pct': round(mape(y_test_full, preds), 2),
                    'rmse_full_res': round(np.sqrt(mean_squared_error(y_test_full, preds)), 4),
                    'r2_full_res': round(r2_score(y_test_full, preds), 3),
                    'n_test_full_res': len(y_test_full),
                })

            # SCM
            scm_model = _fit_scm_bivariate(df_train)
            mech = scm_model.causal_mechanism('Target')
            scm_preds = mech.prediction_model.predict(X_test_full).ravel() * scale
            records.append({
                'system': 'TrainTicket', 'service': svc, 'metric': metric_name, 'model': 'SCM_DoWhy',
                'mape_full_res_pct': round(mape(y_test_full, scm_preds), 2),
                'rmse_full_res': round(np.sqrt(mean_squared_error(y_test_full, scm_preds)), 4),
                'r2_full_res': round(r2_score(y_test_full, scm_preds), 3),
                'n_test_full_res': len(y_test_full),
            })
            print(f"  {svc:<32} | {metric_name:<7} done")

    out = pd.DataFrame(records)
    out.to_csv(os.path.join(OUT_DIR, 'trainticket_model_comparison.csv'), index=False)
    print(f"\n  [OK] {len(out)} dong da luu: trainticket_model_comparison.csv")
    return out


# ============================================================
# Kiem dinh thong ke (giong het cach da sua cho SockShop: tach theo metric, mape_full_res_pct)
# ============================================================
def run_tt_statistical_significance(mc_df=None):
    print("\n" + "=" * 95)
    print("  [TRAIN TICKET] Kiem dinh Wilcoxon/Friedman (tach theo metric, mape_full_res_pct)")
    print("=" * 95)
    if mc_df is None:
        mc_df = pd.read_csv(os.path.join(OUT_DIR, 'trainticket_model_comparison.csv'))

    baselines = ['LinearReg', 'GradBoost', 'GaussianProcess']
    value_col = 'mape_full_res_pct'
    results = []

    def paired(sub, a, b):
        av = sub[sub['model'] == a].set_index(['service', 'metric'])[value_col]
        bv = sub[sub['model'] == b].set_index(['service', 'metric'])[value_col]
        common = av.index.intersection(bv.index)
        return av.loc[common].values, bv.loc[common].values, len(common)

    for metric_name in sorted(mc_df['metric'].unique()):
        sub = mc_df[mc_df['metric'] == metric_name]
        for bl in baselines:
            a, b, n = paired(sub, 'SCM_DoWhy', bl)
            if n >= 2 and not np.allclose(a, b):
                w_stat, w_p = stats.wilcoxon(a, b)
            else:
                w_stat, w_p = float('nan'), float('nan')
            results.append({'comparison': f'SCM_vs_{bl}', 'metric': metric_name, 'n_pairs': n,
                             'median_diff_SCM_minus_baseline': round(float(np.median(a - b)), 3) if n >= 1 else '',
                             'wilcoxon_stat': round(w_stat, 3) if n >= 2 else '',
                             'p_value': round(w_p, 5) if n >= 2 else '',
                             'significant_p_lt_0.05': bool(w_p < 0.05) if n >= 2 else False})

    for bl in baselines:
        a, b, n = paired(mc_df, 'SCM_DoWhy', bl)
        if n >= 2 and not np.allclose(a, b):
            w_stat, w_p = stats.wilcoxon(a, b)
        else:
            w_stat, w_p = float('nan'), float('nan')
        results.append({'comparison': f'SCM_vs_{bl}', 'metric': 'ALL_COMBINED', 'n_pairs': n,
                         'median_diff_SCM_minus_baseline': round(float(np.median(a - b)), 3) if n >= 1 else '',
                         'wilcoxon_stat': round(w_stat, 3) if n >= 2 else '',
                         'p_value': round(w_p, 5) if n >= 2 else '',
                         'significant_p_lt_0.05': bool(w_p < 0.05) if n >= 2 else False})

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUT_DIR, 'trainticket_p_value_statistical_test.csv'), index=False)
    print(out.to_string(index=False))
    print(f"\n  [OK] Da luu: trainticket_p_value_statistical_test.csv")
    return out


# ============================================================
# Ground-truth direct match (giong het compare_with_ground_truth.py, tach theo THOI GIAN trong tung scenario)
# ============================================================
def run_tt_ground_truth_match():
    print("\n" + "=" * 95)
    print("  [TRAIN TICKET] Ground-truth direct match (tach theo thoi gian TRONG TUNG scenario)")
    print("=" * 95)

    scenarios = sorted(d for d in os.listdir(TT_DATA_DIR) if d.startswith('re2tt_'))
    point_rows = []
    for sc in scenarios:
        sc_dir = os.path.join(TT_DATA_DIR, sc)
        mp, ip = os.path.join(sc_dir, 'metrics.parquet'), os.path.join(sc_dir, 'inject_time.txt')
        if not (os.path.exists(mp) and os.path.exists(ip)):
            continue
        with open(ip) as f:
            it = int(f.read().strip())
        df_raw = pd.read_parquet(mp)
        df_normal = df_raw[df_raw['time'] < it]

        for svc in TRAINTICKET_SERVICES:
            for metric_name, metric_col, unit, scale in TT_METRICS:
                wlc, tgc = f'{svc}_workload', f'{svc}_{metric_col}'
                if wlc not in df_normal.columns or tgc not in df_normal.columns:
                    continue
                df_pair = df_normal[[wlc, tgc]].dropna()
                df_pair.columns = ['Workload', tgc]
                if len(df_pair) < 100:
                    continue
                split_idx = int(len(df_pair) * 0.7)
                df_train, df_test = df_pair.iloc[:split_idx], df_pair.iloc[split_idx:]
                if len(df_test) < 10 or df_train['Workload'].nunique() < 3:
                    continue
                try:
                    model = _fit_scm_bivariate(df_train.rename(columns={tgc: 'Target'}))
                    mech = model.causal_mechanism('Target')
                    y_true = df_test[tgc].values
                    y_pred = mech.prediction_model.predict(df_test[['Workload']].values).ravel()
                except Exception:
                    continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    ape = np.where(y_true != 0, np.abs(y_pred - y_true) / np.abs(y_true) * 100.0, np.nan)
                for a in ape:
                    if np.isfinite(a):
                        point_rows.append({'scenario': sc, 'service': svc, 'metric': metric_name, 'abs_pct_error': a})
        print(f"  [{sc}] xong.")

    df_points = pd.DataFrame(point_rows)
    summary = df_points.groupby('metric')['abs_pct_error'].agg(
        n='count', mean='mean', median='median', p90=lambda s: np.percentile(s, 90)).reset_index().round(3)
    summary.to_csv(os.path.join(OUT_DIR, 'trainticket_ground_truth_direct_match_summary.csv'), index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\n  [OK] {len(df_points):,} diem. Da luu: trainticket_ground_truth_direct_match_summary.csv")
    return df_points, summary


if __name__ == '__main__':
    df_all = load_trainticket_data(TT_DATA_DIR)
    run_tt_f1_rmse_benchmark(df_all)
    mc = run_tt_model_comparison(df_all)
    run_tt_statistical_significance(mc)
    run_tt_ground_truth_match()
