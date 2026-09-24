# -*- coding: utf-8 -*-
"""
Model Comparison: SCM vs Linear Regression vs Gradient Boosting vs Gaussian Process
=====================================================================================
So sanh 4 mo hinh tren cung bo du lieu va cung phuong phap danh gia (Gold Standard):
  - Train on LOW workload (bottom 67%)
  - Test  on HIGH workload (top 33%) -> simulate "adding new requests"

Metrics: CPU, Memory, Socket, Latency-p50, Latency-p90
Output:  05_model_comparison.csv
"""

import os, sys, warnings, time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from sklearn.linear_model  import LinearRegression, QuantileRegressor
from sklearn.ensemble      import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR  = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))  # core scm lib (data_processor)
from dowhy.gcm import AdditiveNoiseModel, EmpiricalDistribution
from dowhy.gcm.ml import SklearnRegressionModel
from data_processor import load_normal_data, METRICS, SERVICES

N_PROJ = 500

# ----------------------------------------------------------------------------
# FAIR-COMPARISON PROTOCOL (added after the RQ2 methodology audit)
# ----------------------------------------------------------------------------
# Truoc day moi model duoc fit tren mot luong du lieu KHAC NHAU:
#   LinearReg/GradBoost -> toan bo df_train (~43k diem)
#   GaussianProcess     -> 500 diem random
#   SCM                 -> 2000 diem random
# => chenh toi 86x, nen ket luan "SCM thua GradBoost" co the chi la artefact
#    cua cheng lech du lieu chu khong phai cua lop mo hinh. Ngoai ra CSV ghi
#    n_train giong nhau cho ca 4 model nen nhin file khong phat hien duoc.
#
# Tu nay MOI model fit tren DUNG CUNG MOT mau con co dinh (N_FIT, seed co dinh).
# Tran N_FIT bi chan boi GaussianProcess (do phuc tap O(n^3)); 2000 la muc lon
# nhat con chay duoc trong thoi gian hop ly cho ca hai testbed.
N_FIT = 2000
FIT_SEED = 42

MODEL_ORDER = ['LinearReg', 'GradBoost', 'GaussianProcess', 'SCM_Auto', 'SCM_Deployed']

# Hai bien the SCM duoc bao cao SONG SONG, thay vi chon mot:
#   SCM_Auto     = gcm.auto.assign_causal_mechanisms  -> tra loi cau hoi
#                  "cau truc nhan qua co giup gi khong" voi co che linh hoat nhat
#   SCM_Deployed = LinearRegression(positive=True)    -> DUNG co che thuc su
#                  duoc trien khai (capacity_agent.py) va dung trong RQ1
# Truoc day RQ2 chi bao cao SCM_Auto trong khi RQ1/production dung ban constrained,
# khien Bang RQ1 va bang RQ2 bao cao hai SCM khac nhau tren cung he thong
# (vi du Socket Sock Shop: 26.7% o RQ1 vs 14.57% o RQ2).


def subsample_train(df_train, n_fit=N_FIT, seed=FIT_SEED):
    """Mau con dung chung cho MOI model. Tra ve (df, n_hieu_dung)."""
    if len(df_train) <= n_fit:
        return df_train, len(df_train)
    return df_train.sample(n=n_fit, random_state=seed), n_fit

def mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = (yt != 0) & np.isfinite(yt) & np.isfinite(yp)
    return np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100 if m.sum()>0 else float('nan')

def smape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    num = np.abs(yp - yt)
    den = (np.abs(yt) + np.abs(yp)) / 2.0
    m = (den != 0) & np.isfinite(yt) & np.isfinite(yp)
    return np.mean(num[m]/den[m])*100 if m.sum()>0 else float('nan')


# ============================================================
# MODEL DEFINITIONS
# ============================================================
def get_models():
    """Return dict of {model_name: (fit_fn, predict_fn)} compatible functions."""
    return {
        'LinearReg': {
            'type': 'sklearn',
            'model': Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
        },
        'GradBoost': {
            'type': 'sklearn',
            'model': GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                               learning_rate=0.05, random_state=42)
        },
        'GaussianProcess': {
            'type': 'sklearn',
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('gp', GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0)*RBF(1.0)+WhiteKernel(0.1),
                    n_restarts_optimizer=1, alpha=1e-3, normalize_y=True))
            ])
        },
        'SCM_Auto': {
            'type': 'scm_auto',
            'model': None  # built per call
        },
        'SCM_Deployed': {
            'type': 'scm_deployed',
            'model': None  # built per call
        },
    }


def fit_predict_sklearn(model_name, model, df_fit, test_wl_values):
    """Fit sklearn model on the SHARED subsample and predict at given workloads."""
    X_train = df_fit[['Workload']].values
    y_train = df_fit['Target'].values
    model.fit(X_train, y_train)
    preds = model.predict(np.array(test_wl_values).reshape(-1, 1))
    return preds


def _build_scm(df_fit, constrained):
    """Fit a bivariate SCM on the SHARED subsample.

    constrained=False -> gcm.auto (flexible mechanism, 'does causal structure help?')
    constrained=True  -> LinearRegression(positive=True), i.e. the mechanism actually
                         deployed in capacity_agent.py and used by RQ1.
    """
    g = nx.DiGraph(); g.add_edge('Workload', 'Target')
    m = gcm.InvertibleStructuralCausalModel(g)
    if constrained:
        m.set_causal_mechanism('Workload', EmpiricalDistribution())
        m.set_causal_mechanism(
            'Target',
            AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))
    else:
        gcm.auto.assign_causal_mechanisms(m, df_fit)
    gcm.fit(m, df_fit)
    return m


def fit_predict_scm(df_fit, test_wl_values, constrained):
    """Fit SCM and predict bucket means via do(Workload) intervention."""
    m = _build_scm(df_fit, constrained)
    preds = []
    for wlv in test_wl_values:
        dp = gcm.interventional_samples(
            m, interventions={'Workload': lambda x, w=wlv: w},
            num_samples_to_draw=N_PROJ)
        preds.append(dp['Target'].mean())
    return preds, m


# ============================================================
# EVALUATION ENGINE
# ============================================================
def evaluate_all():
    all_records = []

    for metric_name, metric_col, unit, scale in METRICS:
        print(f"\n[{metric_name}]")
        for svc in SERVICES:
            df = load_normal_data(svc, metric_col)
            if df is None or len(df) < 200:
                continue

            # Gold Standard split by workload level
            df = df.sort_values('Workload').reset_index(drop=True)
            split = int(len(df) * 0.67)
            df_train = df.iloc[:split].copy()
            df_test  = df.iloc[split:].copy()

            # Create test buckets
            n_bins = min(8, df_test['Workload'].nunique())
            df_test['bkt'] = pd.qcut(df_test['Workload'], q=n_bins, duplicates='drop')
            bkts = df_test.groupby('bkt', observed=True)[['Workload','Target']].mean()
            test_wl = bkts['Workload'].values
            y_true  = (bkts['Target'] * scale).values

            wl_range_train = f"{df_train['Workload'].min():.1f}-{df_train['Workload'].max():.1f}"
            wl_range_test  = f"{df_test['Workload'].min():.1f}-{df_test['Workload'].max():.1f}"

            # MOT mau con duy nhat, dung chung cho CA 4 model (xem N_FIT o dau file)
            df_fit, n_fit = subsample_train(df_train)

            row_base = {
                'evaluation_protocol': 'OOD_Gold_Standard (Train Low -> Test High)',
                'risk_threshold': 'P80_Percentile',
                'service': svc, 'metric': metric_name, 'unit': unit,
                'n_train': len(df_train), 'n_fit': n_fit, 'n_test': len(df_test),
                'wl_train': wl_range_train, 'wl_test': wl_range_test,
            }

            models_def = get_models()
            model_results = {}

            y_true_full = (df_test['Target'] * scale).values
            wl_full = df_test['Workload'].values

            for model_name, mdef in models_def.items():
                t0 = time.time()
                try:
                    if mdef['type'] == 'sklearn':
                        import copy
                        m_clone = copy.deepcopy(mdef['model'])
                        preds = fit_predict_sklearn(model_name, m_clone, df_fit, test_wl)
                        preds = np.array(preds) * scale
                        # Da fit trong fit_predict_sklearn -> tai su dung de du bao TOAN BO
                        # diem test tho (khong bucket-averaging).
                        preds_full = m_clone.predict(wl_full.reshape(-1, 1)) * scale
                    else:  # SCM_Auto hoac SCM_Deployed -- CUNG df_fit voi cac model tren
                        constrained = (mdef['type'] == 'scm_deployed')
                        preds_raw, scm_fitted = fit_predict_scm(df_fit, test_wl, constrained)
                        preds = np.array(preds_raw) * scale
                        # Voi AdditiveNoiseModel: E[Target|do(Workload=w)] = prediction_model.predict(w)
                        # -> tinh CHINH XAC (khong Monte Carlo) tren toan bo diem test tho.
                        mech = scm_fitted.causal_mechanism('Target')
                        preds_full = mech.prediction_model.predict(wl_full.reshape(-1, 1)).ravel() * scale

                    mp_val = mape(y_true, preds)
                    smp_val = smape(y_true, preds)
                    mae_v  = mean_absolute_error(y_true, preds)
                    rmse_v = np.sqrt(mean_squared_error(y_true, preds))
                    r_range = y_true.max() - y_true.min()
                    nrmse_v = rmse_v / r_range if r_range != 0 else float('nan')
                    r2_v   = r2_score(y_true, preds)

                    # Chi so full-resolution (tren toan bo n_test diem, khong bucket)
                    mp_full  = mape(y_true_full, preds_full)
                    mae_full = mean_absolute_error(y_true_full, preds_full)
                    rmse_full = np.sqrt(mean_squared_error(y_true_full, preds_full))
                    r2_full = r2_score(y_true_full, preds_full)

                    elapsed = time.time() - t0

                    row = row_base.copy()
                    row.update({
                        'model': model_name,
                        'mape_pct': round(mp_val, 2) if not np.isnan(mp_val) else '',
                        'smape_pct': round(smp_val, 2) if not np.isnan(smp_val) else '',
                        'rmse': round(rmse_v, 4),
                        'nrmse': round(nrmse_v, 4) if not np.isnan(nrmse_v) else '',
                        'mae': round(mae_v, 4),
                        'r2': round(r2_v, 3),
                        'mape_full_res_pct': round(mp_full, 2) if not np.isnan(mp_full) else '',
                        'mae_full_res': round(mae_full, 4) if not np.isnan(mae_full) else '',
                        'rmse_full_res': round(rmse_full, 4) if not np.isnan(rmse_full) else '',
                        'r2_full_res': round(r2_full, 3) if not np.isnan(r2_full) else '',
                        'n_test_full_res': len(y_true_full),
                        'train_time_s': round(elapsed, 2)
                    })
                    all_records.append(row)
                    
                    model_results[model_name] = {
                        'mape': mp_val, 'smape': smp_val, 'mae': mae_v, 'rmse': rmse_v, 'nrmse': nrmse_v, 'r2': r2_v, 'time_s': elapsed
                    }

                except Exception as e:
                    model_results[model_name] = {'mape': float('nan'), 'rmse': float('nan'), 'error': str(e)[:60]}
                    all_records.append({**row_base, 'model': model_name,
                                        'mape_pct': float('nan'), 'rmse': float('nan'), 'error': str(e)[:60]})

            # Print summary row
            parts = [f"{svc:<14}"]
            for mn in MODEL_ORDER:
                r = model_results.get(mn, {})
                if 'mape' in r and not np.isnan(r['mape']):
                    tag = '*' if r['mape'] < 10 else ' '
                    parts.append(f"{r['mape']:>6.1f}%{tag}")
                else:
                    parts.append(f"{'ERR':>8}")
            print('  ' + ' | '.join(parts))

    return pd.DataFrame(all_records)


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*90)
    print("  MODEL COMPARISON: SCM vs Linear Regression vs Gradient Boosting vs Gaussian Process")
    print("  Gold Standard: Train on LOW load -> Test on HIGH load (MAPE %)")
    print("="*90)
    print(f"  {'Service':<14} | {'LinearReg':>8} | {'GradBoost':>9} | {'GaussProc':>9} | {'SCM':>8}")
    print("  " + "-"*65)

    df_all = evaluate_all()

    # ---- Summary: Best model per metric (MAPE, RMSE, F1) ----
    print("\n" + "="*95)
    print("  SUMMARY: Average MAPE (%) by Model and Metric")
    print("="*95)
    print(f"  {'Metric':<14} | {'LinearReg':>10} | {'GradBoost':>10} | {'GaussProc':>10} | {'SCM':>10} | {'WINNER'}")
    print("  " + "-"*85)

    for metric_name, _, unit, _ in METRICS:
        sub = df_all[df_all['metric']==metric_name]
        row = f"  {metric_name:<14}"
        best_mape = float('inf')
        best_model = ''
        for mn in MODEL_ORDER:
            s = pd.to_numeric(sub[sub['model']==mn]['mape_pct'], errors='coerce')
            val = s.mean() if not s.empty else float('nan')
            row += f" | {val:>9.1f}%"
            if not np.isnan(val) and val < best_mape:
                best_mape = val
                best_model = mn
        print(row + f" | {best_model}")

    print("\n" + "="*95)
    print("  SUMMARY: Average RMSE by Model and Metric")
    print("="*95)
    print(f"  {'Metric':<14} | {'LinearReg':>10} | {'GradBoost':>10} | {'GaussProc':>10} | {'SCM':>10} | {'WINNER'}")
    print("  " + "-"*85)
    for metric_name, _, unit, _ in METRICS:
        sub = df_all[df_all['metric']==metric_name]
        row = f"  {metric_name:<14}"
        best_rmse = float('inf')
        best_model = ''
        for mn in MODEL_ORDER:
            s = pd.to_numeric(sub[sub['model']==mn]['rmse'], errors='coerce')
            val = s.mean() if not s.empty else float('nan')
            row += f" | {val:>10.4f}"
            if not np.isnan(val) and val < best_rmse:
                best_rmse = val
                best_model = mn
        print(row + f" | {best_model}")

    print("\n" + "="*95)
    print("  SUMMARY: Average SMAPE (%) by Model and Metric")
    print("="*95)
    print(f"  {'Metric':<14} | {'LinearReg':>10} | {'GradBoost':>10} | {'GaussProc':>10} | {'SCM':>10} | {'WINNER'}")
    print("  " + "-"*85)
    for metric_name, _, unit, _ in METRICS:
        sub = df_all[df_all['metric']==metric_name]
        row = f"  {metric_name:<14}"
        best_smape = float('inf')
        best_model = ''
        for mn in MODEL_ORDER:
            s = pd.to_numeric(sub[sub['model']==mn]['smape_pct'], errors='coerce')
            val = s.mean() if not s.empty else float('nan')
            row += f" | {val:>9.1f}%"
            if not np.isnan(val) and val < best_smape:
                best_smape = val
                best_model = mn
        print(row + f" | {best_model}")

    # ---- Per-service winner ----
    print("\n" + "="*90)
    print("  PER-SERVICE: Which model wins most often?")
    print("="*90)
    model_wins = {m: 0 for m in MODEL_ORDER}
    for (svc, metric), grp in df_all.groupby(['service','metric']):
        valid = grp.dropna(subset=['mape_pct'])
        if valid.empty: continue
        winner = valid.loc[valid['mape_pct'].idxmin(), 'model']
        model_wins[winner] = model_wins.get(winner, 0) + 1

    total = sum(model_wins.values())
    for mn, wins in sorted(model_wins.items(), key=lambda x: -x[1]):
        pct = wins/total*100 if total > 0 else 0
        bar = '#' * int(pct/2)
        print(f"  {mn:<18}: {wins:>3} wins ({pct:>4.1f}%)  {bar}")

    # ---- Trade-off analysis ----
    print("\n" + "="*90)
    print("  TRADE-OFF: Accuracy vs Training Speed")
    print("="*90)
    print(f"  {'Model':<20} | {'Avg MAPE':>10} | {'Avg Time(s)':>12} | {'Interpretable':>14} | {'Causal'}")
    print("  " + "-"*80)
    model_props = {
        'LinearReg':       ('Yes (slope)',  'No'),
        'GradBoost':       ('No (black box)','No'),
        'GaussianProcess': ('Yes (kernel)', 'No'),
        'SCM_Auto':        ('Yes (DAG)',    'Yes - do-calculus'),
        'SCM_Deployed':    ('Yes (DAG)',    'Yes - do-calculus'),
    }
    for mn in MODEL_ORDER:
        sub = df_all[df_all['model']==mn]
        avg_mape = pd.to_numeric(sub['mape_pct'], errors='coerce').mean()
        avg_time = pd.to_numeric(sub['train_time_s'], errors='coerce').mean() if 'train_time_s' in sub.columns else float('nan')
        interp, causal = model_props.get(mn, ('?', '?'))
        print(f"  {mn:<20} | {avg_mape:>9.1f}% | {avg_time:>11.2f}s | {interp:<14} | {causal}")

    # Save
    out = os.path.join(OUT_DIR, '05_model_comparison.csv')
    df_all.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    print("="*90)


if __name__ == '__main__':
    main()
