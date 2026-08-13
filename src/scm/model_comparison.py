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

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'
OUT_DIR  = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
METRICS  = [
    ('CPU',        'cpu',        '%',   1.0   ),
    ('Memory',     'mem',        'MB',  1/1e6 ),
    ('Socket',     'socket',     'cnt', 1.0   ),
    ('Latency_p50','latency-50', 'ms',  1000.0),
    ('Latency_p90','latency-90', 'ms',  1000.0),
]
N_PROJ = 500

def mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = (yt != 0) & np.isfinite(yt) & np.isfinite(yp)
    return np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100 if m.sum()>0 else float('nan')


# ============================================================
# DATA LOADER
# ============================================================
def load_normal_data(service: str, metric_col: str) -> pd.DataFrame:
    """Load normal-period data efficiently: [Workload, Target]"""
    data_dir = os.path.join(BASE_DIR, 'data', 'raw')
    dfs = []
    wlc = f'{service}_workload'
    tgc = f'{service}_{metric_col}'

    for scenario in os.listdir(data_dir):
        sp = os.path.join(data_dir, scenario)
        if not os.path.isdir(sp): continue
        for run_id in os.listdir(sp):
            rp = os.path.join(sp, run_id)
            if not os.path.isdir(rp): continue
            mp = os.path.join(rp, 'simple_metrics.csv')
            ip = os.path.join(rp, 'inject_time.txt')
            if not (os.path.exists(mp) and os.path.exists(ip)): continue
            try:
                with open(ip) as f: it = int(f.read().strip())
                cols = pd.read_csv(mp, nrows=0).columns
                tc = 'imte' if 'imte' in cols else ('time' if 'time' in cols else None)
                if tc is None or wlc not in cols or tgc not in cols: continue

                df_raw = pd.read_csv(mp, usecols=[tc, wlc, tgc])
                df = df_raw[[tc, wlc, tgc]].copy()
                df.columns = [tc, 'Workload', 'Target']
                dfs.append(df[df[tc] < it].drop(columns=[tc]).dropna())
            except Exception:
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else None


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
                    n_restarts_optimizer=3, alpha=1e-3, normalize_y=True))
            ])
        },
        'SCM_DoWhy': {
            'type': 'scm',
            'model': None  # built per call
        },
    }


def fit_predict_sklearn(model_name, model, df_train, test_wl_values):
    """Fit sklearn model and predict at specified workload values."""
    X_train = df_train[['Workload']].values
    y_train = df_train['Target'].values
    if model_name == 'GaussianProcess' and len(X_train) > 500:
        np.random.seed(42)
        idx = np.random.choice(len(X_train), 500, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    model.fit(X_train, y_train)
    preds = model.predict(np.array(test_wl_values).reshape(-1, 1))
    return preds


def fit_predict_scm(df_train, test_wl_values):
    """Fit DoWhy SCM and predict via do(Workload) intervention."""
    g = nx.DiGraph(); g.add_edge('Workload', 'Target')
    m = gcm.InvertibleStructuralCausalModel(g)
    gcm.auto.assign_causal_mechanisms(m, df_train)
    gcm.fit(m, df_train)
    preds = []
    for wlv in test_wl_values:
        wlc = wlv
        dp = gcm.interventional_samples(
            m, interventions={'Workload': lambda x, w=wlc: w},
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

            row_base = {
                'evaluation_protocol': 'OOD_Gold_Standard (Train Low -> Test High)',
                'risk_threshold': 'P80_Percentile',
                'service': svc, 'metric': metric_name, 'unit': unit,
                'n_train': len(df_train), 'n_test': len(df_test),
                'wl_train': wl_range_train, 'wl_test': wl_range_test,
            }

            models_def = get_models()
            model_results = {}

            for model_name, mdef in models_def.items():
                t0 = time.time()
                try:
                    if mdef['type'] == 'sklearn':
                        import copy
                        m_clone = copy.deepcopy(mdef['model'])
                        preds = fit_predict_sklearn(model_name, m_clone, df_train, test_wl)
                        preds = np.array(preds) * scale
                    else:  # SCM
                        preds_raw, _ = fit_predict_scm(df_train, test_wl)
                        preds = np.array(preds_raw) * scale

                    from sklearn.metrics import f1_score

                    mp_val = mape(y_true, preds)
                    mae_v  = mean_absolute_error(y_true, preds)
                    rmse_v = np.sqrt(mean_squared_error(y_true, preds))
                    r2_v   = r2_score(y_true, preds)

                    # Compute F1 Score for Risk Detection (Threshold = P80 of Full Dataset Distribution)
                    thresh = np.percentile(df['Target'].values * scale, 80)
                    y_true_bin = (y_true >= thresh).astype(int)
                    y_pred_bin = (preds >= thresh).astype(int)
                    f1_v       = f1_score(y_true_bin, y_pred_bin, zero_division=0)

                    elapsed = time.time() - t0

                    model_results[model_name] = {
                        'mape': mp_val, 'mae': mae_v, 'rmse': rmse_v, 'f1': f1_v, 'r2': r2_v, 'time_s': elapsed
                    }
                    all_records.append({
                        **row_base, 'model': model_name,
                        'mape_pct': round(mp_val, 2),
                        'rmse': round(rmse_v, 4),
                        'mae': round(mae_v, 4),
                        'f1_score': round(f1_v, 3),
                        'r2': round(r2_v, 3),
                        'train_time_s': round(elapsed, 2),
                    })
                except Exception as e:
                    model_results[model_name] = {'mape': float('nan'), 'rmse': float('nan'), 'f1': float('nan'), 'error': str(e)[:60]}
                    all_records.append({**row_base, 'model': model_name,
                                        'mape_pct': float('nan'), 'rmse': float('nan'), 'f1_score': float('nan'), 'error': str(e)[:60]})

            # Print summary row
            parts = [f"{svc:<14}"]
            for mn in ['LinearReg','GradBoost','GaussianProcess','SCM_DoWhy']:
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
        for mn in ['LinearReg','GradBoost','GaussianProcess','SCM_DoWhy']:
            s = sub[sub['model']==mn]['mape_pct']
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
        for mn in ['LinearReg','GradBoost','GaussianProcess','SCM_DoWhy']:
            s = sub[sub['model']==mn]['rmse']
            val = s.mean() if not s.empty else float('nan')
            row += f" | {val:>10.4f}"
            if not np.isnan(val) and val < best_rmse:
                best_rmse = val
                best_model = mn
        print(row + f" | {best_model}")

    print("\n" + "="*95)
    print("  SUMMARY: Average F1-Score (Risk Detection) by Model and Metric")
    print("="*95)
    print(f"  {'Metric':<14} | {'LinearReg':>10} | {'GradBoost':>10} | {'GaussProc':>10} | {'SCM':>10} | {'WINNER'}")
    print("  " + "-"*85)
    for metric_name, _, unit, _ in METRICS:
        sub = df_all[df_all['metric']==metric_name]
        row = f"  {metric_name:<14}"
        best_f1 = -1.0
        best_model = ''
        for mn in ['LinearReg','GradBoost','GaussianProcess','SCM_DoWhy']:
            s = sub[sub['model']==mn]['f1_score']
            val = s.mean() if not s.empty else float('nan')
            row += f" | {val:>10.3f}"
            if not np.isnan(val) and val > best_f1:
                best_f1 = val
                best_model = mn
        print(row + f" | {best_model}")

    # ---- Per-service winner ----
    print("\n" + "="*90)
    print("  PER-SERVICE: Which model wins most often?")
    print("="*90)
    model_wins = {m: 0 for m in ['LinearReg','GradBoost','GaussianProcess','SCM_DoWhy']}
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
        'SCM_DoWhy':       ('Yes (DAG)',    'Yes - do-calculus'),
    }
    for mn in ['LinearReg','GradBoost','GaussianProcess','SCM_DoWhy']:
        sub = df_all[df_all['model']==mn]
        avg_mape = sub['mape_pct'].mean()
        avg_time = sub['train_time_s'].mean() if 'train_time_s' in sub.columns else float('nan')
        interp, causal = model_props.get(mn, ('?', '?'))
        print(f"  {mn:<20} | {avg_mape:>9.1f}% | {avg_time:>11.2f}s | {interp:<14} | {causal}")

    # Save
    out = os.path.join(OUT_DIR, '05_model_comparison.csv')
    df_all.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    print("="*90)


if __name__ == '__main__':
    main()
