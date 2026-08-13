# -*- coding: utf-8 -*-
"""
Multi-Metric SCM Evaluation
============================
Mo rong du doan len toan bo cac chi so co trong dataset:
  - CPU
  - Memory (mem)
  - Socket count
  - Latency-50 (p50)
  - Latency-90 (p90)

Phuong phap "Gold Standard": Train tren WL thap (bottom 67%) -> Test tren WL cao (top 33%)
Danh gia theo: MAE, MAPE, R2
"""

import os, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

warnings.filterwarnings('ignore')
BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'
SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
N_PROJ   = 500

# All metrics to evaluate (target variable, column suffix, display unit, scale)
METRICS = [
    ('CPU',        'cpu',        '%',   1.0),
    ('Memory',     'mem',        'MB',  1/1e6),
    ('Socket',     'socket',     'cnt', 1.0),
    ('Latency-50', 'latency-50', 'ms',  1000.0),
    ('Latency-90', 'latency-90', 'ms',  1000.0),
]


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    m = y_true != 0
    return np.mean(np.abs((y_true[m]-y_pred[m])/y_true[m]))*100 if m.sum()>0 else float('nan')


def load_normal_for_metric(service: str, metric_col: str) -> pd.DataFrame:
    """Load normal-period data efficiently: [Workload, Target]"""
    data_dir = os.path.join(BASE_DIR, 'data', 'raw')
    dfs = []
    wl_col = f'{service}_workload'
    tgt_col = f'{service}_{metric_col}'

    for scenario in os.listdir(data_dir):
        sp = os.path.join(data_dir, scenario)
        if not os.path.isdir(sp): continue
        for run_id in os.listdir(sp):
            rp = os.path.join(sp, run_id)
            if not os.path.isdir(rp): continue
            mp = os.path.join(rp, 'simple_metrics.csv')
            ip = os.path.join(rp, 'inject_time.txt')
            if not os.path.exists(mp) or not os.path.exists(ip): continue
            try:
                with open(ip) as f: it = int(f.read().strip())
                cols = pd.read_csv(mp, nrows=0).columns
                tc = 'imte' if 'imte' in cols else ('time' if 'time' in cols else None)
                if tc is None or wl_col not in cols or tgt_col not in cols: continue

                df_raw = pd.read_csv(mp, usecols=[tc, wl_col, tgt_col])
                df = df_raw[[tc, wl_col, tgt_col]].copy()
                df.columns = [tc, 'Workload', 'Target']
                normal = df[df[tc] < it].drop(columns=[tc]).dropna()
                dfs.append(normal)
            except Exception:
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else None


def build_bivariate_scm(df_train: pd.DataFrame):
    """Bivariate SCM: Workload -> Target"""
    g = nx.DiGraph()
    g.add_edge('Workload', 'Target')
    m = gcm.InvertibleStructuralCausalModel(g)
    gcm.auto.assign_causal_mechanisms(m, df_train)
    gcm.fit(m, df_train)
    return m


def predict_at(model, wl: float) -> float:
    wlc = wl
    df_p = gcm.interventional_samples(
        model, interventions={'Workload': lambda x: wlc}, num_samples_to_draw=N_PROJ)
    return df_p['Target'].mean()


def evaluate(service: str, metric_name: str, metric_col: str, scale: float):
    df = load_normal_for_metric(service, metric_col)
    if df is None or len(df) < 200:
        return None

    # Sort by workload, split 67%/33%
    df = df.sort_values('Workload').reset_index(drop=True)
    split = int(len(df)*0.67)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]

    model = build_bivariate_scm(df_train)

    # Bucket test set into 8 bins
    n_bins = min(8, df_test['Workload'].nunique())
    df_test = df_test.copy()
    df_test['bucket'] = pd.qcut(df_test['Workload'], q=n_bins, duplicates='drop')
    buckets = df_test.groupby('bucket', observed=True)[['Workload','Target']].mean()

    y_true, y_pred = [], []
    for _, row in buckets.iterrows():
        p = predict_at(model, row['Workload'])
        y_true.append(row['Target'])
        y_pred.append(p)

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Risk detection threshold tau = mean + 0.5 * std of training set
    thresh = df_train['Target'].mean() + 0.5 * df_train['Target'].std()
    y_true_bin = (y_true >= thresh).astype(int)
    y_pred_bin = (y_pred >= thresh).astype(int)
    f1_v = f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=1)

    return {
        'service':    service,
        'metric':     metric_name,
        'n_train':    len(df_train),
        'n_test':     len(df_test),
        'wl_train':   f"{df_train['Workload'].min():.1f}-{df_train['Workload'].max():.1f}",
        'wl_test':    f"{df_test['Workload'].min():.1f}-{df_test['Workload'].max():.1f}",
        'mae':        mean_absolute_error(y_true, y_pred) * scale,
        'rmse':       np.sqrt(mean_squared_error(y_true, y_pred)) * scale,
        'f1_score':   round(f1_v, 3),
        'mape':       mape(y_true, y_pred),
        'r2':         r2_score(y_true, y_pred),
        'mean_actual': np.mean(y_true) * scale,
        'mean_pred':   np.mean(y_pred) * scale,
        'model':      model,
        'baseline_wl': df_train['Workload'].mean(),
        'baseline_val': df_train['Target'].mean() * scale,
    }


def main():
    print("="*95)
    print("  MULTI-METRIC GOLD STANDARD EVALUATION")
    print("  Train on LOW load -> Test on HIGH load (actual measurements as ground truth)")
    print("="*95)

    all_results = []
    models_store = {}  # (service, metric) -> result

    for metric_name, metric_col, unit, scale in METRICS:
        print(f"\n  [{metric_name}]", end='')
        for svc in SERVICES:
            r = evaluate(svc, metric_name, metric_col, scale)
            if r is None:
                print(f"  x{svc}", end='')
                continue
            all_results.append(r)
            models_store[(svc, metric_name)] = r
            print(f"  .{svc}", end='')
        print()

    df_res = pd.DataFrame([{k:v for k,v in r.items() if k != 'model'} for r in all_results])

    # ---- REPORT A: Per metric, per service ----
    print("\n" + "="*95)
    print("  REPORT A: MAPE BY SERVICE AND METRIC")
    print("="*95)
    header = f"  {'Service':<14}"
    for mn, _, unit, _ in METRICS:
        header += f" | {mn+' MAPE':>12}"
    print(header)
    print("  " + "-"*85)

    for svc in SERVICES:
        row = f"  {svc:<14}"
        for metric_name, _, unit, _ in METRICS:
            sub = df_res[(df_res['service']==svc) & (df_res['metric']==metric_name)]
            if sub.empty:
                row += f" | {'N/A':>12}"
            else:
                v = sub['mape'].values[0]
                tag = '*' if v < 10 else ' '
                row += f" | {v:>10.1f}%{tag}"
        print(row)

    print("  (* = MAPE < 10%, excellent accuracy)")

    # ---- REPORT B: Per metric summary ----
    print("\n" + "="*95)
    print("  REPORT B: AVERAGE ACCURACY BY METRIC (across all 7 services)")
    print("="*95)
    print(f"  {'Metric':<14} | {'Avg MAPE':>9} | {'Best MAPE':>10} | {'Worst MAPE':>11} | {'Unit'}")
    print("  " + "-"*65)
    for metric_name, _, unit, _ in METRICS:
        sub = df_res[df_res['metric']==metric_name]
        if sub.empty: continue
        print(f"  {metric_name:<14} | {sub['mape'].mean():>8.1f}% | "
              f"{sub['mape'].min():>9.1f}% | {sub['mape'].max():>10.1f}% | {unit}")

    # ---- REPORT C: Workload increase prediction for PLACE_ORDER ----
    print("\n" + "="*95)
    print("  REPORT C: PLACE_ORDER prediction at +20% workload - ALL METRICS")
    print("="*95)
    place_order_svcs = ['front-end','user','catalogue','carts','orders','payment','shipping']

    header2 = f"  {'Service':<14} | {'Base WL':>8}"
    for mn, _, unit, _ in METRICS:
        header2 += f" | {mn+' +20%':>13}"
    print(header2)
    print("  " + "-"*90)

    for svc in place_order_svcs:
        # Get baseline workload from CPU model (most complete)
        key = (svc, 'CPU')
        if key not in models_store: continue
        r_cpu = models_store[key]
        base_wl = r_cpu['baseline_wl']
        new_wl  = base_wl * 1.20

        row = f"  {svc:<14} | {base_wl:>8.2f}"
        for metric_name, metric_col, unit, scale in METRICS:
            k = (svc, metric_name)
            if k not in models_store:
                row += f" | {'N/A':>13}"
                continue
            mr = models_store[k]
            pred_new = predict_at(mr['model'], new_wl)
            base_val = mr['baseline_val']
            new_val  = pred_new * scale
            if base_val != 0:
                chg = (new_val - base_val) / abs(base_val) * 100
                row += f" | {chg:>+11.1f}%"
            else:
                row += f" | {'0':>13}"
        print(row)

    # ---- REPORT D: Detailed per-bucket for best performing services ----
    print("\n" + "="*95)
    print("  REPORT D: BEST ACCURACY HIGHLIGHTS (MAPE < 10%)")
    print("="*95)
    best = df_res[df_res['mape'] < 10].sort_values('mape')
    if best.empty:
        print("  No results with MAPE < 10%")
    else:
        print(f"  {'Service':<14} | {'Metric':<14} | {'MAPE':>8} | {'MAE':>10} | "
              f"{'Mean Actual':>12} | {'Mean Pred':>10}")
        print("  " + "-"*80)
        for _, row in best.iterrows():
            print(f"  {row['service']:<14} | {row['metric']:<14} | {row['mape']:>7.1f}% | "
                  f"{row['mae']:>10.4f} | {row['mean_actual']:>12.4f} | {row['mean_pred']:>10.4f}")

    # Save
    out = os.path.join(BASE_DIR,'data','processed','scm_results','multi_metric_evaluation.csv')
    df_res.drop(columns=['model'], errors='ignore').to_csv(out, index=False)
    print(f"\n  Saved to: {out}")
    print("="*95)


if __name__ == '__main__':
    main()
