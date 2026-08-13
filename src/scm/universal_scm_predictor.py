# -*- coding: utf-8 -*-
"""
Gold Standard Evaluation: "What happens when we add more requests?"
===================================================================
Phuong phap dung de chung minh do chinh xac SCM cho bai toan them yeu cau:

  1. Load toan bo du lieu BINH THUONG tu 90 runs.
  2. Chia theo MUC TAI (workload quantile), khong phai theo thoi gian.
     - Nhom LOW  = Workload < P33  (tai thap)
     - Nhom MED  = P33..P66        (tai trung binh)  
     - Nhom HIGH = Workload > P66  (tai cao = "da them request")
  3. Train SCM chi tren nhom LOW+MED.
  4. Du doan CPU/Latency tai cac muc workload cua nhom HIGH.
  5. So sanh prediction vs ACTUAL measurements tai cung muc workload do.
  -> Day chinh xac la kich ban "them 1 yeu cau moi" vi:
     * SCM chi hoc tu nhat de thap
     * Test: lieu no co du doan dung duoc nhat de cao khong?

Ket qua:
  - Scatter plot: Predicted vs Actual (dung workload thuc te lam truc X)
  - MAE, RMSE, MAPE, R2 
  - Bang du doan tang tai theo CALL CHAIN (GET_CATALOGUE, PLACE_ORDER, ...)
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

warnings.filterwarnings('ignore')

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'
SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
N_PROJ   = 500

CALL_CHAINS = {
    'GET_CATALOGUE':       ['front-end', 'catalogue'],
    'ADD_TO_CART':         ['front-end', 'catalogue', 'carts'],
    'VIEW_CART':           ['front-end', 'carts'],
    'REGISTER':            ['front-end', 'user'],
    'LOGIN':               ['front-end', 'user'],
    'PLACE_ORDER':         ['front-end', 'user', 'catalogue', 'carts', 'orders', 'payment', 'shipping'],
    # --- TÍNH NĂNG MỚI (NEW HYPOTHETICAL FEATURES) ---
    'APPLY_PROMO_CODE':    ['front-end', 'carts', 'orders', 'payment'],
    'RECOMMEND_PRODUCTS':  ['front-end', 'user', 'catalogue', 'orders'],
    'TRACK_PACKAGE':       ['front-end', 'orders', 'shipping'],
    'WRITE_PRODUCT_REVIEW':['front-end', 'user', 'catalogue'],
}

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    m = y_true != 0
    return np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100 if m.sum() > 0 else float('nan')


# ============================================================
# 1. Load ALL normal data from 90 runs
# ============================================================
def load_all_normal(service: str) -> pd.DataFrame:
    data_dir = os.path.join(BASE_DIR, 'data', 'raw')
    dfs = []
    for scenario in os.listdir(data_dir):
        sp = os.path.join(data_dir, scenario)
        if not os.path.isdir(sp): continue
        for run_id in os.listdir(sp):
            rp = os.path.join(sp, run_id)
            if not os.path.isdir(rp): continue
            mp = os.path.join(rp, 'simple_metrics.csv')
            ip = os.path.join(rp, 'inject_time.txt')
            if not os.path.exists(mp) or not os.path.exists(ip): continue
            with open(ip) as f:
                it = int(f.read().strip())
            df_raw = pd.read_csv(mp)
            tc = 'imte' if 'imte' in df_raw.columns else 'time'
            wl, cpu, lat = f'{service}_workload', f'{service}_cpu', f'{service}_latency-50'
            if not all(c in df_raw.columns for c in [wl, cpu, lat]): continue
            df = df_raw[[tc, wl, cpu, lat]].copy()
            df.columns = [tc, 'Workload', 'CPU', 'Latency']
            normal = df[df[tc] < it].drop(columns=[tc]).dropna()
            dfs.append(normal)
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================
# 2. Train SCM
# ============================================================
def build_scm(df_train: pd.DataFrame):
    g = nx.DiGraph()
    g.add_edges_from([('Workload','CPU'),('CPU','Latency'),('Workload','Latency')])
    m = gcm.InvertibleStructuralCausalModel(g)
    gcm.auto.assign_causal_mechanisms(m, df_train)
    gcm.fit(m, df_train)
    return m


# ============================================================
# 3. Predict at a workload level
# ============================================================
def predict_at(model, wl: float) -> tuple:
    wlc = wl
    df_p = gcm.interventional_samples(
        model,
        interventions={'Workload': lambda x: wlc},
        num_samples_to_draw=N_PROJ
    )
    return df_p['CPU'].mean(), df_p['Latency'].mean()


# ============================================================
# 4. Gold Standard Evaluation
# ============================================================
def evaluate_service(service: str, df_all: pd.DataFrame) -> dict:
    """
    Split by workload quantile:
      - Train on LOW + MED (bottom 66%)
      - Test  on HIGH      (top 33% = 'added load' scenario)
    """
    df = df_all.copy().sort_values('Workload').reset_index(drop=True)
    split_idx = int(len(df) * 0.67)

    df_train = df.iloc[:split_idx]        # LOW + MED workload
    df_test  = df.iloc[split_idx:].copy() # HIGH workload (the "added request" zone)

    # Workload range
    wl_train_max = df_train['Workload'].max()
    wl_test_min  = df_test['Workload'].min()
    wl_test_max  = df_test['Workload'].max()

    # Train SCM on lower-load data
    model = build_scm(df_train)

    # Bucket test set by workload (10 bins)
    df_test['bucket'] = pd.qcut(df_test['Workload'], q=min(10, df_test['Workload'].nunique()),
                                 duplicates='drop')
    buckets = df_test.groupby('bucket', observed=True)[['Workload','CPU','Latency']].mean()

    y_true_cpu, y_pred_cpu = [], []
    y_true_lat, y_pred_lat = [], []
    rows = []

    for _, row in buckets.iterrows():
        pc, pl = predict_at(model, row['Workload'])
        y_true_cpu.append(row['CPU'])
        y_pred_cpu.append(pc)
        y_true_lat.append(row['Latency'])
        y_pred_lat.append(pl)
        rows.append({
            'workload':   round(row['Workload'], 3),
            'actual_cpu': round(row['CPU'], 4),
            'pred_cpu':   round(pc, 4),
            'cpu_err%':   round(abs(row['CPU']-pc)/row['CPU']*100 if row['CPU']!=0 else 0, 1),
            'actual_lat_ms': round(row['Latency']*1000, 2),
            'pred_lat_ms':   round(pl*1000, 2),
            'lat_err%':   round(abs(row['Latency']-pl)/row['Latency']*100 if row['Latency']!=0 else 0, 1),
        })

    cpu_thresh = df_train['CPU'].mean() + 0.5 * df_train['CPU'].std()
    lat_thresh = df_train['Latency'].mean() + 0.5 * df_train['Latency'].std()
    cpu_yt_bin = (np.array(y_true_cpu) >= cpu_thresh).astype(int)
    cpu_yp_bin = (np.array(y_pred_cpu) >= cpu_thresh).astype(int)
    cpu_f1 = f1_score(cpu_yt_bin, cpu_yp_bin, average='binary', zero_division=1)

    lat_yt_bin = (np.array(y_true_lat) >= lat_thresh).astype(int)
    lat_yp_bin = (np.array(y_pred_lat) >= lat_thresh).astype(int)
    lat_f1 = f1_score(lat_yt_bin, lat_yp_bin, average='binary', zero_division=1)

    return {
        'service':       service,
        'n_train':       len(df_train),
        'n_test':        len(df_test),
        'wl_train_range': f'{df_train["Workload"].min():.2f}–{wl_train_max:.2f}',
        'wl_test_range':  f'{wl_test_min:.2f}–{wl_test_max:.2f}',
        'cpu_mae':  mean_absolute_error(y_true_cpu, y_pred_cpu),
        'cpu_rmse': np.sqrt(mean_squared_error(y_true_cpu, y_pred_cpu)),
        'cpu_f1':   round(cpu_f1, 3),
        'cpu_mape': mape(y_true_cpu, y_pred_cpu),
        'cpu_r2':   r2_score(y_true_cpu, y_pred_cpu),
        'lat_mae':  mean_absolute_error(y_true_lat, y_pred_lat),
        'lat_rmse': np.sqrt(mean_squared_error(y_true_lat, y_pred_lat)),
        'lat_f1':   round(lat_f1, 3),
        'lat_mape': mape(y_true_lat, y_pred_lat),
        'lat_r2':   r2_score(y_true_lat, y_pred_lat),
        'detail':   pd.DataFrame(rows),
        'model':    model,
        'baseline_wl':  df_train['Workload'].mean(),
        'baseline_cpu': df_train['CPU'].mean(),
        'baseline_lat': df_train['Latency'].mean(),
    }


# ============================================================
# 5. Main
# ============================================================
def main():
    print("=" * 90)
    print("  GOLD STANDARD EVALUATION: 'ADDING A NEW REQUEST' SCENARIO")
    print("  Train on LOW load -> Test predictions at HIGH load (actual measurements as ground truth)")
    print("=" * 90)

    all_results = {}

    for svc in SERVICES:
        print(f"\n  Loading [{svc}]...", end=' ')
        df = load_all_normal(svc)
        if df is None:
            print("NO DATA")
            continue
        print(f"{len(df):,} samples | WL range: {df['Workload'].min():.2f}–{df['Workload'].max():.2f}")
        r = evaluate_service(svc, df)
        all_results[svc] = r

    print("\n" + "=" * 90)
    print("  REPORT A: Per-Service Accuracy (SCM trained on LOW load, tested on HIGH load)")
    print("=" * 90)
    print(f"  {'Service':<14} | {'Train WL':>12} | {'Test WL':>12} | {'CPU MAPE':>9} | {'CPU R²':>7} | {'Lat MAPE':>9} | {'Lat R²':>7}")
    print("  " + "-" * 80)
    for svc, r in all_results.items():
        print(f"  {svc:<14} | {r['wl_train_range']:>12} | {r['wl_test_range']:>12} | "
              f"{r['cpu_mape']:>8.1f}% | {r['cpu_r2']:>7.3f} | "
              f"{r['lat_mape']:>8.1f}% | {r['lat_r2']:>7.3f}")

    print("\n" + "=" * 90)
    print("  REPORT B: Bucket-by-Bucket Detail for FRONT-END (Best service)")
    print("=" * 90)
    if 'front-end' in all_results:
        detail = all_results['front-end']['detail']
        print(f"  {'WL (req/s)':>10} | {'Act CPU':>8} | {'Pred CPU':>9} | {'Err%':>5} | "
              f"{'Act Lat(ms)':>11} | {'Pred Lat(ms)':>12} | {'Err%':>5}")
        print("  " + "-" * 75)
        for _, row in detail.iterrows():
            print(f"  {row['workload']:>10.2f} | {row['actual_cpu']:>8.4f} | {row['pred_cpu']:>9.4f} | "
                  f"{row['cpu_err%']:>4.1f}% | {row['actual_lat_ms']:>11.2f} | "
                  f"{row['pred_lat_ms']:>12.2f} | {row['lat_err%']:>4.1f}%")

    print("\n" + "=" * 90)
    print("  REPORT C: Per Request Type - Accuracy of 'adding N% more requests'")
    print("=" * 90)
    for req_type, services in CALL_CHAINS.items():
        svc_results = [all_results[s] for s in services if s in all_results]
        if not svc_results: continue
        avg_cpu_mape = np.mean([r['cpu_mape'] for r in svc_results])
        avg_lat_mape = np.mean([r['lat_mape'] for r in svc_results])
        avg_cpu_r2   = np.mean([r['cpu_r2']   for r in svc_results])
        print(f"\n  [{req_type}] Blast Radius: {services}")
        print(f"    Average CPU MAPE = {avg_cpu_mape:.1f}%  |  CPU R² = {avg_cpu_r2:.3f}  |  Lat MAPE = {avg_lat_mape:.1f}%")

    print("\n" + "=" * 90)
    print("  REPORT D: PLACE_ORDER - Predicted impact at +10%, +20%, +30% workload increase")
    print("  (Using SCM trained on LOW load only — true out-of-distribution test)")
    print("=" * 90)
    print(f"\n  {'Service':<14} | {'Base WL':>8} | {'+10%':>12} | {'+20%':>12} | {'+30%':>12}")
    print("  " + "-" * 70)
    for svc in CALL_CHAINS['PLACE_ORDER']:
        if svc not in all_results: continue
        r = all_results[svc]
        model = r['model']
        bwl   = r['baseline_wl']
        bcpu  = r['baseline_cpu']
        preds = []
        for pct in [10, 20, 30]:
            pc, pl = predict_at(model, bwl * (1 + pct/100))
            delta = (pc - bcpu) / bcpu * 100
            preds.append(f"CPU {delta:+.1f}%")
        print(f"  {svc:<14} | {bwl:>8.2f} | {preds[0]:>12} | {preds[1]:>12} | {preds[2]:>12}")

    # Save summary
    summary_rows = []
    for svc, r in all_results.items():
        summary_rows.append({
            'service': svc, 'n_train': r['n_train'], 'n_test': r['n_test'],
            'train_wl_range': r['wl_train_range'], 'test_wl_range': r['wl_test_range'],
            'cpu_mape': r['cpu_mape'], 'cpu_r2': r['cpu_r2'],
            'lat_mape': r['lat_mape'], 'lat_r2': r['lat_r2'],
        })
    out = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results', 'gold_standard_evaluation.csv')
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print(f"\n  Results saved: {out}")
    print("=" * 90)


if __name__ == '__main__':
    main()
