# -*- coding: utf-8 -*-
# encoding: utf-8
"""
SCM Impact Prediction Pipeline - Structured Evaluation
=======================================================
Quy trinh 4 buoc ro rang de trinh bay trong bai bao / bao cao:

  STEP 1: DATA OVERVIEW      -> 01_data_overview.csv
  STEP 2: MODEL ACCURACY     -> 02_model_accuracy.csv
  STEP 3: WORKLOAD SENSITIVITY -> 03_workload_sensitivity.csv
  STEP 4: REQUEST IMPACT     -> 04_request_impact.csv

Phuong phap Gold Standard:
  - Train: du lieu binh thuong tai muc tai THAP (bottom 67% workload)
  - Test : du doan tai muc tai CAO (top 33%) va so sanh voi do that
"""

import os, sys, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'
OUT_DIR  = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']

# Core Resource Capacity Metrics: (display_name, column_suffix, display_unit, scale_to_unit)
METRICS = [
    ('CPU',        'cpu',        '%',   1.0   ),
    ('Memory',     'mem',        'MB',  1/1e6 ),
    ('Socket',     'socket',     'cnt', 1.0   ),
]

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

N_PROJ = 500

def mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100 if m.sum()>0 else float('nan')

def sep(title='', w=85):
    if title:
        pad = (w - len(title) - 4) // 2
        print('\n' + '='*w)
        print(' '*pad + '  ' + title)
        print('='*w)
    else:
        print('-'*w)


# ============================================================
# DATA LOADER
# ============================================================
def load_normal_data(service: str, metric_col: str) -> pd.DataFrame:
    """Load normal-period [Workload, Target] from all 90 runs efficiently."""
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
                # Quick column check
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
# STEP 1 : DATA OVERVIEW
# ============================================================
def step1_data_overview():
    sep('STEP 1: DATA OVERVIEW - How much data do we have?')
    rows = []
    for svc in SERVICES:
        df = load_normal_data(svc, 'cpu')
        if df is None:
            rows.append({'service': svc, 'n_samples': 0})
            continue
        rows.append({
            'service':        svc,
            'n_samples':      len(df),
            'n_scenarios':    90,
            'workload_min':   round(df['Workload'].min(), 2),
            'workload_mean':  round(df['Workload'].mean(), 2),
            'workload_max':   round(df['Workload'].max(), 2),
            'workload_std':   round(df['Workload'].std(), 2),
        })
        print(f"  [{svc}] {len(df):>6,} samples | "
              f"WL: {df['Workload'].min():.1f} - {df['Workload'].max():.1f} req/s "
              f"(mean={df['Workload'].mean():.1f}, std={df['Workload'].std():.1f})")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, '01_data_overview.csv'), index=False)
    print(f"\n  Saved: 01_data_overview.csv")
    print(f"  Total normal samples: {df_out['n_samples'].sum():,}")
    print(f"  Source: 90 runs (30 fault scenarios x 3 runs), first 12min each")
    return df_out


# ============================================================
# STEP 2 : MODEL ACCURACY
# ============================================================
def step2_model_accuracy():
    sep('STEP 2: MODEL ACCURACY - Gold Standard Evaluation')
    print('  Method: Train on LOW workload (bottom 67%) -> Test on HIGH workload (top 33%)')
    print('  This simulates: "Can SCM predict what happens when we ADD more requests?"')
    sep()

    all_rows = []
    trained_models = {}   # (service, metric_name) -> {model, baseline_wl, baseline_val}

    for metric_name, metric_col, unit, scale in METRICS:
        print(f"\n  [{metric_name}]")
        for svc in SERVICES:
            df = load_normal_data(svc, metric_col)
            if df is None or len(df) < 200:
                print(f"    {svc:<14}: no data")
                continue

            df = df.sort_values('Workload').reset_index(drop=True)
            split = int(len(df) * 0.67)
            df_train = df.iloc[:split]
            df_test  = df.iloc[split:].copy()

            # Build & train bivariate SCM: Workload -> Target
            g = nx.DiGraph(); g.add_edge('Workload','Target')
            model = gcm.InvertibleStructuralCausalModel(g)
            gcm.auto.assign_causal_mechanisms(model, df_train)
            gcm.fit(model, df_train)

            # Bucket test set (8 bins by workload)
            df_test['bkt'] = pd.qcut(df_test['Workload'], q=min(8,df_test['Workload'].nunique()), duplicates='drop')
            bkts = df_test.groupby('bkt', observed=True)[['Workload','Target']].mean()

            y_true, y_pred = [], []
            for _, row in bkts.iterrows():
                wlc = row['Workload']
                dp  = gcm.interventional_samples(model,
                        interventions={'Workload': lambda x, w=wlc: w},
                        num_samples_to_draw=N_PROJ)
                y_true.append(row['Target'])
                y_pred.append(dp['Target'].mean())

            yt, yp = np.array(y_true), np.array(y_pred)
            mp_val = mape(yt, yp)
            mae_v  = mean_absolute_error(yt, yp) * scale
            rmse_v = np.sqrt(mean_squared_error(yt, yp)) * scale
            r2_v   = r2_score(yt, yp)

            # Compute Risk Detection F1-Score (Threshold = Mean_train + 0.5 * Std_train)
            thresh = df_train['Target'].mean() + 0.5 * df_train['Target'].std()
            yt_bin = (yt >= thresh).astype(int)
            yp_bin = (yp >= thresh).astype(int)
            f1_v   = f1_score(yt_bin, yp_bin, average='binary', zero_division=1)

            good   = '*' if mp_val < 10 else ' '

            trained_models[(svc, metric_name)] = {
                'model':        model,
                'baseline_wl':  df_train['Workload'].mean(),
                'baseline_val': df_train['Target'].mean() * scale,
                'wl_train_max': df_train['Workload'].max(),
                'wl_test_max':  df_test['Workload'].max(),
            }

            all_rows.append({
                'service':     svc, 'metric': metric_name, 'unit': unit,
                'n_train':     len(df_train), 'n_test': len(df_test),
                'wl_train':    f"{df_train['Workload'].min():.1f}-{df_train['Workload'].max():.1f}",
                'wl_test':     f"{df_test['Workload'].min():.1f}-{df_test['Workload'].max():.1f}",
                'mape_pct':    round(mp_val, 2),
                'mae':         round(mae_v,  4),
                'rmse':        round(rmse_v, 4),
                'f1_score':    round(f1_v,   3),
                'r2':          round(r2_v,   3),
                'mean_actual': round(np.mean(yt)*scale, 4),
                'mean_pred':   round(np.mean(yp)*scale, 4),
            })
            tag = 'EXCELLENT' if mp_val < 10 else ('FAIR' if mp_val < 25 else 'POOR')
            print(f"    {svc:<14}: MAPE={mp_val:>5.1f}%{good} RMSE={rmse_v:>8.4f} F1={f1_v:>5.3f} | {tag}")

    df_acc = pd.DataFrame(all_rows)
    df_acc.to_csv(os.path.join(OUT_DIR, '02_model_accuracy.csv'), index=False)

    print(f"\n  --- Summary by Metric ---")
    for metric_name, _, unit, _ in METRICS:
        sub = df_acc[df_acc['metric']==metric_name]
        if sub.empty: continue
        n_good = (sub['mape_pct'] < 10).sum()
        print(f"  {metric_name:<14}: avg MAPE={sub['mape_pct'].mean():>5.1f}% | "
              f"{n_good}/7 services excellent (<10%)")

    print(f"\n  Saved: 02_model_accuracy.csv  (* = MAPE < 10%)")
    return df_acc, trained_models


# ============================================================
# STEP 3 : WORKLOAD SENSITIVITY
# ============================================================
def step3_workload_sensitivity(trained_models: dict):
    sep('STEP 3: WORKLOAD SENSITIVITY - Predicted change at +10%, +20%, +30%, +50%')
    print('  Question: "If workload increases by X%, what happens to each metric?"')
    sep()

    pct_list = [10, 20, 30, 50]
    rows = []

    for svc in SERVICES:
        key_cpu = (svc, 'CPU')
        if key_cpu not in trained_models: continue
        base_wl = trained_models[key_cpu]['baseline_wl']

        svc_rows = {}
        for metric_name, _, unit, _ in METRICS:
            key = (svc, metric_name)
            if key not in trained_models: continue
            m_info = trained_models[key]
            model   = m_info['model']
            base_v  = m_info['baseline_val']

            for pct in pct_list:
                new_wl = base_wl * (1 + pct/100)
                wlc = new_wl
                dp  = gcm.interventional_samples(model,
                        interventions={'Workload': lambda x, w=wlc: w},
                        num_samples_to_draw=N_PROJ)
                # scale already in baseline_val
                pred_raw = dp['Target'].mean()
                # scale to unit
                _, mc, unit2, scale2 = next((x for x in METRICS if x[0]==metric_name), (None,None,None,1))
                pred_v = pred_raw * scale2
                chg_pct = (pred_v - base_v) / abs(base_v) * 100 if base_v != 0 else 0
                rows.append({
                    'service': svc, 'metric': metric_name, 'unit': unit,
                    'baseline_workload': round(base_wl, 2),
                    'workload_increase_pct': pct,
                    'baseline_value': round(base_v, 4),
                    'predicted_value': round(pred_v, 4),
                    'change_pct': round(chg_pct, 2),
                })

    df_sens = pd.DataFrame(rows)

    # Print clean table per service
    for svc in SERVICES:
        sub = df_sens[df_sens['service']==svc]
        if sub.empty: continue
        bwl = sub['baseline_workload'].iloc[0]
        print(f"  [{svc}]  Baseline WL = {bwl:.2f} req/s")
        header = f"    {'Metric':<14} | {'Baseline':>10}"
        for p in pct_list: header += f" | {'+'+str(p)+'%':>8}"
        print(header)
        print("    " + "-"*70)
        for metric_name, _, unit, _ in METRICS:
            sub2 = sub[sub['metric']==metric_name].sort_values('workload_increase_pct')
            if sub2.empty: continue
            bv = sub2['baseline_value'].iloc[0]
            row_str = f"    {metric_name:<14} | {bv:>8.2f}{unit}"
            for _, r2 in sub2.iterrows():
                chg = r2['change_pct']
                row_str += f" | {chg:>+6.1f}%"
            print(row_str)
        print()

    df_sens.to_csv(os.path.join(OUT_DIR, '03_workload_sensitivity.csv'), index=False)
    print(f"  Saved: 03_workload_sensitivity.csv")
    return df_sens


# ============================================================
# STEP 4 : REQUEST IMPACT
# ============================================================
def step4_request_impact(trained_models: dict):
    sep('STEP 4: REQUEST IMPACT - Full System Impact by Request Type')
    print('  Question: "What happens to the WHOLE SYSTEM when we add a new feature/request?"')

    DELTA_PCT = 20  # simulate adding 20% more load
    rows = []

    for req_type, services in CALL_CHAINS.items():
        print(f"\n  [{req_type}]  Blast Radius: {' -> '.join(services)}")
        print(f"  Simulating: +{DELTA_PCT}% workload increase on each affected service")
        print(f"  {'Service':<14} | {'CPU':>10} | {'Memory':>10} | {'Socket':>10} | {'Lat_p50':>10} | {'Lat_p90':>10}")
        print("  " + "-"*68)

        for svc in services:
            key_cpu = (svc, 'CPU')
            if key_cpu not in trained_models:
                print(f"  {svc:<14} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")
                continue
            base_wl = trained_models[key_cpu]['baseline_wl']
            new_wl  = base_wl * (1 + DELTA_PCT/100)

            cols = {}
            for metric_name, mc, unit, scale in METRICS:
                key = (svc, metric_name)
                if key not in trained_models:
                    cols[metric_name] = 'N/A'
                    continue
                m_info = trained_models[key]
                model  = m_info['model']
                base_v = m_info['baseline_val']
                wlc = new_wl
                dp  = gcm.interventional_samples(model,
                        interventions={'Workload': lambda x, w=wlc: w},
                        num_samples_to_draw=N_PROJ)
                pred_v = dp['Target'].mean() * scale
                chg    = (pred_v - base_v) / abs(base_v) * 100 if base_v != 0 else 0
                cols[metric_name] = f"{chg:+.1f}%"

                rows.append({
                    'request_type': req_type,
                    'service': svc,
                    'metric': metric_name,
                    'baseline_workload': round(base_wl, 2),
                    'new_workload': round(new_wl, 2),
                    'workload_delta_pct': DELTA_PCT,
                    'change_pct': round(chg, 2),
                })

            print(f"  {svc:<14} | "
                  f"{cols.get('CPU','N/A'):>10} | "
                  f"{cols.get('Memory','N/A'):>10} | "
                  f"{cols.get('Socket','N/A'):>10} | "
                  f"{cols.get('Latency_p50','N/A'):>10} | "
                  f"{cols.get('Latency_p90','N/A'):>10}")

    df_imp = pd.DataFrame(rows)
    df_imp.to_csv(os.path.join(OUT_DIR, '04_request_impact.csv'), index=False)
    print(f"\n  Saved: 04_request_impact.csv")
    return df_imp


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*85)
    print("  SCM IMPACT PREDICTION PIPELINE  |  RCAEval Dataset  |  SockShop System")
    print("="*85)
    print("  Output directory:", OUT_DIR)

    df_overview                = step1_data_overview()
    df_accuracy, trained_models = step2_model_accuracy()
    df_sensitivity             = step3_workload_sensitivity(trained_models)
    df_impact                  = step4_request_impact(trained_models)

    sep('PIPELINE COMPLETE')
    print("  Output files:")
    print("    01_data_overview.csv       - Data statistics per service")
    print("    02_model_accuracy.csv      - MAPE/MAE/R2 per service per metric")
    print("    03_workload_sensitivity.csv - Predicted change at +10/20/30/50%")
    print("    04_request_impact.csv      - System impact per request type")
    print("="*85)


if __name__ == '__main__':
    main()
