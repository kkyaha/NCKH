# -*- coding: utf-8 -*-
"""
CHAN DOAN: Yeu to ngoai Workload nao giup giai thich phan du CPU (R2 am)?
==========================================================================
Boi canh: RQ1 (docs/paper_draft.tex, Section 5) bao cao R2 am o da so cap
service-metric tren Sock Shop. Script nay kiem dinh 3 gia thuyet ve nguyen
nhan, THAY VI suy doan, truoc khi dua ket luan vao paper:

  H1 (noisy-neighbor)  : tong workload dong thoi cua CAC SERVICE KHAC co giai
                          thich duoc phan du CPU khong? -- KET QUA: khong
                          (corr 0.004-0.12, R2 tang khong dang ke).
  H2 (nhieu do nhanh)  : lam muot target (rolling mean 5/15 mau) co tang R2
                          voi workload khong? -- KET QUA: khong (R2 khong doi).
  H3 (call-chain nghen) : CPU cua CALLER (dich vu goi truc tiep, theo dung
                          canh HTTP that trong sockshop_agent_graph.json) co
                          giai thich duoc phan du CPU cua CALLEE (ngoai workload
                          rieng cua callee) khong -- KET QUA: co, tren mot so
                          canh (orders->shipping, orders->carts, front-end->user),
                          khong tren cac canh khac (front-end->catalogue,
                          orders->payment).

  Phu: phan ra phuong sai giua-run vs trong-run cho catalogue_cpu, kiem tra
  R2~0 co phai do gop nhieu "che do" (scenario) khac nhau khong -- KET QUA:
  khong, giua-run chi chiem 2.3% tong phuong sai.

Output: data/processed/scm_results/call_chain_neighbor_diagnostic.csv
"""

import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data

SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
# Canh caller->callee thuc te (chi 7 service chinh), lay tu src/graph/sockshop_agent_graph.json
EDGES = [
    ('front-end', 'catalogue'), ('front-end', 'carts'), ('front-end', 'user'), ('front-end', 'orders'),
    ('orders', 'carts'), ('orders', 'user'), ('orders', 'payment'), ('orders', 'shipping'),
]


def h1_noisy_neighbor(df):
    df = df.copy()
    wl_cols = [f'{s}_workload' for s in SERVICES]
    df['total_workload'] = df[wl_cols].sum(axis=1)
    rows = []
    for svc in SERVICES:
        wlc, cpuc = f'{svc}_workload', f'{svc}_cpu'
        sub = df[[wlc, cpuc, 'total_workload']].dropna()
        if len(sub) < 200:
            continue
        m1 = LinearRegression(positive=True).fit(sub[[wlc]], sub[cpuc])
        r2_1 = m1.score(sub[[wlc]], sub[cpuc])
        resid1 = sub[cpuc] - m1.predict(sub[[wlc]])
        other_wl = sub['total_workload'] - sub[wlc]
        m2 = LinearRegression(positive=True).fit(sub[[wlc, 'total_workload']], sub[cpuc])
        r2_2 = m2.score(sub[[wlc, 'total_workload']], sub[cpuc])
        rows.append({'hypothesis': 'H1_noisy_neighbor', 'service': svc,
                     'r2_baseline': round(r2_1, 3), 'r2_with_extra': round(r2_2, 3),
                     'corr_resid_extra': round(float(np.corrcoef(resid1, other_wl)[0, 1]), 3)})
    return rows


def h2_target_smoothing(df):
    rows = []
    for svc in SERVICES:
        wlc, cpuc = f'{svc}_workload', f'{svc}_cpu'
        sub = df[[wlc, cpuc]].dropna().reset_index(drop=True)
        if len(sub) < 200:
            continue
        m1 = LinearRegression(positive=True).fit(sub[[wlc]], sub[cpuc])
        r2_raw = m1.score(sub[[wlc]], sub[cpuc])
        for win in [5, 15]:
            smoothed = sub[cpuc].rolling(win, center=True, min_periods=1).mean()
            m2 = LinearRegression(positive=True).fit(sub[[wlc]], smoothed)
            r2_smooth = m2.score(sub[[wlc]], smoothed)
            rows.append({'hypothesis': f'H2_smooth_win{win}', 'service': svc,
                         'r2_baseline': round(r2_raw, 3), 'r2_with_extra': round(r2_smooth, 3),
                         'corr_resid_extra': ''})
    return rows


def h3_call_chain_caller_cpu(df):
    rows = []
    for caller, callee in EDGES:
        wlc, cpuc = f'{callee}_workload', f'{callee}_cpu'
        caller_cpu_c = f'{caller}_cpu'
        sub = df[[wlc, cpuc, caller_cpu_c]].dropna()
        if len(sub) < 200:
            continue
        m1 = LinearRegression(positive=True).fit(sub[[wlc]], sub[cpuc])
        r2_1 = m1.score(sub[[wlc]], sub[cpuc])
        resid1 = sub[cpuc] - m1.predict(sub[[wlc]])
        m2 = LinearRegression(positive=True).fit(sub[[wlc, caller_cpu_c]], sub[cpuc])
        r2_2 = m2.score(sub[[wlc, caller_cpu_c]], sub[cpuc])
        rows.append({'hypothesis': 'H3_call_chain_caller_cpu', 'service': f'{caller}->{callee}',
                     'r2_baseline': round(r2_1, 3), 'r2_with_extra': round(r2_2, 3),
                     'corr_resid_extra': round(float(np.corrcoef(resid1, sub[caller_cpu_c])[0, 1]), 3)})
    return rows


def between_within_run_variance(metric='catalogue_cpu', workload_col='catalogue_workload'):
    """Phan ra phuong sai giua-run vs trong-run tu du lieu THO (co nhan run),
    khong dung load_multi_service_data (khong giu ten run)."""
    RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'RE2-SS')
    rows = []
    for scenario in os.listdir(RAW_DIR):
        sp = os.path.join(RAW_DIR, scenario)
        if not os.path.isdir(sp):
            continue
        for run_id in os.listdir(sp):
            rp = os.path.join(sp, run_id)
            mp, ip = os.path.join(rp, 'simple_metrics.csv'), os.path.join(rp, 'inject_time.txt')
            if not (os.path.exists(mp) and os.path.exists(ip)):
                continue
            try:
                with open(ip) as f:
                    it = int(f.read().strip())
                cols = pd.read_csv(mp, nrows=0).columns
                tc = 'imte' if 'imte' in cols else ('time' if 'time' in cols else None)
                if tc is None or workload_col not in cols or metric not in cols:
                    continue
                d = pd.read_csv(mp, usecols=[tc, workload_col, metric])
                d = d[d[tc] < it]
                d['run'] = f'{scenario}/{run_id}'
                rows.append(d[[workload_col, metric, 'run']])
            except Exception:
                continue
    df = pd.concat(rows, ignore_index=True).dropna()
    overall_var = df[metric].var()
    run_means = df.groupby('run')[metric].mean()
    between_var = run_means.var()
    within_var = df.groupby('run')[metric].var().mean()
    within_corrs = []
    for _, g in df.groupby('run'):
        if g[workload_col].nunique() > 3 and len(g) > 20:
            c = g[[workload_col, metric]].corr().iloc[0, 1]
            if np.isfinite(c):
                within_corrs.append(c)
    return {
        'hypothesis': 'run_variance_decomposition', 'service': metric,
        'n_rows': len(df), 'n_runs': df['run'].nunique(),
        'between_run_pct_of_var': round(between_var / overall_var * 100, 1),
        'within_run_pct_of_var': round(within_var / overall_var * 100, 1),
        'overall_corr_workload_target': round(float(df[[workload_col, metric]].corr().iloc[0, 1]), 3),
        'mean_within_run_corr': round(float(np.mean(within_corrs)), 3) if within_corrs else '',
    }


if __name__ == '__main__':
    df = load_multi_service_data(None, system_type='sockshop').dropna().reset_index(drop=True)

    all_rows = []
    print("H1: Noisy-neighbor (tong workload cac service khac)")
    r = h1_noisy_neighbor(df)
    all_rows += r
    for x in r:
        print(f"  {x['service']:<12} R2 {x['r2_baseline']:.3f} -> {x['r2_with_extra']:.3f}  corr={x['corr_resid_extra']}")

    print("\nH2: Lam muot target (rolling mean)")
    r = h2_target_smoothing(df)
    all_rows += r
    for x in r:
        print(f"  {x['service']:<12} [{x['hypothesis']}] R2 {x['r2_baseline']:.3f} -> {x['r2_with_extra']:.3f}")

    print("\nH3: Call-chain caller CPU (canh HTTP thuc te)")
    r = h3_call_chain_caller_cpu(df)
    all_rows += r
    for x in r:
        print(f"  {x['service']:<22} R2 {x['r2_baseline']:.3f} -> {x['r2_with_extra']:.3f}  corr={x['corr_resid_extra']}")

    print("\nPhu: phan ra phuong sai giua-run vs trong-run (catalogue_cpu)")
    rv = between_within_run_variance()
    all_rows.append(rv)
    print(f"  n_runs={rv['n_runs']}  giua-run={rv['between_run_pct_of_var']}%  "
          f"trong-run={rv['within_run_pct_of_var']}%  corr(workload,cpu) tong the={rv['overall_corr_workload_target']}  "
          f"corr trong-run={rv['mean_within_run_corr']}")

    out = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, 'call_chain_neighbor_diagnostic.csv')
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu: {out_path}")
