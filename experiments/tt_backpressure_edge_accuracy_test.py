# -*- coding: utf-8 -*-
"""
TRAIN TICKET: kiem dinh canh backpressure (caller_cpu lam parent thu 2) tren
du lieu HELD-OUT, dung DUNG protocol OOD Gold Standard nhu RQ1/backpressure_edge_accuracy_test.py
(Sock Shop). Dung cho toan bo cac canh co gain R2 > threshold tu
tt_call_chain_neighbor_diagnostic.csv (50/66 canh o threshold 0.03).

Output: data/processed/scm_results/tt_backpressure_edge_accuracy_test.csv
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
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score

sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data

GAIN_THRESHOLD = 0.03


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def evaluate_edge(caller, callee, df):
    wlc, cpuc = f'{callee}_workload', f'{callee}_cpu'
    caller_cpu_c = f'{caller}_cpu'
    sub = df[[wlc, cpuc, caller_cpu_c]].dropna()
    if len(sub) < 300:
        return None
    if len(sub) > 4000:
        sub = sub.sample(4000, random_state=42)
    sub = sub.sort_values(wlc).reset_index(drop=True)
    split = int(len(sub) * 0.67)
    train, test = sub.iloc[:split], sub.iloc[split:]
    if train[wlc].nunique() < 3:
        return None

    m_base = LinearRegression(positive=True).fit(train[[wlc]], train[cpuc])
    m_add = LinearRegression(positive=True).fit(train[[wlc, caller_cpu_c]], train[cpuc])

    yt = test[cpuc].values
    yp_base = m_base.predict(test[[wlc]]).ravel()
    yp_add = m_add.predict(test[[wlc, caller_cpu_c]]).ravel()

    thresh = train[cpuc].mean() + 0.5 * train[cpuc].std()
    yt_bin = (yt >= thresh).astype(int)
    if yt_bin.sum() == 0 or yt_bin.sum() == len(yt_bin):
        f1_base = f1_add = prec_base = prec_add = rec_base = rec_add = np.nan
    else:
        yp_base_bin = (yp_base >= thresh).astype(int)
        yp_add_bin = (yp_add >= thresh).astype(int)
        f1_base = f1_score(yt_bin, yp_base_bin, zero_division=0)
        f1_add = f1_score(yt_bin, yp_add_bin, zero_division=0)
        prec_base = precision_score(yt_bin, yp_base_bin, zero_division=0)
        prec_add = precision_score(yt_bin, yp_add_bin, zero_division=0)
        rec_base = recall_score(yt_bin, yp_base_bin, zero_division=0)
        rec_add = recall_score(yt_bin, yp_add_bin, zero_division=0)

    return {
        'caller': caller, 'callee': callee, 'n_test': len(test),
        'baseline_mape': round(_mape(yt, yp_base), 2), 'add_mape': round(_mape(yt, yp_add), 2),
        'baseline_r2': round(r2_score(yt, yp_base), 4), 'add_r2': round(r2_score(yt, yp_add), 4),
        'baseline_f1': round(f1_base, 3) if not np.isnan(f1_base) else '',
        'add_f1': round(f1_add, 3) if not np.isnan(f1_add) else '',
        'baseline_precision': round(prec_base, 3) if not np.isnan(prec_base) else '',
        'add_precision': round(prec_add, 3) if not np.isnan(prec_add) else '',
        'baseline_recall': round(rec_base, 3) if not np.isnan(rec_base) else '',
        'add_recall': round(rec_add, 3) if not np.isnan(rec_add) else '',
    }


if __name__ == '__main__':
    diag = pd.read_csv(os.path.join(OUT_DIR, 'tt_call_chain_neighbor_diagnostic.csv'))
    strong = diag[diag['gain'] > GAIN_THRESHOLD]
    print(f"So canh se kiem dinh (gain>{GAIN_THRESHOLD}): {len(strong)}")

    df = load_multi_service_data(os.path.join(BASE_DIR, 'data', 'raw', 'trainticket'), system_type='trainticket')

    rows = []
    for _, r in strong.iterrows():
        res = evaluate_edge(r['caller'], r['callee'], df)
        if res is not None:
            rows.append(res)

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'tt_backpressure_edge_accuracy_test.csv')
    out.to_csv(out_path, index=False)
    print(f"[OK] Da luu: {out_path}  (n={len(out)} canh danh gia duoc)")

    print("\n" + "=" * 70)
    print("  TONG HOP")
    print("=" * 70)
    n_mape_improve = (out['add_mape'] < out['baseline_mape']).sum()
    n_r2_improve = (out['add_r2'] > out['baseline_r2']).sum()
    f1_valid = out.dropna(subset=['baseline_f1', 'add_f1'])
    n_f1_improve = (f1_valid['add_f1'] > f1_valid['baseline_f1']).sum()
    print(f"  MAPE giam (cai thien): {n_mape_improve}/{len(out)} canh")
    print(f"  R2 tang (cai thien)  : {n_r2_improve}/{len(out)} canh")
    print(f"  F1 tang (cai thien)  : {n_f1_improve}/{len(f1_valid)} canh (co the danh gia)")
    print(f"\n  Trung binh MAPE : baseline={out['baseline_mape'].mean():.2f}%  add={out['add_mape'].mean():.2f}%")
    print(f"  Trung binh R2   : baseline={out['baseline_r2'].mean():.4f}  add={out['add_r2'].mean():.4f}")
    print(f"  Trung binh F1   : baseline={f1_valid['baseline_f1'].mean():.3f}  add={f1_valid['add_f1'].mean():.3f}")
    print(f"  Median MAPE     : baseline={out['baseline_mape'].median():.2f}%  add={out['add_mape'].median():.2f}%")
