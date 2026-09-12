# -*- coding: utf-8 -*-
"""
KIEM DINH held-out (OOD Gold Standard, giong RQ1/backpressure_edge_accuracy_test.py)
cho viec them own-service memory + socket lam parent bo sung cua CPU (Sock Shop).

So sanh 2 mechanism (LinearRegression(positive=True), khong doi learner -- chi
doi so luong parent, dung cau hoi dang hoi):
  A) baseline_deployed : Target ~ own_workload (+ caller_cpu neu node da co
     backpressure deploy, de so sanh dung voi trang thai SAN XUAT hien tai)
  B) with_mem_socket    : Target ~ own_workload (+ caller_cpu neu co) + own_mem + own_socket

Dung DUNG protocol RQ1 (train workload THAP 67% -> test workload CAO 33%,
full-resolution, khong bucket) cho ca MAPE/R2 va F1 nguong.

Output: data/processed/scm_results/memsocket_edge_accuracy_test.csv
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
from data_processor import load_multi_service_data, SERVICES

# Backpressure DA DEPLOY (Sock Shop) -- de fit dung mechanism SAN XUAT hien tai
BACKPRESSURE = {
    'shipping': ['orders'],
    'carts': ['orders'],
    'user': ['front-end'],
}


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def evaluate(svc, df):
    wlc, cpuc, memc, sockc = f'{svc}_workload', f'{svc}_cpu', f'{svc}_mem', f'{svc}_socket'
    callers = BACKPRESSURE.get(svc, [])
    caller_cols = [f'{c}_cpu' for c in callers]
    base_cols = [wlc] + caller_cols
    ext_cols = base_cols + [memc, sockc]

    sub = df[list(set(ext_cols + [cpuc]))].dropna()
    if len(sub) > 4000:
        sub = sub.sample(4000, random_state=42)
    sub = sub.sort_values(wlc).reset_index(drop=True)
    split = int(len(sub) * 0.67)
    train, test = sub.iloc[:split], sub.iloc[split:]
    if train[wlc].nunique() < 3:
        return None

    m_base = LinearRegression(positive=True).fit(train[base_cols], train[cpuc])
    m_ext = LinearRegression(positive=True).fit(train[ext_cols], train[cpuc])

    yt = test[cpuc].values
    yp_base = m_base.predict(test[base_cols]).ravel()
    yp_ext = m_ext.predict(test[ext_cols]).ravel()

    thresh = train[cpuc].mean() + 0.5 * train[cpuc].std()
    yt_bin = (yt >= thresh).astype(int)
    if yt_bin.sum() == 0 or yt_bin.sum() == len(yt_bin):
        f1_base = f1_ext = np.nan
    else:
        f1_base = f1_score(yt_bin, (yp_base >= thresh).astype(int), zero_division=0)
        f1_ext = f1_score(yt_bin, (yp_ext >= thresh).astype(int), zero_division=0)

    return {
        'service': svc, 'n_test': len(test), 'has_backpressure': bool(callers),
        'baseline_mape': round(_mape(yt, yp_base), 2), 'ext_mape': round(_mape(yt, yp_ext), 2),
        'baseline_r2': round(r2_score(yt, yp_base), 4), 'ext_r2': round(r2_score(yt, yp_ext), 4),
        'baseline_f1': round(f1_base, 3) if not np.isnan(f1_base) else '',
        'ext_f1': round(f1_ext, 3) if not np.isnan(f1_ext) else '',
    }


if __name__ == '__main__':
    df = load_multi_service_data(None, system_type='sockshop')
    rows = []
    for svc in SERVICES:
        res = evaluate(svc, df)
        if res:
            rows.append(res)
            print(f"{svc:12s} bp={res['has_backpressure']}  MAPE {res['baseline_mape']:6.2f}%->{res['ext_mape']:6.2f}%  "
                  f"R2 {res['baseline_r2']:7.3f}->{res['ext_r2']:7.3f}  F1 {res['baseline_f1']}->{res['ext_f1']}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'memsocket_edge_accuracy_test.csv'), index=False)
    print(f"\nMAPE cai thien: {(out['ext_mape']<out['baseline_mape']).sum()}/{len(out)}")
    print(f"R2 cai thien  : {(out['ext_r2']>out['baseline_r2']).sum()}/{len(out)}")
    print(f"Trung binh MAPE: baseline={out['baseline_mape'].mean():.2f}%  ext={out['ext_mape'].mean():.2f}%")
    print(f"Trung binh R2  : baseline={out['baseline_r2'].mean():.4f}  ext={out['ext_r2'].mean():.4f}")
