# -*- coding: utf-8 -*-
"""
SUA MOT KHOANG TRONG KIEM DINH: tt_backpressure_edge_accuracy_test.py danh gia
tung CANH rieng le (own_workload + 1 caller_cpu). Nhung thuc te 50 canh (gain>0.03)
chi cham vao 20/84 node CPU duy nhat -- nhieu node (vd ts-station-service_cpu) co
TOI 5 caller_cpu lam parent DONG THOI trong do thi that da deploy. Script nay
danh gia dung mo hinh DA TRIEN KHAI (own_workload + TOAN BO caller_cpu cua node
do cung luc), khong phai tung canh doc lap.

Output: data/processed/scm_results/tt_backpressure_multiparent_accuracy_test.csv
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
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score

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


if __name__ == '__main__':
    diag = pd.read_csv(os.path.join(OUT_DIR, 'tt_call_chain_neighbor_diagnostic.csv'))
    strong = diag[diag['gain'] > GAIN_THRESHOLD]
    callers_by_callee = strong.groupby('callee')['caller'].apply(list).to_dict()
    print(f"So node callee duy nhat bi anh huong: {len(callers_by_callee)}")

    df = load_multi_service_data(os.path.join(BASE_DIR, 'data', 'raw', 'trainticket'), system_type='trainticket')

    rows = []
    for callee, callers in callers_by_callee.items():
        wlc, cpuc = f'{callee}_workload', f'{callee}_cpu'
        caller_cols = [f'{c}_cpu' for c in callers]
        needed = [wlc, cpuc] + caller_cols
        if not all(c in df.columns for c in needed):
            continue
        sub = df[needed].dropna()
        if len(sub) < 300:
            continue
        if len(sub) > 4000:
            sub = sub.sample(4000, random_state=42)
        sub = sub.sort_values(wlc).reset_index(drop=True)
        split = int(len(sub) * 0.67)
        train, test = sub.iloc[:split], sub.iloc[split:]
        if train[wlc].nunique() < 3:
            continue

        m_base = LinearRegression(positive=True).fit(train[[wlc]], train[cpuc])
        m_multi = LinearRegression(positive=True).fit(train[[wlc] + caller_cols], train[cpuc])

        yt = test[cpuc].values
        yp_base = m_base.predict(test[[wlc]]).ravel()
        yp_multi = m_multi.predict(test[[wlc] + caller_cols]).ravel()

        thresh = train[cpuc].mean() + 0.5 * train[cpuc].std()
        yt_bin = (yt >= thresh).astype(int)
        if yt_bin.sum() == 0 or yt_bin.sum() == len(yt_bin):
            f1_base = f1_multi = np.nan
        else:
            f1_base = f1_score(yt_bin, (yp_base >= thresh).astype(int), zero_division=0)
            f1_multi = f1_score(yt_bin, (yp_multi >= thresh).astype(int), zero_division=0)

        rows.append({
            'callee': callee, 'n_callers': len(callers), 'callers': ';'.join(callers), 'n_test': len(test),
            'baseline_mape': round(_mape(yt, yp_base), 2), 'multi_mape': round(_mape(yt, yp_multi), 2),
            'baseline_r2': round(r2_score(yt, yp_base), 4), 'multi_r2': round(r2_score(yt, yp_multi), 4),
            'baseline_f1': round(f1_base, 3) if not np.isnan(f1_base) else '',
            'multi_f1': round(f1_multi, 3) if not np.isnan(f1_multi) else '',
        })
        print(f"  {callee:<32} (n_caller={len(callers)}) MAPE {rows[-1]['baseline_mape']:>6.1f}%->{rows[-1]['multi_mape']:>6.1f}%  "
              f"R2 {rows[-1]['baseline_r2']:>7.3f}->{rows[-1]['multi_r2']:>7.3f}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'tt_backpressure_multiparent_accuracy_test.csv')
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu: {out_path}  (n={len(out)} node)")

    n_mape_improve = (out['multi_mape'] < out['baseline_mape']).sum()
    n_r2_improve = (out['multi_r2'] > out['baseline_r2']).sum()
    f1v = out.dropna(subset=['baseline_f1', 'multi_f1'])
    n_f1_improve = (f1v['multi_f1'] > f1v['baseline_f1']).sum()
    print(f"\nMAPE cai thien: {n_mape_improve}/{len(out)}")
    print(f"R2 cai thien  : {n_r2_improve}/{len(out)}")
    print(f"F1 cai thien  : {n_f1_improve}/{len(f1v)}")
    print(f"Trung binh MAPE: baseline={out['baseline_mape'].mean():.2f}%  multi={out['multi_mape'].mean():.2f}%")
    print(f"Trung binh R2  : baseline={out['baseline_r2'].mean():.4f}  multi={out['multi_r2'].mean():.4f}")
    print(f"Trung binh F1  : baseline={f1v['baseline_f1'].mean():.3f}  multi={f1v['multi_f1'].mean():.3f}")
