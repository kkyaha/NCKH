# -*- coding: utf-8 -*-
"""
KIEM DINH: Them canh backpressure (caller_cpu lam parent thu 2) co thuc su
tang cac chi so (MAPE, R2, F1 vuot nguong) tren du lieu HELD-OUT khong?
=============================================================================
Ke tiep replace_vs_add_edge_test.py (da xac nhan ADD dung hon REPLACE ve mat
nhan qua). Script nay dung DUNG protocol OOD Gold Standard cua RQ1/Fast Path
(train LOW 67% -> test HIGH 33% theo workload cua CHINH callee, giu nguyen
tieu chi chia de so sanh cong bang) va do FULL-RESOLUTION (khong bucket, toan
bo diem test tho) tren ca MAPE/R2 (do chinh xac diem) VA F1 vuot nguong (chat
luong quyet dinh -- giong mechanism_decision_quality_trial.py da dung truoc do).

Cho 3 canh da co tin hieu (Nhom 1, diagnostic R2):
  orders -> shipping, orders -> carts, front-end -> user

So sanh 2 mechanism (ca hai deu LinearRegression(positive=True), khong doi
learner -- chi doi SO LUONG PARENT, dung cau hoi dang hoi):
  A) baseline : Target ~ own_workload
  B) backpressure_add : Target ~ own_workload + caller_cpu

Output: data/processed/scm_results/backpressure_edge_accuracy_test.csv
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score

sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data

EDGES = [('orders', 'shipping'), ('orders', 'carts'), ('front-end', 'user')]


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def evaluate_edge(caller, callee, df):
    wlc, cpuc = f'{callee}_workload', f'{callee}_cpu'
    caller_cpu_c = f'{caller}_cpu'
    sub = df[[wlc, cpuc, caller_cpu_c]].dropna()
    if len(sub) > 4000:
        sub = sub.sample(4000, random_state=42)
    sub = sub.sort_values(wlc).reset_index(drop=True)  # chia theo workload cua CALLEE, dung protocol RQ1
    split = int(len(sub) * 0.67)
    train, test = sub.iloc[:split], sub.iloc[split:]

    m_base = LinearRegression(positive=True).fit(train[[wlc]], train[cpuc])
    m_add = LinearRegression(positive=True).fit(train[[wlc, caller_cpu_c]], train[cpuc])

    yt = test[cpuc].values
    yp_base = m_base.predict(test[[wlc]]).ravel()
    yp_add = m_add.predict(test[[wlc, caller_cpu_c]]).ravel()

    thresh = train[cpuc].mean() + 0.5 * train[cpuc].std()
    yt_bin = (yt >= thresh).astype(int)

    out = {'caller': caller, 'callee': callee, 'n_train': len(train), 'n_test': len(test),
           'pos_rate': yt_bin.mean()}
    for name, yp in [('baseline', yp_base), ('backpressure_add', yp_add)]:
        yp_bin = (yp >= thresh).astype(int)
        out[f'{name}_mape'] = round(_mape(yt, yp), 2)
        out[f'{name}_rmse'] = round(float(np.sqrt(mean_squared_error(yt, yp))), 4)
        out[f'{name}_r2'] = round(r2_score(yt, yp), 4)
        out[f'{name}_precision'] = round(precision_score(yt_bin, yp_bin, zero_division=0), 3)
        out[f'{name}_recall'] = round(recall_score(yt_bin, yp_bin, zero_division=0), 3)
        out[f'{name}_f1'] = round(f1_score(yt_bin, yp_bin, zero_division=0), 3)
    return out


if __name__ == '__main__':
    df = load_multi_service_data(None, system_type='sockshop').dropna().reset_index(drop=True)
    rows = []
    for caller, callee in EDGES:
        r = evaluate_edge(caller, callee, df)
        rows.append(r)
        print(f"\n=== {caller} -> {callee} (n_test={r['n_test']}, pos_rate={r['pos_rate']:.2f}) ===")
        print(f"  MAPE   : baseline={r['baseline_mape']:6.2f}%  add={r['backpressure_add_mape']:6.2f}%  "
              f"({'TANG' if r['backpressure_add_mape'] < r['baseline_mape'] else 'GIAM'})")
        print(f"  R2     : baseline={r['baseline_r2']:6.3f}   add={r['backpressure_add_r2']:6.3f}   "
              f"({'TANG' if r['backpressure_add_r2'] > r['baseline_r2'] else 'GIAM'})")
        print(f"  F1     : baseline={r['baseline_f1']:.3f}    add={r['backpressure_add_f1']:.3f}    "
              f"({'TANG' if r['backpressure_add_f1'] > r['baseline_f1'] else 'GIAM'})")
        print(f"  Precision: baseline={r['baseline_precision']:.3f}  add={r['backpressure_add_precision']:.3f}")
        print(f"  Recall   : baseline={r['baseline_recall']:.3f}  add={r['backpressure_add_recall']:.3f}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'backpressure_edge_accuracy_test.csv')
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu: {out_path}")

    print("\n" + "=" * 70)
    print("  TONG HOP (trung binh 3 canh)")
    print("=" * 70)
    for metric in ['mape', 'r2', 'precision', 'recall', 'f1']:
        b = out[f'baseline_{metric}'].mean()
        a = out[f'backpressure_add_{metric}'].mean()
        print(f"  {metric:<10} baseline={b:7.3f}  backpressure_add={a:7.3f}")
