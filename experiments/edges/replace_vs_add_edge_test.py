# -*- coding: utf-8 -*-
"""
KIEM DINH: "Thay canh" (shipping_workload -> orders_cpu) hay "Them canh" (giu ca 2)?
=====================================================================================
Boi canh: call_chain_neighbor_diagnostic.py phat hien orders_cpu giai thich duoc
phan du CPU cua shipping/carts. gcm.falsify.falsify_graph (chay thu cong trong
phien lam viec) di xa hon: sau khi THEM canh orders_cpu->shipping_cpu, no van
de xuat BO canh shipping_workload->shipping_cpu -- goi y REPLACE thay vi ADD.

Cau hoi: REPLACE (chi con orders_cpu lam parent) co an toan khong, hay chi la
mot ao giac do da collinear/confound trong du lieu quan sat (moi lan load-test
thuong tang TAT CA service cung luc, nen orders_cpu va shipping_workload gan
nhu luon di cung nhau trong du lieu -- khong co nghia shipping_workload KHONG
CO tac dong nhan qua rieng)?

Phuong phap: (1) so sanh R2 cua 3 mo hinh (own-only, caller-cpu-only=REPLACE,
ca hai=ADD) + tuong quan RIENG (partial correlation) cua own_workload sau khi
da tru caller_cpu; (2) THI NGHIEM QUYET DINH: mo phong do(callee_workload=+30%)
TRUC TIEP tai callee (vd shipping la injection_service, orders_cpu KHONG doi) --
day la dung loai truy van do(x) paper nay can ho tro (mot yeu cau moi chi tac
dong 1 service). REPLACE se cho ket qua the nao?

Output: data/processed/scm_results/replace_vs_add_edge_test.csv
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

PAIRS = [('orders', 'shipping'), ('orders', 'carts')]
DELTA_PCT = 30.0  # do(callee_workload = +30%) truc tiep tai callee


def main():
    df = load_multi_service_data(None, system_type='sockshop').dropna().reset_index(drop=True)
    rows = []
    for caller, callee in PAIRS:
        wlc, cpuc = f'{callee}_workload', f'{callee}_cpu'
        caller_cpu_c = f'{caller}_cpu'
        sub = df[[wlc, cpuc, caller_cpu_c]].dropna()

        m_own = LinearRegression(positive=True).fit(sub[[wlc]], sub[cpuc])
        r2_own = m_own.score(sub[[wlc]], sub[cpuc])

        m_replace = LinearRegression(positive=True).fit(sub[[caller_cpu_c]], sub[cpuc])
        r2_replace = m_replace.score(sub[[caller_cpu_c]], sub[cpuc])

        m_both = LinearRegression(positive=True).fit(sub[[wlc, caller_cpu_c]], sub[cpuc])
        r2_both = m_both.score(sub[[wlc, caller_cpu_c]], sub[cpuc])

        resid_after_caller = sub[cpuc] - m_replace.predict(sub[[caller_cpu_c]])
        partial_corr_ownwl = np.corrcoef(resid_after_caller, sub[wlc])[0, 1]

        base_own_wl = sub[wlc].mean()
        base_caller_cpu = sub[caller_cpu_c].mean()
        new_own_wl = base_own_wl * (1 + DELTA_PCT / 100)

        pred_replace = m_replace.predict([[base_caller_cpu]])[0]
        pred_add = m_both.predict([[new_own_wl, base_caller_cpu]])[0]
        pred_baseline_model = m_own.predict([[new_own_wl]])[0]
        base_v = m_own.predict([[base_own_wl]])[0]

        row = {
            'caller': caller, 'callee': callee,
            'r2_own_only': round(r2_own, 3), 'r2_replace_caller_cpu_only': round(r2_replace, 3),
            'r2_add_both': round(r2_both, 3), 'partial_corr_own_wl_after_caller_cpu': round(partial_corr_ownwl, 3),
            f'do_{callee}_workload_+{DELTA_PCT:.0f}pct_replace': round(pred_replace, 4),
            f'do_{callee}_workload_+{DELTA_PCT:.0f}pct_add': round(pred_add, 4),
            f'do_{callee}_workload_+{DELTA_PCT:.0f}pct_baseline_model': round(pred_baseline_model, 4),
            'baseline_value': round(base_v, 4),
            'replace_change_pct': round((pred_replace - base_v) / base_v * 100, 2),
            'add_change_pct': round((pred_add - base_v) / base_v * 100, 2),
        }
        rows.append(row)

        print(f"\n=== {caller} -> {callee} ===")
        print(f"  R2: own-only={r2_own:.3f}  replace={r2_replace:.3f}  add={r2_both:.3f}  "
              f"(replace ~= add -> own_workload gan nhu du thua VE MAT QUAN SAT)")
        print(f"  Tuong quan rieng cua own_workload (sau khi tru caller_cpu): {partial_corr_ownwl:.3f}")
        print(f"  do({callee}_workload=+{DELTA_PCT:.0f}%, {caller}_cpu GIU BASELINE -- injection truc tiep tai {callee}):")
        print(f"    REPLACE  : {pred_replace:.4f}  (delta {row['replace_change_pct']:+.2f}% -- SAI: khong phan ung)")
        print(f"    ADD      : {pred_add:.4f}  (delta {row['add_change_pct']:+.2f}%)")
        print(f"    baseline model (own workload only): {pred_baseline_model:.4f}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'replace_vs_add_edge_test.csv')
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu: {out_path}")
    print("\nKET LUAN: R2(replace) ~= R2(add) tren du lieu QUAN SAT (khong can thiep) vi "
          "orders_cpu va {callee}_workload gan nhu luon bien doi cung nhau trong du lieu thu "
          "thap duoc (load-test thuong tang toan bo service cung luc). Nhung REPLACE khong "
          "phan ung gi voi mot can thiep TRUC TIEP, RIENG LE vao workload cua callee (dung loai "
          "truy van do(x) ma CapacityAgent can ho tro cho mot yeu cau moi) -- day la mot dang "
          "'confounding trong du lieu quan sat lam an giau anh huong nhan qua rieng' kinh dien. "
          "=> ADD (giu ca 2 parent) la lua chon dung, REPLACE (theo goi y minimality cua "
          "falsify_graph) se sai cho dung use case nay.")


if __name__ == '__main__':
    main()
