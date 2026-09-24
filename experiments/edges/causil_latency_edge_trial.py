# -*- coding: utf-8 -*-
"""
CAUSIL LATENCY-BACKPROPAGATION EDGE -- FEASIBILITY TRIAL
============================================================
Kiem tra tinh kha thi cua MOT loai canh CausIL dinh nghia ma capacity_agent.py
hien KHONG co: L^B -> L^A (latency cua callee anh huong latency cua caller),
mang y nguyen tu CausIL (arXiv:2303.00554) -- KHONG doi bat ky phan nao khac
cua CausIL (khong doi node types, khong doi 3-loai-canh-lien-service, khong
doi cong thuc BIC).

Khac voi BACKPRESSURE_EDGES hien co trong capacity_agent.py (CPU->CPU, tu
nghi ra, khong co doi chieu tai lieu), day la canh CausIL da dinh nghia ro
trong Section "Inter-Service Dependencies": voi moi canh (caller, callee)
trong call graph, candidate lien-service GOM latency callee -> latency
caller (nguoc chieu voi huong goi), khong phai cung chieu nhu CPU.

CHI kiem tra tinh kha thi (co dang lam duoc, co dang giup gi khong) -- KHONG
dong vao src/ (san pham that). Neu kha thi, buoc tiep theo moi la tich hop
vao scm_edge_selector.py/capacity_agent.py that.

Phuong phap (bam sat CausIL, khong pha tron voi QueueingLatencyRegressor
cua rieng du an -- de tach bach cau hoi "co canh nay co giup khong" khoi
"dang ham nao dung"):
  1. Candidate: voi moi (caller, callee) trong graph phu thuoc, candidate
     la (callee_latency-50 -> caller_latency-50) -- neu ca 2 cot latency
     va caller_workload deu ton tai.
  2. Baseline model : caller_latency-50 ~ caller_workload
     Extended model  : caller_latency-50 ~ caller_workload + callee_latency-50
     Ca 2 deu LinearRegression(positive=True) -- dung "Linear" variant cua
     CausIL (ho co ca Poly2/Poly3, nhung Linear la ban co ban nhat, dung de
     kiem tinh kha thi truoc).
  3. Diem BIC (dung cong thuc CausIL): Score = n*log(RSS/n) + rho*k*log(n),
     rho=2 -- phan hang so (2*pi*sigma^2 term) bi loai vi khong doi thu tu
     so sanh giua 2 model tren CUNG mot target/n.
  4. Danh gia CA hai: BIC (tieu chi CausIL dung de CHON canh) VA held-out
     MAPE/R2 theo dung protocol OOD Gold Standard cua du an (67% thap ->
     33% cao) -- de so sanh truc tiep duoc voi ket qua evaluate_node_
     stability() da co, khong dua ra mot thuoc do rieng biet moi.

Dung: python experiments/causil_latency_edge_trial.py --system sockshop
      python experiments/causil_latency_edge_trial.py --system trainticket
"""

import argparse
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from taxonomy_builder import load_graph  # noqa: E402
from data_processor import load_multi_service_data, SERVICES, TRAINTICKET_SERVICES  # noqa: E402


def _bic(y_true: np.ndarray, y_pred: np.ndarray, k: int, rho: float = 2.0) -> float:
    """CausIL Eq.: Score = -2*log-L + rho*k*log(n). Voi nhieu Gauss, -2*log-L
    = n*log(2*pi*sigma^2) + n; ta bo phan hang so (khong doi thu tu so sanh
    giua 2 model cung n, cung target) va dung n*log(RSS/n) lam phan con lai
    -- dung tinh than "score-based, phat do phuc tap" cua BIC, khong phai
    gia tri log-likelihood tuyet doi (khong can thiet khi chi so sanh
    TUONG DOI 2 model)."""
    n = len(y_true)
    rss = float(np.sum((y_true - y_pred) ** 2))
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + rho * k * np.log(n)


def _mape(y_true, y_pred) -> float:
    yt, yp = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    m = yt != 0
    return float(np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100) if m.sum() > 0 else float('nan')


def candidate_latency_edges(graph_path: str, columns) -> list:
    """(caller, callee) tu graph phu thuoc, loc theo cot du lieu co san.
    Chieu SCM se la callee_latency -> caller_latency (NGUOC voi chieu goi
    caller->callee trong graph) -- dung dinh nghia CausIL "L^B -> L^A"."""
    adj, _ = load_graph(graph_path)
    cols = set(columns)
    out = []
    for caller, callees in adj.items():
        for callee in callees:
            if (f'{caller}_latency-50' in cols and f'{callee}_latency-50' in cols
                    and f'{caller}_workload' in cols):
                out.append((caller, callee))
    return out


def evaluate_candidate(df: pd.DataFrame, caller: str, callee: str, split_ratio=0.67) -> dict:
    target_col = f'{caller}_latency-50'
    wl_col = f'{caller}_workload'
    callee_lat_col = f'{callee}_latency-50'
    sub = df[[wl_col, callee_lat_col, target_col]].dropna()
    if len(sub) < 200:
        return None

    sub_sorted = sub.sort_values(wl_col).reset_index(drop=True)
    split = int(len(sub_sorted) * split_ratio)
    train, test = sub_sorted.iloc[:split], sub_sorted.iloc[split:]
    if len(test) < 10:
        return None

    y_train, y_test = train[target_col].values, test[target_col].values

    m_base = LinearRegression(positive=True).fit(train[[wl_col]].values, y_train)
    pred_base_train = m_base.predict(train[[wl_col]].values)
    pred_base_test = m_base.predict(test[[wl_col]].values)

    m_ext = LinearRegression(positive=True).fit(train[[wl_col, callee_lat_col]].values, y_train)
    pred_ext_train = m_ext.predict(train[[wl_col, callee_lat_col]].values)
    pred_ext_test = m_ext.predict(test[[wl_col, callee_lat_col]].values)

    bic_base = _bic(y_train, pred_base_train, k=2)   # intercept + 1 coef
    bic_ext = _bic(y_train, pred_ext_train, k=3)      # intercept + 2 coef

    return {
        'caller': caller, 'callee': callee,
        'bic_base': round(bic_base, 2), 'bic_ext': round(bic_ext, 2),
        'bic_selects_edge': bic_ext < bic_base,
        'r2_base_test': round(m_base.score(test[[wl_col]].values, y_test), 4),
        'r2_ext_test': round(m_ext.score(test[[wl_col, callee_lat_col]].values, y_test), 4),
        'mape_base_test': round(_mape(y_test, pred_base_test), 2),
        'mape_ext_test': round(_mape(y_test, pred_ext_test), 2),
        'callee_lat_coef': round(float(m_ext.coef_[1]), 6),
        'wl_coef_base': round(float(m_base.coef_[0]), 6),
        'n_train': len(train), 'n_test': len(test),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', choices=['sockshop', 'trainticket'], default='sockshop')
    args = ap.parse_args()

    if args.system == 'trainticket':
        graph_path = os.path.join(BASE_DIR, 'src', 'graph', 'trainticket_agent_graph.json')
        data_dir = os.path.join(BASE_DIR, 'data', 'raw', 'trainticket')
        df = load_multi_service_data(data_dir, system_type='trainticket')
    else:
        graph_path = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')
        df = load_multi_service_data(system_type='sockshop')

    print(f"[1] He thong: {args.system} -- {len(df):,} dong telemetry.")

    candidates = candidate_latency_edges(graph_path, df.columns)
    print(f"[2] Tim thay {len(candidates)} candidate canh L^B->L^A tu graph phu thuoc.")

    rows = []
    for caller, callee in candidates:
        r = evaluate_candidate(df, caller, callee)
        if r is not None:
            rows.append(r)

    if not rows:
        print("[KHONG CO KET QUA] Khong candidate nao du du lieu (>=200 dong, >=10 test).")
        return

    res = pd.DataFrame(rows).sort_values('bic_ext')
    pd.set_option('display.width', 160)
    pd.set_option('display.max_rows', 200)
    print(f"\n[3] Ket qua {len(res)} candidate (sap xep theo BIC extended):")
    print(res.to_string(index=False))

    n_bic_select = int(res['bic_selects_edge'].sum())
    n_r2_improve = int((res['r2_ext_test'] > res['r2_base_test']).sum())
    n_mape_improve = int((res['mape_ext_test'] < res['mape_base_test']).sum())
    n_coef_nonzero = int((res['callee_lat_coef'].abs() > 1e-6).sum())
    n_wl_coef_zero = int((res['wl_coef_base'].abs() < 1e-6).sum())

    print(f"\n[4] TOM TAT:")
    print(f"    BIC chon giu canh (BIC_ext < BIC_base): {n_bic_select}/{len(res)}")
    print(f"    R2 held-out cai thien:                   {n_r2_improve}/{len(res)}")
    print(f"    MAPE held-out cai thien:                 {n_mape_improve}/{len(res)}")
    print(f"    He so callee_latency khac 0:              {n_coef_nonzero}/{len(res)}")
    print(f"    (doi chieu) He so workload rieng da la 0: {n_wl_coef_zero}/{len(res)} "
          f"(day la nhom 'mechanism thoai hoa' da phat hien truoc -- canh moi co "
          f"cuu duoc nhung node nay khong?)")

    if n_wl_coef_zero > 0:
        rescued = res[(res['wl_coef_base'].abs() < 1e-6) & (res['callee_lat_coef'].abs() > 1e-6)]
        print(f"    -> Trong so do, {len(rescued)}/{n_wl_coef_zero} node co he so workload=0 "
              f"NHUNG he so callee_latency khac 0 (canh moi cho no mot tin hieu ma workload "
              f"rieng khong co).")


if __name__ == '__main__':
    main()
