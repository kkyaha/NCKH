# -*- coding: utf-8 -*-
"""
TONG QUAT HOA: tu dong chon covariate bo sung (khong con chan doan tay tung node)
====================================================================================
Boi canh: hoi thoai truoc da CHAN DOAN TAY cho dung 2 node (carts_latency-50,
orders_latency-50) va tim ra 1 canh moi (carts_latency->orders_latency, MAPE
100.4%->7.6%). Nhung quy trinh do (i) chon tay dua tren doc code graph, (ii)
CHON VA DANH GIA TREN CUNG MOT TAP TEST -- dung loai "vacuity" ma
RQ_FRAMEWORK_V3.md tu canh bao (data snooping: thu nhieu canh ung vien roi
bao cao canh tot nhat tren chinh tap dùng de chon). Script nay SUA loi do va
TONG QUAT HOA thanh 1 thu tuc chay tu dong tren MOI node cua CA HAI he thong
(SockShop 7 service, Train Ticket 28 service), dung DUY NHAT dau vao la:
  (a) topology graph co san (*_agent_graph.json -- da trich xuat tu docker-
      compose, KHONG can du lieu tracing raw vi repo nay khong co trace log
      muc span/request, chi co time-series metrics gop theo run),
  (b) time-series metrics (dang moi he thong deu co: CPU/Mem/Socket/Latency
      theo workload).

2 huong ung vien (CHINH XAC 2 huong da kiem chung thuc nghiem, KHONG suy dien
them huong moi chua kiem chung):
  - CPU/Memory: cha bo sung = CPU cua CALLER truc tiep (huong nghen tai
    nguyen nguoc dong -- da xac nhan dung cho orders_cpu->carts_cpu).
  - Latency: cha bo sung = Latency cua CALLEE truc tiep (huong lan truyen
    do tre xuoi dong -- da xac nhan dung cho carts_latency->orders_latency).

SUA LOI PHUONG PHAP: chia 3 tap theo thu tu Workload tang dan (khong xao
tron, giu tinh chat OOD):
  train (50% thap nhat)      -- fit he so
  validation (tiep theo 20%) -- CHON co giu covariate bo sung khong (nguong
                                 cai thien MAPE tuong doi >= IMPROVE_THRESHOLD)
  test (30% cao nhat, CHUA TUNG dung de chon) -- BAO CAO MAPE CUOI CUNG, 1 lan
Day la diem khac biet mau chot so voi chan doan tay truoc: neu 1 canh chi
"trung" tren tap validation do ngau nhien (nhieu ung vien, it du lieu), no se
KHONG con loi the tren tap test doc lap -- phoi bay dung nhung phat hien gia.

Output: data/processed/scm_results/generalized_covariate_selection_<system>.csv
"""

import os
import sys
import json
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data, SERVICES, TRAINTICKET_SERVICES

IMPROVE_THRESHOLD = 0.15  # >=15% giam tuong doi tren VALIDATION moi duoc giu canh


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def load_graph_edges(graph_path, services):
    with open(graph_path, 'r', encoding='utf-8') as f:
        g = json.load(f)
    edges = [(e['source'], e['target']) for e in g.get('edges', [])
             if e['source'] in services and e['target'] in services]
    return edges


def three_way_split(d, feat_for_sort):
    d = d.sort_values(feat_for_sort).reset_index(drop=True)
    n = len(d)
    i1, i2 = int(n * 0.50), int(n * 0.70)
    return d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]


def eval_mape(dtr, deval, feats, target, positive=True):
    if dtr[feats].nunique().min() < 2 or len(deval) < 5:
        return None
    m = LinearRegression(positive=positive).fit(dtr[feats], dtr[target])
    deval2 = deval.copy()
    q = min(8, deval2[feats[0]].nunique())
    if q < 2:
        return None
    deval2['bkt'] = pd.qcut(deval2[feats[0]], q=q, duplicates='drop')
    bkt = deval2.groupby('bkt', observed=True)[feats + [target]].mean()
    pred = m.predict(bkt[feats])
    return _mape(bkt[target], pred), m.coef_


def run_system(system_type, services, graph_path, out_suffix):
    print(f"\n{'='*90}\n  HE THONG: {system_type.upper()} ({len(services)} services)\n{'='*90}")
    edges = load_graph_edges(graph_path, services)
    callers = {}   # callee -> [callers]
    callees = {}   # caller -> [callees]
    for src, tgt in edges:
        callers.setdefault(tgt, []).append(src)
        callees.setdefault(src, []).append(tgt)

    df = load_multi_service_data(None, system_type=system_type)
    rows = []

    for svc in services:
        wlc = f'{svc}_workload'
        if wlc not in df.columns:
            continue

        # ---- CPU: ung vien = CPU cua caller truc tiep ----
        cpuc = f'{svc}_cpu'
        if cpuc in df.columns:
            base_data = df[[wlc, cpuc]].dropna()
            if len(base_data) >= 300:
                dtr, dval, dte = three_way_split(base_data, wlc)
                base_val = eval_mape(dtr, dval, [wlc], cpuc)
                base_test = eval_mape(dtr, dte, [wlc], cpuc)
                if base_val and base_test:
                    best_feat, best_val_mape = None, base_val[0]
                    for caller in callers.get(svc, []):
                        cand = f'{caller}_cpu'
                        if cand not in df.columns:
                            continue
                        d2 = df[[wlc, cand, cpuc]].dropna()
                        if len(d2) < 300:
                            continue
                        dtr2, dval2, dte2 = three_way_split(d2, wlc)
                        r = eval_mape(dtr2, dval2, [wlc, cand], cpuc)
                        if r and r[0] < best_val_mape * (1 - IMPROVE_THRESHOLD):
                            best_feat, best_val_mape = cand, r[0]
                    if best_feat:
                        d2 = df[[wlc, best_feat, cpuc]].dropna()
                        dtr2, dval2, dte2 = three_way_split(d2, wlc)
                        final = eval_mape(dtr2, dte2, [wlc, best_feat], cpuc)
                    else:
                        final = base_test
                    rows.append({
                        'system': system_type, 'node': cpuc, 'metric_type': 'cpu',
                        'baseline_test_mape': round(base_test[0], 2),
                        'chosen_extra_parent': best_feat or '(none)',
                        'final_test_mape': round(final[0], 2) if final else None,
                        'n_candidates_tried': len(callers.get(svc, [])),
                    })

        # ---- Latency: ung vien = Latency cua callee truc tiep ----
        latc = f'{svc}_latency-50'
        if latc in df.columns:
            base_data = df[[wlc, latc]].dropna()
            if len(base_data) >= 300:
                dtr, dval, dte = three_way_split(base_data, wlc)
                base_val = eval_mape(dtr, dval, [wlc], latc)
                base_test = eval_mape(dtr, dte, [wlc], latc)
                if base_val and base_test:
                    best_feat, best_val_mape = None, base_val[0]
                    for callee in callees.get(svc, []):
                        cand = f'{callee}_latency-50'
                        if cand not in df.columns:
                            continue
                        d2 = df[[wlc, cand, latc]].dropna()
                        if len(d2) < 300:
                            continue
                        dtr2, dval2, dte2 = three_way_split(d2, wlc)
                        r = eval_mape(dtr2, dval2, [wlc, cand], latc)
                        if r and r[0] < best_val_mape * (1 - IMPROVE_THRESHOLD):
                            best_feat, best_val_mape = cand, r[0]
                    if best_feat:
                        d2 = df[[wlc, best_feat, latc]].dropna()
                        dtr2, dval2, dte2 = three_way_split(d2, wlc)
                        final = eval_mape(dtr2, dte2, [wlc, best_feat], latc)
                    else:
                        final = base_test
                    rows.append({
                        'system': system_type, 'node': latc, 'metric_type': 'latency',
                        'baseline_test_mape': round(base_test[0], 2),
                        'chosen_extra_parent': best_feat or '(none)',
                        'final_test_mape': round(final[0], 2) if final else None,
                        'n_candidates_tried': len(callees.get(svc, [])),
                    })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f'generalized_covariate_selection_{out_suffix}.csv')
    df_out.to_csv(out_path, index=False)

    print(df_out.to_string(index=False))
    n_edges_kept = (df_out['chosen_extra_parent'] != '(none)').sum()
    print(f"\n[{system_type}] Tu dong giu {n_edges_kept}/{len(df_out)} canh bo sung "
          f"(nguong cai thien tren validation >= {IMPROVE_THRESHOLD*100:.0f}%).")
    print(f"[{system_type}] MAPE trung binh: baseline={df_out['baseline_test_mape'].mean():.2f}%  "
          f"final={df_out['final_test_mape'].mean():.2f}%")
    print(f"[OK] Da luu: {out_path}")
    return df_out


if __name__ == '__main__':
    r1 = run_system('sockshop', SERVICES,
                     os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json'), 'sockshop')
    r2 = run_system('trainticket', TRAINTICKET_SERVICES,
                     os.path.join(BASE_DIR, 'src', 'graph', 'trainticket_agent_graph.json'), 'trainticket')
