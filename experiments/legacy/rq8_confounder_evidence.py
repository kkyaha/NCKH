# -*- coding: utf-8 -*-
"""
Fix 3 -- Co ton tai confounder ha tang dung chung khong?
=========================================================
RQ7 cho thay loi the cua cau truc hai tang la BEN VUNG NGOAI SUY, khong phai
khu confounding. Nhung do la vi do thi hien tai KHONG mo hinh hoa confounder
nao ca -- neu khong khai bao thi khong the do duoc tac dung cua viec khu no.

Cau hoi o day: trong du lieu that, co ton tai mot backdoor path chua duoc mo
hinh hoa hay khong? Neu CO, thi Fix 3 (them node nguyen nhan chung) la huong
kha thi de toan tu do() co noi dung -- vi luc do moi co backdoor DE MA CHAN.
Neu KHONG, thi do() se van rong du sua the nao, va SCM nen bi ha cap han
trong bai bao.

Thiet ke
--------
Ung vien confounder: DONG-NODE (pod co-location). Hai service nam cung mot
node Kubernetes chia se CPU/memory/IO cua node do, nen tai cua cai nay anh
huong cai kia QUA MOT DUONG KHONG NAM TRONG CALL GRAPH.

Phep do: tuong quan RIENG PHAN giua R_A va R_B sau khi da khu di cac cha nhan
qua da khai bao (W_A va W_B). Neu do thi hai tang la DU, phan du phai gan doc
lap. Neu cap DONG-NODE co tuong quan rieng phan cao hon han cap KHAC-NODE, do
la bang chung truc tiep cho mot backdoor path bi bo sot.

Chi dung du lieu TRUOC inject (che do quan sat binh thuong).

Output: data/processed/scm_results/rq8_confounder_evidence.csv
"""

import os
import sys
import warnings
import itertools
from collections import defaultdict

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'RE2-SS')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
from rq7_interventional_validity import _iter_runs, SERVICES, MIN_PRE

# Cap co canh truc tiep trong call graph SockShop -- de tach "co duong nhan qua"
# khoi "chi dong-node". Lay tu sockshop_agent_graph.json (Tier-1).
CALL_EDGES = {
    ('front-end', 'catalogue'), ('front-end', 'carts'), ('front-end', 'user'),
    ('front-end', 'orders'), ('orders', 'carts'), ('orders', 'payment'),
    ('orders', 'shipping'), ('orders', 'user'),
}


def _service_of_pod(pod: str) -> str:
    """'carts-7d648bc7b8-q2xtl' -> 'carts'; bo 2 hau to hash cua ReplicaSet."""
    parts = pod.split('-')
    for cut in (2, 1):
        cand = '-'.join(parts[:-cut]) if len(parts) > cut else None
        if cand in SERVICES:
            return cand
    return pod if pod in SERVICES else None


def _node_map(run_dir):
    """service -> node_name, tu pod-node-*.csv."""
    for fn in ('pod-node-1.csv', 'pod-node-2.csv'):
        p = os.path.join(run_dir, fn)
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        m = {}
        for pod, node in zip(df['POD'], df['NODE_NAME']):
            svc = _service_of_pod(str(pod))
            if svc:
                m[svc] = str(node)
        if m:
            return m
    return {}


def partial_corr(x, y, Z):
    """Tuong quan giua x va y sau khi khu tuyen tinh anh huong cua cac cot Z."""
    if len(x) < 30:
        return np.nan
    Z = np.asarray(Z, float)
    rx = x - LinearRegression().fit(Z, x).predict(Z)
    ry = y - LinearRegression().fit(Z, y).predict(Z)
    if np.allclose(rx.std(), 0) or np.allclose(ry.std(), 0):
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def run():
    rows = []
    for scenario, target, fault, run_id, df, it in _iter_runs():
        pre = df[df['time'] < it]
        if len(pre) < MIN_PRE:
            continue
        nmap = _node_map(os.path.join(RAW_DIR, scenario, run_id))
        if len(nmap) < 4:
            continue
        for metric in ('cpu', 'mem'):
            for A, B in itertools.combinations(SERVICES, 2):
                cols = [f'{A}_{metric}', f'{B}_{metric}', f'{A}_workload', f'{B}_workload']
                if not all(c in pre.columns for c in cols):
                    continue
                sub = pre[cols].dropna()
                if len(sub) < MIN_PRE:
                    continue
                rA, rB, wA, wB = (sub[c].values for c in cols)
                # khu CA HAI cha nhan qua da khai bao trong do thi hai tang
                pc = partial_corr(rA, rB, np.column_stack([wA, wB]))
                if not np.isfinite(pc):
                    continue
                if A not in nmap or B not in nmap:
                    continue
                rows.append({
                    'scenario': scenario, 'run_id': run_id, 'metric': metric,
                    'svc_a': A, 'svc_b': B,
                    'same_node': nmap[A] == nmap[B],
                    'call_edge': (A, B) in CALL_EDGES or (B, A) in CALL_EDGES,
                    'partial_corr': round(pc, 4),
                    'abs_partial_corr': round(abs(pc), 4),
                    'n': len(sub),
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'rq8_confounder_evidence.csv'), index=False)
    return out


def report(d):
    if d.empty:
        print("khong co du lieu")
        return
    print(f"n = {len(d)} (cap service x run x metric)\n")
    print("=" * 76)
    print("  |Tuong quan rieng phan| R_A~R_B  SAU khi khu W_A, W_B")
    print("  (do thi hai tang du  =>  phan du phai gan 0)")
    print("=" * 76)
    print(f"  {'nhom':34s} {'n':>6s} {'median':>8s} {'mean':>8s}")
    for label, sub in [
        ('TAT CA', d),
        ('  dong-node', d[d.same_node]),
        ('  khac-node', d[~d.same_node]),
        ('khong co canh call-graph', d[~d.call_edge]),
        ('  dong-node, khong canh', d[(~d.call_edge) & d.same_node]),
        ('  khac-node, khong canh', d[(~d.call_edge) & (~d.same_node)]),
    ]:
        if len(sub) == 0:
            continue
        print(f"  {label:34s} {len(sub):6d} {sub.abs_partial_corr.median():8.3f} "
              f"{sub.abs_partial_corr.mean():8.3f}")

    print("\n" + "=" * 76)
    print("  Kiem dinh: dong-node co tuong quan du CAO HON khac-node khong?")
    print("  (chi tren cap KHONG co canh call-graph -> loai duong nhan qua that)")
    print("=" * 76)
    nc = d[~d.call_edge]
    a = nc[nc.same_node].abs_partial_corr.dropna()
    b = nc[~nc.same_node].abs_partial_corr.dropna()
    if len(a) >= 10 and len(b) >= 10:
        u, p = stats.mannwhitneyu(a, b, alternative='greater')
        # hieu ung: xac suat mot cap dong-node co |pc| lon hon mot cap khac-node
        auc = u / (len(a) * len(b))
        print(f"  dong-node n={len(a)} median={a.median():.3f} | "
              f"khac-node n={len(b)} median={b.median():.3f}")
        print(f"  Mann-Whitney U (one-sided): p={p:.3g}, common-language effect={auc:.3f}")
        verdict = ("CO bang chung backdoor ha tang -> Fix 3 kha thi"
                   if p < 0.05 and auc > 0.56 else
                   "KHONG du bang chung -> Fix 3 kho co loi ich do duoc")
        print(f"  -> {verdict}")
    else:
        print("  khong du mau de kiem dinh")

    print("\n  Ghi chu doc ket qua: |pc| lon tren cap KHONG co canh call-graph nghia la")
    print("  do thi hai tang BO SOT mot duong phu thuoc -- ung vien cho node nguyen")
    print("  nhan chung. |pc| ~ 0 nghia la do thi hien tai da du, va do() se van rong.")


if __name__ == '__main__':
    report(run())
