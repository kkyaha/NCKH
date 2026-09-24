# -*- coding: utf-8 -*-
"""
Fix 2 -- Toan tu do() co noi dung khi can thiep tai NODE NOI BO khong?
=======================================================================
Day la phep kiem quyet dinh cho huong SCM+do.

Lap luan
--------
Trong cau hinh hien tai (can thiep tai gateway, node in-degree=0) khong co
backdoor path, nen P(Y|do(X)) = P(Y|X) dong nhat va do() rong. Nhung neu can
thiep tai mot node NOI BO -- vi du do(R_V) voi V la mot service co cha -- thi
R_V co cha W_V, ton tai backdoor  R_V <- W_V -> ... -> R_B, va cong thuc
backdoor cho:

    E[R_B | do(R_V = v)] = E_{W_V ~ P(W_V)} [ E[R_B | R_V = v, W_V] ]

Khac han voi uoc luong quan sat tho E[R_B | R_V = v], von hap thu ca duong
backdoor. Neu do-calculus co noi dung o day, uoc luong DA HIEU CHINH phai
chuyen sang che do can thiep TOT HON uoc luong tho.

Ground truth
------------
Fault injection cua RCAEval vao service V CHINH LA mot can thiep tren R_V.
Fit tren giai doan truoc inject (quan sat), du bao giai doan sau inject
(can thiep), doi chieu R_B do thuc.

Ba uoc luong so sanh (deu fit CHI tren du lieu truoc inject):
  naive      : R_B ~ R_V                  -- quan sat tho, hap thu backdoor
  backdoor   : R_B ~ R_V + W_V, roi lay ky vong theo P_pre(W_V)
                                          -- dung cong thuc backdoor
  structural : R_B ~ W_B                  -- phuong trinh cau truc hai tang
                                             (baseline cua RQ7)

Doc ket qua
-----------
  backdoor << naive  -> do() CO noi dung o can thiep node noi bo -> Fix 2 kha
                        thi, va SCM co the giu vi tri dong gop.
  backdoor ~= naive  -> hieu chinh backdoor khong cai thien gi -> do() rong
                        ca o node noi bo -> SCM nen bi ha cap trong bai bao.

Output: data/processed/scm_results/rq9_backdoor_adjustment.csv
"""

import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
from rq7_interventional_validity import _iter_runs, mape, SERVICES, MIN_PRE, MIN_POST


def run():
    rows = []
    for scenario, target, fault, run_id, df, it in _iter_runs():
        # target la node NOI BO neu no co cha trong call graph.
        # front-end la gateway (in-degree 0) -> bo qua; o RE2-SS cung khong co
        # scenario nao tiem vao front-end.
        if target == 'front-end':
            continue
        pre, post = df[df['time'] < it], df[df['time'] >= it]
        if len(pre) < MIN_PRE or len(post) < MIN_POST:
            continue

        for metric in ('cpu', 'mem'):
            rV, wV = f'{target}_{metric}', f'{target}_workload'
            if rV not in df.columns or wV not in df.columns:
                continue
            for B in SERVICES:
                if B == target:
                    continue
                rB, wB = f'{B}_{metric}', f'{B}_workload'
                if rB not in df.columns or wB not in df.columns:
                    continue
                cols = [rV, wV, rB, wB]
                sp, so = pre[cols].dropna(), post[cols].dropna()
                if len(sp) < MIN_PRE or len(so) < MIN_POST:
                    continue

                y_pre, y_post = sp[rB].values, so[rB].values

                # (1) naive: R_B ~ R_V
                m_nv = LinearRegression().fit(sp[[rV]].values, y_pre)
                p_nv = m_nv.predict(so[[rV]].values)

                # (2) backdoor: R_B ~ R_V + W_V, lay ky vong theo P_pre(W_V).
                # Voi mo hinh tuyen tinh, E_w[a*v + b*w + c] = a*v + b*E_pre[w] + c,
                # tuc thay W_V quan sat duoc SAU can thiep bang trung binh TRUOC
                # can thiep -- dung y nghia "do() cat dut R_V khoi cha cua no,
                # nhung W_V giu phan phoi tu nhien cua no".
                m_bd = LinearRegression().fit(sp[[rV, wV]].values, y_pre)
                w_bar = sp[wV].mean()
                X_bd = np.column_stack([so[rV].values, np.full(len(so), w_bar)])
                p_bd = m_bd.predict(X_bd)

                # (3) structural: R_B ~ W_B  (baseline hai tang cua RQ7)
                m_st = LinearRegression(positive=True).fit(sp[[wB]].values, y_pre)
                p_st = m_st.predict(so[[wB]].values)

                rows.append({
                    'scenario': scenario, 'target': target, 'fault': fault,
                    'run_id': run_id, 'metric': metric, 'other_service': B,
                    'n_pre': len(sp), 'n_post': len(so),
                    'naive_post': mape(y_post, p_nv),
                    'backdoor_post': mape(y_post, p_bd),
                    'structural_post': mape(y_post, p_st),
                    'naive_pre': mape(y_pre, m_nv.predict(sp[[rV]].values)),
                    'backdoor_pre': mape(y_pre, m_bd.predict(sp[[rV, wV]].values)),
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'rq9_backdoor_adjustment.csv'), index=False)
    return out


def report(d):
    if d.empty:
        print("khong co du lieu")
        return
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['naive_post', 'backdoor_post', 'structural_post'])
    print(f"n = {len(d)} (can thiep tai node noi bo x service khac x metric)\n")
    print("=" * 74)
    print("  MAPE SAU can thiep (fit chi tren du lieu truoc can thiep)")
    print("=" * 74)
    print(f"  {'uoc luong':32s} {'median':>10s} {'mean':>10s}")
    for lab, c in [('naive       E[R_B | R_V]', 'naive_post'),
                   ('backdoor    adjusted for W_V', 'backdoor_post'),
                   ('structural  E[R_B | W_B]', 'structural_post')]:
        print(f"  {lab:32s} {d[c].median():10.2f} {d[c].mean():10.2f}")

    print("\n" + "=" * 74)
    print("  Hieu chinh backdoor co cai thien so voi quan sat tho khong?")
    print("=" * 74)
    diff = (d['backdoor_post'] - d['naive_post']).dropna()
    st, p = stats.wilcoxon(diff)
    med = float(np.median(diff))
    better = int((d['backdoor_post'] < d['naive_post']).sum())
    print(f"  median(backdoor - naive) = {med:+.3f}   p = {p:.3g}")
    print(f"  backdoor tot hon tren {better}/{len(d)} cap ({100*better/len(d):.1f}%)")
    # Nguong phai dua tren CO LON HIEU UNG, khong chi p-value: o n=1080 mot
    # khac biet vo nghia van dat p<0.05. Tieu chi: ty le thang phai lech ro
    # khoi dong xu (>=60%) VA median cua hieu phai khac 0 mot cach dang ke.
    win_rate = better / len(d)
    meaningful = (p < 0.05) and (win_rate >= 0.60) and (abs(med) > 0.5)
    if meaningful and med < 0:
        rel = 100 * (1 - d['backdoor_post'].median() / d['naive_post'].median())
        print(f"  -> backdoor TOT HON, giam {rel:.1f}% MAPE trung vi")
        print("  -> do() CO noi dung tai node noi bo: Fix 2 KHA THI")
    elif meaningful and med > 0:
        print("  -> backdoor TE HON quan sat tho: hieu chinh phan tac dung")
    else:
        print(f"  -> ty le thang {100*win_rate:.1f}% ~ dong xu, median hieu ~ 0:")
        print("     hieu chinh backdoor KHONG cai thien gi. Khong co backdoor dang ke")
        print("     de chan -> do() van rong ca o node noi bo -> Fix 2 KHONG kha thi.")
        if p < 0.05:
            print(f"     (p={p:.3g} dat nguong chi vi n={len(d)} lon; co lon hieu ung ~ 0)")

    print("\n  Doi chieu voi baseline cau truc (RQ7):")
    b_vs_s = (d['backdoor_post'] - d['structural_post']).dropna()
    st2, p2 = stats.wilcoxon(b_vs_s)
    print(f"  median(backdoor - structural) = {np.median(b_vs_s):+.3f}  p={p2:.3g}")

    print("\n  Theo loai fault (median MAPE sau can thiep):")
    print(d.groupby('fault')[['naive_post', 'backdoor_post', 'structural_post']]
          .median().round(2).to_string())


if __name__ == '__main__':
    report(run())
