# -*- coding: utf-8 -*-
"""
RQ7: Fault injection lam GROUND TRUTH cho mot CAN THIEP thuc su
================================================================
Cau hoi: cau truc nhan qua co gia tri gi HON hoi quy khi he thong BI CAN
THIEP that -- thay vi chi duoc quan sat?

Vi sao can thi nghiem nay
-------------------------
RQ2 cho thay co che SCM duoc trien khai TRUNG KHIT hoi quy tuyen tinh tren
86/105 cap. Do khong phai trung hop, ma la tat yeu toan hoc:

    P(Y | do(X)) != P(Y | X)  <=>  ton tai backdoor path

Do thi Tier-2 hien tai la `W_j -> R_j` (hai node), va diem tiem la gateway
(in-degree = 0). Do thi hai node KHONG co backdoor, va node goc KHONG co cha.
Vay nen trong dung cau hinh dang do, E[Y|do(X)] = E[Y|X] DONG NHAT -- toan tu
do khong the khac regression du cai dat the nao.

Nhung dieu do KHONG co nghia toan tu do vo dung. No co nghia phep do hien tai
dat o dung cho ma do khong co noi dung. Thi nghiem nay dat phep do vao cho do
CO noi dung.

Thiet ke
--------
RCAEval tiem fault vao mot service muc tieu A tai `inject_time`. Fault injection
CHINH LA mot can thiep: no thay doi trang thai cua A ma KHONG thay doi tai
toan cuc. Do la dung dieu kien de tach do() khoi P(.|.).

Voi moi cap (A = service bi tiem, B = service khac):

  * Fit CHI tren du lieu TRUOC inject (giai doan quan sat binh thuong):
      - M_causal : R_B ~ W_B      (dung phuong trinh cau truc cua SCM:
                                   resource cua B phu thuoc workload cua B)
      - M_assoc  : R_B ~ R_A      (tuong quan quan sat manh nhat, nhung di qua
                                   backdoor  R_A <- tai toan cuc -> R_B)
      - M_assoc_w: R_B ~ W_A      (workload upstream cua A)
  * Du bao tren du lieu SAU inject (giai doan can thiep), doi chieu R_B do thuc.

Du doan cua ly thuyet: M_assoc hap thu tuong quan backdoor nen phai bi CHECH
co he thong duoi can thiep, trong khi M_causal thi khong. Neu dung, ta se thay
M_assoc TOT HON truoc can thiep nhung TE HON sau can thiep -- day la dau hieu
kinh dien cua confounding, va la bang chung truc tiep cho gia tri cua cau truc
nhan qua.

Neu KHONG thay dau hieu do: ket luan trung thuc la o topology/bai toan nay cau
truc nhan qua khong mang lai gia tri du bao, va dong gop cua bai bao nam hoan
toan o lop guardrail LLM. Ca hai ket cuc deu la ket qua bao cao duoc; diem
quan trong la gia thuyet nay KHA SAI, khac voi "du bao cho feature chua xay"
von khong the kiem chung.

Output: data/processed/scm_results/rq7_interventional_validity.csv
"""

import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'RE2-SS')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
METRIC_COLS = ['cpu', 'mem']       # socket/latency phan ung khac, tach rieng sau
MIN_PRE, MIN_POST = 120, 60        # so diem toi thieu moi phia de fit/test


def mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = (y != 0) & np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100) if m.sum() else np.nan


def _iter_runs():
    """Sinh (scenario, target_service, fault_type, run_id, df, inject_time)."""
    if not os.path.isdir(RAW_DIR):
        return
    for scenario in sorted(os.listdir(RAW_DIR)):
        sp = os.path.join(RAW_DIR, scenario)
        if not os.path.isdir(sp):
            continue
        # ten thu muc: "<service>_<fault>" -- service co the chua dau '-'
        target, _, fault = scenario.rpartition('_')
        if target not in SERVICES:
            continue
        for run_id in sorted(os.listdir(sp)):
            rp = os.path.join(sp, run_id)
            mp, ip = os.path.join(rp, 'simple_metrics.csv'), os.path.join(rp, 'inject_time.txt')
            if not (os.path.exists(mp) and os.path.exists(ip)):
                continue
            try:
                with open(ip) as f:
                    it = int(f.read().strip())
                df = pd.read_csv(mp)
                tc = 'imte' if 'imte' in df.columns else ('time' if 'time' in df.columns else None)
                if tc is None:
                    continue
                df = df.rename(columns={tc: 'time'})
                yield scenario, target, fault, run_id, df, it
            except Exception:
                continue


def _fit_predict(df_pre, df_post, xcol, ycol):
    """Fit tuyen tinh khong am tren PRE, du bao tren ca PRE va POST.

    Dung LinearRegression(positive=True) -- dung co che SCM da trien khai --
    de so sanh la ve CAU TRUC DO THI (dung bien nao lam cha), khong bi lan
    voi khac biet ve lop mo hinh.
    """
    Xtr, ytr = df_pre[[xcol]].values, df_pre[ycol].values
    if len(Xtr) < MIN_PRE or np.allclose(Xtr.std(), 0):
        return None
    m = LinearRegression(positive=True).fit(Xtr, ytr)
    return {
        'mape_pre': mape(ytr, m.predict(Xtr)),
        'mape_post': mape(df_post[ycol].values, m.predict(df_post[[xcol]].values)),
    }


def run():
    rows = []
    n_runs = 0
    for scenario, target, fault, run_id, df, it in _iter_runs():
        pre, post = df[df['time'] < it], df[df['time'] >= it]
        if len(pre) < MIN_PRE or len(post) < MIN_POST:
            continue
        n_runs += 1
        wl_A = f'{target}_workload'
        for metric in METRIC_COLS:
            r_A = f'{target}_{metric}'
            if r_A not in df.columns or wl_A not in df.columns:
                continue
            for B in SERVICES:
                if B == target:
                    continue
                r_B, wl_B = f'{B}_{metric}', f'{B}_workload'
                if r_B not in df.columns or wl_B not in df.columns:
                    continue
                sub_pre = pre[[wl_A, r_A, wl_B, r_B]].dropna()
                sub_post = post[[wl_A, r_A, wl_B, r_B]].dropna()
                if len(sub_pre) < MIN_PRE or len(sub_post) < MIN_POST:
                    continue

                variants = {
                    'causal_RB~WB': (wl_B, r_B),   # phuong trinh cau truc cua SCM
                    'assoc_RB~RA': (r_A, r_B),     # tuong quan quan sat qua backdoor
                    'assoc_RB~WA': (wl_A, r_B),    # workload upstream cua A
                }
                res = {}
                for name, (xc, yc) in variants.items():
                    out = _fit_predict(sub_pre, sub_post, xc, yc)
                    if out is None:
                        res = {}
                        break
                    res[name] = out
                if not res:
                    continue

                row = {
                    'scenario': scenario, 'target_service': target, 'fault_type': fault,
                    'run_id': run_id, 'metric': metric, 'other_service': B,
                    'n_pre': len(sub_pre), 'n_post': len(sub_post),
                }
                for name, out in res.items():
                    row[f'{name}_pre'] = round(out['mape_pre'], 3)
                    row[f'{name}_post'] = round(out['mape_post'], 3)
                    # degradation = phan sai so tang len khi he thong bi can thiep
                    row[f'{name}_degrade'] = round(out['mape_post'] - out['mape_pre'], 3)
                rows.append(row)

    out = pd.DataFrame(rows)
    dest = os.path.join(OUT_DIR, 'rq7_interventional_validity.csv')
    out.to_csv(dest, index=False)
    print(f"runs used: {n_runs} | rows: {len(out)} -> {dest}\n")
    return out


def report(df):
    if df.empty:
        print("khong co du lieu")
        return
    V = ['causal_RB~WB', 'assoc_RB~RA', 'assoc_RB~WA']
    print("=" * 78)
    print("  MAPE trung vi: TRUOC can thiep (quan sat) vs SAU can thiep")
    print("=" * 78)
    print(f"  {'predictor':16s} {'pre':>9s} {'post':>9s} {'degrade':>9s}  {'wins_post':>10s}")
    post_cols = [f'{v}_post' for v in V]
    best = df[post_cols].idxmin(axis=1)
    for v in V:
        w = int((best == f'{v}_post').sum())
        print(f"  {v:16s} {df[f'{v}_pre'].median():9.2f} {df[f'{v}_post'].median():9.2f} "
              f"{df[f'{v}_degrade'].median():9.2f}  {w:6d}/{len(df)}")

    print("\n" + "=" * 78)
    print("  Kiem dinh: causal co suy giam IT HON associational duoi can thiep?")
    print("=" * 78)
    for v in ['assoc_RB~RA', 'assoc_RB~WA']:
        d = (df['causal_RB~WB_degrade'] - df[f'{v}_degrade']).dropna()
        if len(d) < 10:
            continue
        stat, p = stats.wilcoxon(d)
        med = float(np.median(d))
        verdict = ("causal suy giam IT hon (ung ho gia thuyet)" if med < 0 and p < 0.05
                   else "causal suy giam NHIEU hon" if med > 0 and p < 0.05
                   else "khong khac biet co y nghia")
        print(f"  causal vs {v:14s} n={len(d):4d} median_diff={med:+8.3f} "
              f"p={p:.3g}  -> {verdict}")

    print("\n" + "=" * 78)
    print("  Dau hieu confounding kinh dien: assoc TOT HON truoc, TE HON sau")
    print("=" * 78)
    for v in ['assoc_RB~RA', 'assoc_RB~WA']:
        better_pre = df[f'{v}_pre'] < df['causal_RB~WB_pre']
        worse_post = df[f'{v}_post'] > df['causal_RB~WB_post']
        both = int((better_pre & worse_post).sum())
        print(f"  {v:14s}: tot hon truoc = {int(better_pre.sum()):4d}/{len(df)} | "
              f"te hon sau = {int(worse_post.sum()):4d}/{len(df)} | "
              f"CA HAI (dau hieu) = {both:4d}/{len(df)} ({100*both/len(df):.1f}%)")

    print("\n  Theo loai fault (median MAPE sau can thiep):")
    g = df.groupby('fault_type')[post_cols].median().round(2)
    print(g.to_string())


if __name__ == '__main__':
    report(run())
