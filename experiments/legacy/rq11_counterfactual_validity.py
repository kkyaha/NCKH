# -*- coding: utf-8 -*-
"""
Huong phan thuc (Pearl tang 3) -- co kha thi khong?
====================================================
Phan thuc khac can thiep o cho: can thiep LAY KY VONG theo phan phoi nhieu
(E[N] = 0), con phan thuc GIU CO DINH mot hien thuc nhieu cu the. Ba buoc:

  1. Abduction : suy nguoc nhieu N tu quan sat thuc te (W, R)
  2. Action    : sua co che, do(W = w')
  3. Prediction: tinh lai R voi CHINH nhieu da suy nguoc

Y nghia thuc te: thay vi hoi "trung binh mot service phan ung the nao khi tai
tang", ta hoi "DUNG service nay, dang chay o muc rieng cua no, se phan ung the
nao" -- dung cau hoi cua capacity planning.

Bay quan trong can tranh
------------------------
Mot hoi quy KHONG lam duoc phan thuc. Nhung mot ky su can than co the lam mot
thu GAN GIONG: cong phan du quan sat duoc vao du bao (residual carry-forward).
Neu phan thuc day du chi ngang bang cach do, thi bo may phan thuc lai roi vao
dung loi da gap o huong 2: ngu nghia dung nhung khong cho nang luc moi.

Vi vay so sanh BA uoc luong, khong phai hai:

  (a) marginal        : E[R_j | W_j]                  -- cau tra loi cua hoi quy
  (b) residual-carry  : E[R_j | W_j] + resid_j(t)     -- ky su can than
  (c) counterfactual  : lan truyen nhieu DA SUY NGUOC tai MOI node qua DAG
                        -- nhieu cua node thuong nguon di qua co che ha nguon

Chi khi (c) hon (b) thi lan truyen nhieu theo cau truc moi la nang luc that.
Neu (c) ~ (b) thi ket luan trung thuc: phan thuc o day cung khong cuu duoc
SCM khoi vi tri trang tri.

Giao thuc
---------
Fit tren giai doan TRUOC inject. Voi moi diem t trong giai doan sau inject:
suy nguoc nhieu tai t, roi du bao R tai t+LAG voi W quan sat duoc tai t+LAG.
Doi chieu R thuc do tai t+LAG. LAG > 0 de khong tu tra loi chinh minh.

Output: data/processed/scm_results/rq11_counterfactual_validity.csv
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

# Tier-1 cua SockShop: cha workload cua tung service (tu sockshop_agent_graph)
TIER1_PARENT = {
    'catalogue': 'front-end', 'carts': 'front-end', 'user': 'front-end',
    'orders': 'front-end', 'payment': 'orders', 'shipping': 'orders',
}
LAGS = [5, 15, 30]


def run():
    rows = []
    for scenario, target, fault, run_id, df, it in _iter_runs():
        pre, post = df[df['time'] < it], df[df['time'] >= it]
        if len(pre) < MIN_PRE or len(post) < MIN_POST:
            continue
        post = post.sort_values('time').reset_index(drop=True)

        for metric in ('cpu', 'mem'):
            for svc, parent in TIER1_PARENT.items():
                rj, wj, wp = f'{svc}_{metric}', f'{svc}_workload', f'{parent}_workload'
                if not all(c in df.columns for c in (rj, wj, wp)):
                    continue
                sp = pre[[rj, wj, wp]].dropna()
                so = post[[rj, wj, wp]].dropna().reset_index(drop=True)
                if len(sp) < MIN_PRE or len(so) < MIN_POST:
                    continue

                # Co che Tier-2: R_j = g(W_j) + N_R ; Tier-1: W_j = f(W_parent) + N_W
                g = LinearRegression(positive=True).fit(sp[[wj]].values, sp[rj].values)
                f = LinearRegression(positive=True).fit(sp[[wp]].values, sp[wj].values)

                W_j, W_p, R_j = so[wj].values, so[wp].values, so[rj].values
                # Abduction: nhieu suy nguoc tai tung thoi diem
                n_R = R_j - g.predict(so[[wj]].values)          # nhieu tai node resource
                n_W = W_j - f.predict(so[[wp]].values)          # nhieu tai node workload

                for lag in LAGS:
                    if len(so) <= lag + 10:
                        continue
                    t = np.arange(len(so) - lag)
                    tk = t + lag
                    y_true = R_j[tk]

                    # (a) marginal: chi dung W_j quan sat duoc tai t+lag
                    p_marg = g.predict(so[[wj]].values[tk])

                    # (b) residual-carry: cong phan du cua CHINH node do tai t
                    p_carry = p_marg + n_R[t]

                    # (c) counterfactual day du: giu CA HAI hien thuc nhieu co dinh,
                    #     tai tao W_j tu cha cua no roi day qua co che Tier-2.
                    #     Nhieu cua node thuong nguon (n_W) di qua g, dieu ma
                    #     residual-carry tai node resource khong the hien duoc.
                    W_j_cf = f.predict(so[[wp]].values[tk]) + n_W[t]
                    p_cf = g.predict(W_j_cf.reshape(-1, 1)) + n_R[t]

                    rows.append({
                        'scenario': scenario, 'fault': fault, 'run_id': run_id,
                        'metric': metric, 'service': svc, 'parent': parent, 'lag': lag,
                        'n': len(t),
                        'marginal': mape(y_true, p_marg),
                        'residual_carry': mape(y_true, p_carry),
                        'counterfactual': mape(y_true, p_cf),
                    })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'rq11_counterfactual_validity.csv'), index=False)
    return out


def report(d):
    if d.empty:
        print("khong co du lieu")
        return
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['marginal', 'residual_carry', 'counterfactual'])
    print(f"n = {len(d)}\n")
    print("=" * 72)
    print("  MAPE theo do tre (lag) -- fit tren truoc-inject, test sau-inject")
    print("=" * 72)
    g = d.groupby('lag')[['marginal', 'residual_carry', 'counterfactual']].median().round(3)
    print(g.to_string())

    print("\n" + "=" * 72)
    print("  Hai so sanh quyet dinh")
    print("=" * 72)
    for lab, a, b in [
        ('abduction co gia tri gi?      (carry vs marginal)', 'residual_carry', 'marginal'),
        ('lan truyen cau truc co them?  (cf vs carry)      ', 'counterfactual', 'residual_carry'),
    ]:
        diff = (d[a] - d[b]).dropna()
        if len(diff) < 10:
            continue
        st, p = stats.wilcoxon(diff)
        med = float(np.median(diff))
        win = int((d[a] < d[b]).sum())
        wr = win / len(d)
        rel = 100 * (1 - d[a].median() / d[b].median())
        meaningful = (p < 0.05) and (wr >= 0.60) and abs(med) > 0.01
        verdict = ('CAI THIEN dang ke' if meaningful and med < 0 else
                   'TE HON dang ke' if meaningful and med > 0 else
                   'khong khac biet dang ke')
        print(f"  {lab}")
        print(f"    median diff={med:+8.4f}  p={p:.3g}  win={100*wr:5.1f}%  "
              f"giam MAPE trung vi={rel:+.1f}%  -> {verdict}")

    print("\n  Ket luan doc duoc:")
    print("   - carry << marginal  : abduction CO gia tri (nhieu tu tuong quan thoi gian)")
    print("   - cf   << carry      : lan truyen nhieu theo CAU TRUC la nang luc that")
    print("   - cf   ~= carry      : phan thuc quy ve giu phan du -> khong cuu duoc SCM")


if __name__ == '__main__':
    report(run())
