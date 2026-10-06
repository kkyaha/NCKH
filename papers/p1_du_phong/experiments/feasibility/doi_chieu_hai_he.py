# -*- coding: utf-8 -*-
"""DOI CHIEU HE A va HE B: `rho_s` co tai tao duoc lan truyen Tier-1 khong?

VI SAO SCRIPT NAY TON TAI
=========================
`src/agents/orchestrator.py` va `papers/p1_du_phong/docs/HE_THONG.md` deu dua vao mot
cau de bien minh cho viec giu HAI ngan xep:

    "`rho_s` cua bo du doan kha thi tai tao DUNG lan truyen Tier-1 cua CapacityAgent,
     lech <= 0,5% tren moi service -- chung la MOT mo hinh viet o hai dang."

Cau do dang do CA lap luan "hai he la mot mo hinh", nhung **khong script nao trong repo
tinh ra con so 0,5%**. Day la dung mau loi ma chinh HE_THONG.md da tu canh bao voi
`R^2 = 0,0034` ("hang so viet cung, khong script nao tinh ra no"). Script nay dong cho ho do.

KET QUA DO (xem main): menh de DUNG hay SAI phu thuoc do o MUC NAO --
  * muc workload  W_s : lech toi 13-19%  -> cau nhu dang viet la SAI
  * muc su dung   u_s : lech <= 0,30%    -> cau DUNG neu noi ro la tren u
Ly do hoa giai: `user` lech W 13,25% nhung beta_user = 0,0674 rat nho va u_user ~ 0,078,
nen sai so tuong doi lon o W gan nhu khong dich u. Va nut nghen LUON la front-end, noi
lech dung 0,00% vi no chinh la goc cua ca hai cach tinh.

=> Phat bieu dung: "lech <= 0,5% TREN MUC SU DUNG u", khong phai "tren moi service".

HAI CACH TINH DUOC SO SANH
  he B (FeasibilityPredictor): W_s = rho_s * W_gateway,  rho_s = hoi quy QUA GOC
  he A (CapacityAgent Tier-1): W_s = NNLS(W_cha1, W_cha2, ...),  cha lay tu do thi goi that

Chay:  python papers/p1_du_phong/experiments/feasibility/doi_chieu_hai_he.py
Ra:    data/processed/scm_results/doi_chieu_hai_he.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo'
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
import feasibility_predictor as FP   # noqa: E402

TRAIN = os.path.join(BASE_DIR, 'data', 'raw', 'SS-TRAIN')
FROZEN = os.path.join(BASE_DIR, 'data', 'processed', 'frozen',
                      'predictions_frozen_RE2_P2_prosp2.json')
DO_THI = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')


def cha_tier1():
    """Cha Tier-1 cua moi service: lay tu CANH GOI THAT trong do thi, khong gan tay.
    Service khong co cha nao trong do thi -> coi gateway la cha (truong hop front-end)."""
    G = json.load(open(DO_THI, encoding='utf-8'))
    cha = {}
    for e in G['edges']:
        s, t = e['source'], e['target']
        if s in FP.SERVICES and t in FP.SERVICES:
            cha.setdefault(t, []).append(s)
    return {s: cha.get(s, [FP.GATEWAY]) for s in FP.SERVICES}


def main():
    j = json.load(open(FROZEN, encoding='utf-8'))
    M, C = j['mechanism'], j['params']['cores']
    tm = j['params']['train_max_rps']
    d = FP.load_runs(TRAIN)
    tr = d[FP.level_of(d) <= tm]
    cha = cha_tier1()
    wg = tr[f'{FP.GATEWAY}_workload'].to_numpy(dtype=float)

    hang = []
    for s in FP.SERVICES:
        y = tr[f'{s}_workload'].to_numpy(dtype=float)
        # --- he B: mot he so duy nhat, hoi quy QUA GOC theo workload gateway
        rho = float(wg @ y / (wg @ wg))
        pB = rho * wg
        # --- he A: NNLS tren workload cua CAC CHA THAT trong do thi goi
        P = cha[s]
        X = tr[[f'{p}_workload' for p in P]].to_numpy(dtype=float)
        pA = LinearRegression(positive=True, fit_intercept=False).fit(X, y).predict(X)

        m = y > 0
        lech_W = 100 * np.abs(pA[m] - pB[m]).mean() / y[m].mean()
        # --- chuyen qua muc su dung: day moi la dai luong vao PHAN QUYET
        uB = (M[s]['alpha'] + M[s]['beta'] * pB[m].mean()) / (100 * C[s])
        uA = (M[s]['alpha'] + M[s]['beta'] * pA[m].mean()) / (100 * C[s])
        lech_u = 100 * abs(uA - uB) / uB if uB else float('nan')

        hang.append(dict(service=s, cham_diem=s in FP.SCORED, cha_tier1=';'.join(P),
                         rho_heB=round(rho, 5), beta=M[s]['beta'], cores=C[s],
                         lech_W_pct=round(lech_W, 2), u_heB=round(uB, 4),
                         u_heA=round(uA, 4), lech_u_pct=round(lech_u, 3)))

    D = pd.DataFrame(hang)
    p = os.path.join(OUT_DIR, 'doi_chieu_hai_he.csv')
    D.to_csv(p, index=False)
    print(f'n = {len(tr)} hang (SS-TRAIN, tai <= {tm:g} req/s)\n')
    print(D.to_string(index=False))

    sc = D[D.cham_diem]
    print(f'\n{"":4s}{"muc do":28s} {"lech lon nhat":>14s}  menh de "<= 0,5%"')
    for ten, col, tap in (('workload W_s, ca 7 service', 'lech_W_pct', D),
                          ('workload W_s, 5 node SCORED', 'lech_W_pct', sc),
                          ('MUC SU DUNG u_s, 5 SCORED', 'lech_u_pct', sc)):
        v = tap[col].max()
        print(f'    {ten:28s} {v:13.2f}%  {"DUNG" if v <= 0.5 else "SAI"}')
    ng = sc.loc[sc.u_heB.idxmax(), 'service']
    print(f'\n  nut nghen ({ng}) lech u = '
          f'{sc.loc[sc.u_heB.idxmax(), "lech_u_pct"]:.2f}% -- phan quyet KHONG doi')
    print(f'\n  => Phat bieu dung: lech <= 0,5% TREN MUC SU DUNG u (khong phai tren moi service).')
    print(f'  => {p}')


if __name__ == '__main__':
    main()
