# -*- coding: utf-8 -*-
"""DOI CHUNG CHAIN-SAI DUOI DANG PHAN PHOI, thay vi MOT lan boc duy nhat
===========================================================================
Ly do ton tai: `FP.wrong_chain(feature)` mac dinh seed=0, nen moi so lieu doi chung
bao cao truoc day (docs/DATA_FRAMEWORK.md muc 5d/5e) la ket qua cua DUNG MOT hien
thuc ngau nhien. Do manh cua bang chung doi theo seed -- Wilcoxon p trai tu 0.008
den 1.000 giua cac lan boc. Bao cao mot con so trong dai do, roi ket luan ve "chain
co gia tri hay khong", la ket luan ve hat giong chu khong phai ve chain.

Script nay chay lai MOI dai luong phu thuoc doi chung tren n_draws hien thuc doc lap
va bao cao PHAN PHOI, kem gia tri seed=0 de doi chieu.

KHONG dong den ban dong bang (predictions_frozen_RE2.json, SHA-256 f05c8eb5...bc8e2):
do la chung cu tien dang ky, tai sinh no se pha dau vet kiem toan. P1_ctrl trong file
do VAN la doi chung tien dang ky hop le -- phan phoi o day la lop ROBUSTNESS hau kiem,
bo sung chu khong thay the.

    python experiments/control_chain_distribution.py --draws 200
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
sys.path.insert(0, os.path.join(BASE, 'src', 'scm'))
import feasibility_predictor as FP  # noqa: E402
import evaluate_frozen as EF  # noqa: E402

FROZEN = os.path.join(BASE, 'data', 'processed', 'frozen')
SPLITS = [('dev', EF.DEV[1:], 'SS-LIMITS'), ('khoa', EF.LOCKED, 'SS-LIMITS'),
          ('doc_lap', EF.INDEP, 'SS-LIMITS-CLEAN'), ('tien_cuu', EF.PROSP, 'SS-LIMITS-CLEAN')]


def predictor():
    fz = json.load(open(os.path.join(FROZEN, 'predictions_frozen_RE2.json'), encoding='utf-8'))
    p2 = json.load(open(os.path.join(FROZEN, 'p2_params_dev.json'), encoding='utf-8'))['params']
    return FP.FeasibilityPredictor(fz['mechanism'], fz['params']['cores'], fz['params']['u_star'],
                                   feature_cost=p2), fz


def measured_utilisation(runs, P):
    """Muc su dung DO DUOC tung node o cac buoc TRUOC diem gay (khong can bao hoa)."""
    out = []
    for r in runs:
        lo = r['bp'].get('lo')
        if lo is None:
            continue
        d = r['df']
        d = d[(d['step_warm'] == 0) & (d['target_rps'] <= lo)]
        for L, g in d.groupby('target_rps'):
            out.append((r['feature'], r['scale'], float(L),
                        {s: float(g[f'{s}_cpu'].mean()) / P.cap[s] for s in FP.SCORED}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--out', default=os.path.join(BASE, 'data', 'processed', 'scm_results',
                                                  'control_chain_distribution.csv'))
    a = ap.parse_args()
    P, _ = predictor()

    rows = []
    for split, feats, root in SPLITS:
        runs = EF.load_ramps(os.path.join(BASE, 'data', 'raw', root), feats)
        obs = measured_utilisation(runs, P)
        if not obs:
            continue
        feats_seen = sorted({f for f, _, _, _ in obs})

        for draw in range(a.draws):
            wc = {f: FP.wrong_chain(f, draw) for f in feats_seen}
            cells = {}
            for feat, scale, L, um in obs:
                chain = FP.SOCKSHOP_CALL_CHAINS[FP.FEATURE_ARCHETYPE[feat]]['services']
                u1 = P.utilization(L, mode='P1', feature=feat, scale=scale)
                uc = P.utilization(L, mode='P1', feature=feat, scale=scale, chain=wc[feat])
                for s in FP.SCORED:
                    if s == FP.GATEWAY or s not in chain:
                        continue           # node TRONG chain la noi doi chung co the phan biet
                    cells.setdefault((feat, s), []).append(
                        (abs(u1[s] - um[s]) * 100, abs(uc[s] - um[s]) * 100))
            if not cells:
                continue
            arr = np.array([[np.mean([x[0] for x in v]), np.mean([x[1] for x in v])]
                            for v in cells.values()])
            diff = arr[:, 0] - arr[:, 1]

            # phan quyet tren luoi tai: acc + so "kha thi gia" (du doan FEASIBLE o tai da vo SLO)
            grid = {'P1': [0, 0, 0], 'ctrl': [0, 0, 0]}          # [dung, tong, kha_thi_gia]
            for r in runs:
                for L in range(40, 281, 20):
                    lab = EF.label_at(r['bp'], L)
                    if lab is None:
                        continue
                    for nm, ch in (('P1', None), ('ctrl', wc[r['feature']])):
                        v = P.verdict(L, mode='P1', feature=r['feature'], scale=r['scale'], chain=ch)
                        pred = 1 if v['verdict'] == 'INFEASIBLE' else 0
                        grid[nm][1] += 1
                        grid[nm][0] += int(pred == lab)
                        grid[nm][2] += int(lab == 1 and pred == 0)

            rows.append({'split': split, 'draw': draw, 'n_cell': len(arr),
                         'err_P1': float(arr[:, 0].mean()), 'err_ctrl': float(arr[:, 1].mean()),
                         'P1_thang': int((diff < -1e-9).sum()), 'ctrl_thang': int((diff > 1e-9).sum()),
                         'hoa': int((np.abs(diff) <= 1e-9).sum()),
                         'acc_P1': grid['P1'][0] / max(grid['P1'][1], 1),
                         'acc_ctrl': grid['ctrl'][0] / max(grid['ctrl'][1], 1),
                         'kha_thi_gia_P1': grid['P1'][2], 'kha_thi_gia_ctrl': grid['ctrl'][2],
                         'n_grid': grid['P1'][1]})
    R = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    R.to_csv(a.out, index=False)

    print('=' * 92)
    print(f'  DOI CHUNG CHAIN-SAI: {a.draws} hien thuc doc lap (mot chain sai moi tinh nang moi lan boc)')
    print('=' * 92)
    for split, _, _ in SPLITS:
        S = R[R.split == split]
        if S.empty:
            continue
        s0 = S[S.draw == 0].iloc[0]
        print(f'\n--- {split} ({int(s0.n_cell)} o tinh nang x node trong chain) ---')
        print(f'  sai so muc su dung (diem % cua tran, thap = tot):')
        print(f'    chain THAT : {S.err_P1.median():6.2f}   [khong doi theo lan boc]')
        print(f'    chain SAI  : trung vi {S.err_ctrl.median():6.2f}   '
              f'khoang 5-95% [{S.err_ctrl.quantile(.05):.2f}, {S.err_ctrl.quantile(.95):.2f}]   '
              f'seed=0: {s0.err_ctrl:.2f}')
        better = (S.err_P1 < S.err_ctrl).mean() * 100
        print(f'    ty le lan boc ma chain THAT tot hon (trung binh o): {better:.1f}%')
        print(f'  dem theo o: chain THAT thang trung vi {S.P1_thang.median():.0f}, '
              f'chain SAI thang {S.ctrl_thang.median():.0f}, HOA {S.hoa.median():.0f}'
              f'  ({S.hoa.median()/s0.n_cell*100:.0f}% so o KHONG phan biet duoc)')
        print(f'    seed=0 (gia tri da bao cao truoc day): P1 thang {int(s0.P1_thang)}, '
              f'ctrl thang {int(s0.ctrl_thang)}, hoa {int(s0.hoa)}')
        print(f'  phan quyet tren luoi tai ({int(s0.n_grid)} diem):')
        print(f'    acc  chain THAT {S.acc_P1.median():.3f} | chain SAI trung vi {S.acc_ctrl.median():.3f} '
              f'[{S.acc_ctrl.quantile(.05):.3f}, {S.acc_ctrl.quantile(.95):.3f}]  seed=0: {s0.acc_ctrl:.3f}')
        print(f'    kha thi gia  THAT {S.kha_thi_gia_P1.median():.0f} | SAI trung vi '
              f'{S.kha_thi_gia_ctrl.median():.0f} [{S.kha_thi_gia_ctrl.quantile(.05):.0f}, '
              f'{S.kha_thi_gia_ctrl.quantile(.95):.0f}]  seed=0: {int(s0.kha_thi_gia_ctrl)}')

    print(f'\n[OK] -> {a.out}')
    print('\nDOC KET QUA: doi chung nay YEU DO CAU TAO -- chain ngau nhien cung kich thuoc')
    print('luon chua gateway nen trung lap nhieu voi chain that. Ty le o HOA cao la bang')
    print('chung truc tiep cua dieu do, va la ly do khong nen doi no gach chan mot ket luan.')


if __name__ == '__main__':
    main()
