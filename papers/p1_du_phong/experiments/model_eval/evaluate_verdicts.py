# -*- coding: utf-8 -*-
"""DANH GIA MUC QUYET DINH: phan quyet kha thi TAI TUNG MUC TAI.

Vi sao doi muc tieu danh gia
----------------------------
Bo danh gia cu cham MOT con so moi o: R* (diem gay). Nhung:
  * chan ly la mot KHOANG [lo, hi], nen muc tieu diem dat sai bai toan -- MAPE cua
    cung mot du doan la 21,7% / 18,2% / 19,9% tuy chon bien duoi / giua / bien tren;
  * nguoi van hanh khong hoi "diem gay o dau" ma hoi "o tai L, con kha thi khong";
  * he DA TINH san phan quyet tung muc tai (truong `verdict` trong `grid`) -- 2364
    phan quyet nam trong cac file dong bang va bi bo di khi danh gia.

Doi muc tieu sang phan quyet dua n tu 27 du doan diem len ~159 quyet dinh SACH,
khong can do them mot o nao.

Chan ly cho tung (o, muc tai), LOAI dai nhap nhang
--------------------------------------------------
    L <  lo   -> chac chan KHA THI     (tai nay da xac minh chay duoc)
    L >= hi   -> chac chan KHONG kha thi
    lo <= L < hi -> NHAP NHANG, loai khoi danh gia
Loai dai nhap nhang lam bai toan DE HON, nen bat buoc phai bao do chinh xac PHAN
TANG theo khoang cach tu bien: toan bo sai sot cua mo hinh nam trong dai +-15%.

Hai cach doc `verdict`
----------------------
    long  : du KHA THI <=> verdict != INFEASIBLE   (MARGINAL tinh la kha thi)
    chat  : du KHA THI <=> verdict == FEASIBLE     (MARGINAL tinh la KHONG)
Bao ca hai: lua chon nay doi truc tiep ti le kha-thi-gia, nen khong duoc chon am tham.

Chay:  python papers/p1_du_phong/experiments/model_eval/evaluate_verdicts.py
Ra:    data/processed/frozen/verdict_decisions.csv  (1 dong / o / predictor / muc tai)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

_P = os.path.dirname(os.path.abspath(__file__))
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
FROZEN = os.path.join(_P, 'data', 'processed', 'frozen')


def cp95(k, n):
    """Khoang tin cay Clopper-Pearson 95% cho ti le k/n."""
    if n == 0:
        return (np.nan, np.nan)
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return 100 * lo, 100 * hi


def nap_quyet_dinh():
    """Doc moi grid trong cac file dong bang -> mot dong cho moi (o, predictor, muc tai)."""
    E = pd.read_csv(os.path.join(FROZEN, 'p3_evaluation.csv'))
    khoang = {r.cell: (float(r.do_duoc_lo), float(r.hi))
              for r in E.drop_duplicates('cell').itertuples()}
    rows = []
    for f in sorted(glob.glob(os.path.join(FROZEN, 'predictions_frozen_RE2*.json'))):
        d = json.load(open(f, encoding='utf-8'))
        stage = d.get('stage', '')
        for p in d.get('predictions', []):
            if p.get('feature') == 'base' or not p.get('grid'):
                continue
            cell = f"{p['feature']}x{int(p.get('scale', 1))}"
            if cell not in khoang:
                continue
            lo, hi = khoang[cell]
            giua = (lo + hi) / 2
            for L, g in p['grid'].items():
                L = float(L)
                v = g.get('verdict')
                if v is None:
                    continue
                if L < lo:
                    that = True
                elif L >= hi:
                    that = False
                else:
                    continue                      # dai nhap nhang -> loai
                rows.append(dict(
                    file=os.path.basename(f), stage=stage, predictor=p.get('predictor'),
                    cell=cell, feature=p['feature'], scale=p.get('scale'),
                    muc_tai=L, verdict=str(v), that_kha_thi=that,
                    kc_bien=abs(L - giua) / giua, lo=lo, hi=hi))
    D = pd.DataFrame(rows)
    # mot (o, predictor, muc tai) chi tinh MOT lan
    return D.drop_duplicates(['cell', 'predictor', 'muc_tai'])


def bang(D, cach):
    """In bang danh gia cho mot cach doc verdict ('long' hoac 'chat')."""
    du = (D.verdict.str.upper() != 'INFEASIBLE') if cach == 'long' \
        else (D.verdict.str.upper() == 'FEASIBLE')
    D = D.assign(du_kha_thi=du, dung=lambda x: x.du_kha_thi == x.that_kha_thi)
    print(f'\n{"="*104}\n  CACH DOC: {cach.upper()}  '
          f'({"MARGINAL = kha thi" if cach == "long" else "MARGINAL = KHONG kha thi"})\n{"="*104}')
    print(f'  {"predictor":10s} {"n":>5s} {"do chinh xac":>13s} {"KHA THI GIA":>13s} {"KTC 95%":>16s} '
          f'{"bo thieu":>10s}')
    for pr in ['P0', 'P1', 'P1_ctrl', 'P2', 'P3']:
        g = D[D.predictor == pr]
        if not len(g):
            continue
        kt = g[~g.that_kha_thi]
        fp = int((g.du_kha_thi & ~g.that_kha_thi).sum())
        fn = int((~g.du_kha_thi & g.that_kha_thi).sum())
        l, h = cp95(fp, len(kt))
        print(f'  {pr:10s} {len(g):5d} {100*g.dung.mean():12.1f}% {fp:4d}/{len(kt):<4d}={100*fp/max(len(kt),1):5.1f}% '
              f'[{l:5.1f}, {h:5.1f}]% {fn:6d}/{int(g.that_kha_thi.sum()):<4d}')
    print(f'\n  PHAN TANG theo khoang cach tu bien (predictor = P2, cach {cach}):')
    g0 = D[D.predictor == 'P2']
    for a, b in [(0, .15), (.15, .30), (.30, .50), (.50, 9)]:
        q = g0[(g0.kc_bien >= a) & (g0.kc_bien < b)]
        if not len(q):
            continue
        kt = q[~q.that_kha_thi]
        fp = int((q.du_kha_thi & ~q.that_kha_thi).sum())
        l, h = cp95(fp, len(kt))
        print(f'    kc {a:.2f}-{b:.2f}  n={len(q):4d}  dung {100*q.dung.mean():6.1f}%  '
              f'kha thi gia {fp}/{len(kt)} = {100*fp/max(len(kt),1):5.1f}%  [{l:.1f}, {h:.1f}]%')


def main():
    D = nap_quyet_dinh()
    out = os.path.join(FROZEN, 'verdict_decisions.csv')
    D.to_csv(out, index=False)
    print(f'{len(D)} quyet dinh SACH | {D.cell.nunique()} o | {D.predictor.nunique()} predictor'
          f' | da loai dai nhap nhang')
    print(f'  -> {out}')
    print(f'\n  doi chieu: bo danh gia cu dung {D.cell.nunique()} du doan DIEM')
    print(f'  verdict co trong du lieu: {sorted(D.verdict.unique())}')
    for cach in ('long', 'chat'):
        bang(D, cach)


if __name__ == '__main__':
    main()
