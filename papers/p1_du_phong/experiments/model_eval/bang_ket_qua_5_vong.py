# -*- coding: utf-8 -*-
"""BANG KET QUA DAY DU — 5 vong / 27 o, ca HAI truc danh gia.

VI SAO SCRIPT NAY TON TAI
=========================
`papers/p1_du_phong/docs/paper_draft.tex` bao cao pham vi CU: 8 tinh nang / 15 o / 4 vong.
Hien vat dong bang co 18 tinh nang / 27 o / 5 vong -- vong thu nam (`TIEN CUU 2`,
12 o, lon nhat) chua vao bai. Script nay dung bang theo pham vi THAT de so sanh.

Moi con so cua ban thao da duoc doi chieu va DUNG cho vong cua no:
  9,6% -> 2,8%      = tap `khoa` (4 o)              [P1 -> P2]
  14%, 0 nguy hiem  = tap `TIEN CUU browse` (2 o)   [P3]
  +297%             = `quickadd` x2 trong `DOC LAP`
Ban thao khong SAI, chi CU ve pham vi.

HAI TRUC, HAI HIEN VAT -- khong gop duoc thanh mot bang
  p3_evaluation.csv    : sai so DIEM GAY, 27 o, model P1/P2/P3/P3+chain
  verdict_decisions.csv: PHAN QUYET, 288 cap (o, muc tai), model P0/P1/P1_ctrl/P2/P3
`P1_ctrl` va `P0` CHI co o hien vat phan quyet; `P3+chain` CHI co o hien vat diem gay.

Chay:  python papers/p1_du_phong/experiments/model_eval/bang_ket_qua_5_vong.py
Ra:    data/processed/frozen/bang_ket_qua_5_vong.csv
"""
import itertools
import os

import pandas as pd
from scipy.stats import binomtest

_P = os.path.dirname(os.path.abspath(__file__))
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
FROZEN = os.path.join(_P, 'data', 'processed', 'frozen')

# vong -> (so thu tu, ten ngan, trang thai bang chung)
VONG = {
    'dev (kiem tra khong hoi quy)':          (1, 'dev',        'phat trien'),
    'khoa (k~1 do duoc, ky vong khong doi)': (2, 'khoa',       'held-out KHOA'),
    'DOC LAP (chinh, du lieu SACH)':         (3, 'doc lap',    'DOC LAP (agent viet)'),
    'TIEN CUU (browse, du lieu SACH)':       (4, 'browse',     'TIEN CUU'),
    'TIEN CUU 2':                            (5, 'tien cuu 2', 'TIEN CUU'),
}
MODEL = ['P1 (k=1)', 'P2 (c,x; k=1)', 'P3 (k do duoc)', 'P3+chain']
BO_PQ = ['P0', 'P1', 'P1_ctrl', 'P2', 'P3']


def truc_diem_gay():
    """|sai so| diem gay + so o 'kha thi gia' (nguy hiem), theo vong x model."""
    E = pd.read_csv(os.path.join(FROZEN, 'p3_evaluation.csv'))
    E['e'] = E['sai_so_%'].abs()
    E['nguy'] = E.phan_loai.str.contains('gia', na=False)
    E['vong'] = E.set.map(lambda s: VONG[s][0])
    E['ten'] = E.set.map(lambda s: VONG[s][1])
    E['tt'] = E.set.map(lambda s: VONG[s][2])

    hang = []
    nhom = [(f'vong {v}', E[E.vong == v]) for v in sorted(E.vong.unique())]
    nhom += [('GOP 27 o', E), ('GOP TIEN CUU (4+5)', E[E.vong.isin([4, 5])]),
             ('GOP ngoai phat trien (2-5)', E[E.vong >= 2])]
    for ten, g in nhom:
        r = {'nhom': ten, 'tap': g.ten.iloc[0] if g.vong.nunique() == 1 else '-',
             'trang_thai': g.tt.iloc[0] if g.vong.nunique() == 1 else '-',
             'n_o': g.cell.nunique()}
        for m in MODEL:
            h = g[g.model == m]
            r[f'{m} |sai so|%'] = round(h.e.mean(), 2)
            r[f'{m} nguy hiem'] = int(h.nguy.sum())
        hang.append(r)
    return pd.DataFrame(hang)


def truc_phan_quyet():
    """Ti le dung + ti le NGUY HIEM + McNemar ghep cap tren 288 quyet dinh."""
    D = pd.read_csv(os.path.join(FROZEN, 'verdict_decisions.csv'))
    D['dung'] = D.verdict.str.upper().ne('INFEASIBLE') == D.that_kha_thi
    # moi bo du doan dong gop DUNG MOT quyet dinh cho moi (o, muc tai) -- dieu kien ghep cap
    D = D.sort_values('file').drop_duplicates(['predictor', 'cell', 'muc_tai'], keep='last')
    K = ['cell', 'muc_tai']
    DUNG = D.pivot_table(index=K, columns='predictor', values='dung', aggfunc='first').astype(float)
    PQ = D.pivot_table(index=K, columns='predictor', values='verdict', aggfunc='first')

    tom = []
    for p in BO_PQ:
        s = D[D.predictor == p]
        n = int(DUNG[p].notna().sum())
        ng = int(((s.verdict.str.upper() != 'INFEASIBLE') & (~s.that_kha_thi)).sum())
        tom.append(dict(bo=p, n=n, dung_pct=round(100 * DUNG[p].mean(), 2),
                        nguy_hiem=ng, nguy_pct=round(100 * ng / n, 2)))

    cap = []
    for a, b in itertools.combinations(BO_PQ, 2):
        m = DUNG[a].notna() & DUNG[b].notna()
        if not m.any():
            continue
        ga, gb = DUNG.loc[m, a].astype(bool), DUNG.loc[m, b].astype(bool)
        h, k = int((ga & ~gb).sum()), int((~ga & gb).sum())
        cap.append(dict(so_sanh=f'{a} vs {b}', n=int(m.sum()),
                        giong_het=int((PQ.loc[m, a] == PQ.loc[m, b]).sum()),
                        thang=f'{a}+{h}/{b}+{k}', cap_bat_dong=h + k,
                        p=round(binomtest(h, h + k, 0.5).pvalue, 4) if h + k else None))
    return pd.DataFrame(tom), pd.DataFrame(cap)


def main():
    A = truc_diem_gay()
    B, C = truc_phan_quyet()
    pd.set_option('display.width', 200)
    print('=' * 100)
    print('  TRUC 1 — sai so DIEM GAY (%), 27 o   [so nguy hiem = o "kha thi gia"]')
    print('=' * 100)
    print(A.to_string(index=False))
    print('\n' + '=' * 100)
    print('  TRUC 2 — PHAN QUYET, 288 cap (o, muc tai)')
    print('=' * 100)
    print(B.to_string(index=False))
    print('\n  ghep cap McNemar:')
    print(C.to_string(index=False))
    p = os.path.join(FROZEN, 'bang_ket_qua_5_vong.csv')
    A.to_csv(p, index=False)
    print(f'\n  truc 1 -> {p}')


if __name__ == '__main__':
    main()
