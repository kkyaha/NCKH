# -*- coding: utf-8 -*-
"""RQ6 — BO CHI SO CHUAN CUA RCAEval: AC@1, AC@3, AC@5, Avg@5, MRR.

Vi sao can script nay. Bai dang bao top-1/top-3, tuong duong AC@1/AC@3 cua
RCAEval. Nhung bang chuan cua benchmark (arXiv 2412.17015) va cua moi baseline
trong do bao NAM chi so: AC@1, AC@3, AC@5, Avg@5, MRR. Khong co ba chi so sau
thi khong dat canh bang cua ho duoc, va reviewer se hoi ngay.

Cong thuc, nguyen van theo RCAEval:
  AC@k   = ti le ca co HANG cua service bi tiem <= k
  Avg@5  = trung binh cua AC@1..AC@5   (mot con so gop, bot nhay cam voi k)
  MRR    = trung binh cua 1/hang       (thuong hang cao, phat hang thap, lien tuc)

Khong chay lai thuc nghiem nao: moi CSV ket qua da luu HANG cua service bi tiem,
nen day chi la doi cach TONG HOP. Do la ly do viec nay re.

Ba bang:
  (T1) ba tang x ba he x hai nhom loi   -- tu rq6_faultset_floor.csv (CA HAI muc san)
  (T2) luoi khop 2x2x2 voi BARO         -- tu rq6_baro_matched_comparison.csv
  (T3) ba tang, muc san deu             -- tu rq6_three_layer_localization.csv (doi chieu)

Chay:  python experiments/legacy/rq6_rcaeval_metrics.py
Ra:    data/processed/scm_results/rq6_rcaeval_metrics_layers.csv
       data/processed/scm_results/rq6_rcaeval_metrics_baro_grid.csv
"""
import os

import numpy as np
import pandas as pd

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RES = os.path.join(_P, 'data', 'processed', 'scm_results')
RESOURCE = ['cpu', 'mem', 'disk', 'socket']
KS = (1, 3, 5)


def metrics(ranks: pd.Series) -> dict:
    """Bo chi so chuan RCAEval tu mot chuoi HANG (1 = dung dau)."""
    r = pd.to_numeric(ranks, errors='coerce').dropna()
    if len(r) == 0:
        return {}
    out = {'n': len(r)}
    for k in KS:
        out[f'AC@{k}'] = 100 * (r <= k).mean()
    # Avg@5 theo dinh nghia cua RCAEval = trung binh AC@1..AC@5 (ca k=2,4)
    out['Avg@5'] = 100 * np.mean([(r <= k).mean() for k in range(1, 6)])
    out['MRR'] = (1.0 / r).mean()
    out['hang trung vi'] = r.median()
    return out


def floor_metrics(n_cand: pd.Series) -> dict:
    """Muc san cua CHINH bo chi so do, khi xep hang ngau nhien deu tren k ung vien.

    AC@j = min(j,k)/k;  MRR = (1/k) * sum_{i=1..k} 1/i  (ky vong cua 1/hang).
    Phai bao kem, vi Avg@5 va MRR co muc san CAO hon AC@1 rat nhieu -- bo chi so
    nao cung vo nghia neu khong doi chieu muc san cua chinh no.
    """
    k = float(pd.to_numeric(n_cand, errors='coerce').dropna().median())
    out = {'ung vien k': k}
    for j in KS:
        out[f'san AC@{j}'] = 100 * min(j, k) / k
    out['san Avg@5'] = 100 * np.mean([min(j, k) / k for j in range(1, 6)])
    out['san MRR'] = sum(1.0 / i for i in range(1, int(k) + 1)) / k
    return out


def show(D: pd.DataFrame, tieu_de: str, cols_front: list):
    print('\n' + '=' * 120)
    print('  ' + tieu_de)
    print('=' * 120)
    with pd.option_context('display.width', 240, 'display.max_columns', 40,
                           'display.float_format', lambda v: f'{v:,.3f}'):
        print(D[cols_front + [c for c in D.columns if c not in cols_front]]
              .to_string(index=False))


def main():
    # ---------------- (T1) ba tang, CA HAI muc san ----------------
    F = pd.read_csv(os.path.join(RES, 'rq6_faultset_floor.csv'))
    F['nhom'] = np.where(F.fault.isin(RESOURCE), 'tai nguyen', 'mang')
    rows = []
    for (sys_, nhom, la), g in F.groupby(['system', 'nhom', 'layer']):
        for ten, rcol, ncol in [('(a) moi ung vien', 'rank_all', 'n_all'),
                                ('(b) chi 5 svc bi tiem', 'rank_fs', 'n_fs')]:
            m = metrics(g[rcol])
            if not m:
                continue
            rows.append(dict(he=sys_, nhom_loi=nhom, tang=la, tap_ung_vien=ten,
                             **m, **floor_metrics(g[ncol])))
    T1 = pd.DataFrame(rows)
    T1.to_csv(os.path.join(RES, 'rq6_rcaeval_metrics_layers.csv'), index=False)

    for nhom in ['tai nguyen', 'mang']:
        for tap in ['(a) moi ung vien', '(b) chi 5 svc bi tiem']:
            sub = T1[(T1.nhom_loi == nhom) & (T1.tap_ung_vien == tap)].sort_values(['he', 'tang'])
            show(sub.drop(columns=['nhom_loi', 'tap_ung_vien']),
                 f'(T1) LOI {nhom.upper()}  —  tap ung vien {tap}',
                 ['he', 'tang', 'n'])

    # ---------------- (T2) luoi khop voi BARO ----------------
    B = pd.read_csv(os.path.join(RES, 'rq6_baro_matched_comparison.csv'))
    B = B[B.fault.isin(RESOURCE)]
    rows = []
    for (cv, la, st), g in B.groupby(['cand', 'layer', 'stat']):
        rows.append(dict(ung_vien={'app7': '7 service app', 'all': 'moi service (~15)'}[cv],
                         tang={'chi_cpu': 'CHI cot _cpu', 'moi_cot': 'moi cot metric'}[la],
                         thong_ke=st, **metrics(g['rank']), **floor_metrics(g.n_cand)))
    T2 = pd.DataFrame(rows).sort_values('AC@1', ascending=False)
    T2.to_csv(os.path.join(RES, 'rq6_rcaeval_metrics_baro_grid.csv'), index=False)
    show(T2, '(T2) LUOI KHOP 2x2x2 voi BARO — loi tai nguyen, 90 run x 8 dieu kien',
         ['ung_vien', 'tang', 'thong_ke', 'n'])

    # chenh lech theo TUNG chi so, khong chi AC@1
    print('\n--- CHENH LECH DO LECH TANG, theo TUNG chi so (BARO, moi service) ---')
    a = T2[(T2.ung_vien == 'moi service (~15)') & (T2.tang == 'CHI cot _cpu') & (T2.thong_ke == 'BARO')].iloc[0]
    b = T2[(T2.ung_vien == 'moi service (~15)') & (T2.tang == 'moi cot metric') & (T2.thong_ke == 'BARO')].iloc[0]
    for c in ['AC@1', 'AC@3', 'AC@5', 'Avg@5']:
        print(f'  {c:6s}  chi _cpu {a[c]:6.1f}%  vs  moi cot {b[c]:6.1f}%   -> chenh {a[c] - b[c]:+6.1f} diem')
    print(f'  {"MRR":6s}  chi _cpu {a["MRR"]:6.3f}   vs  moi cot {b["MRR"]:6.3f}    -> chenh {a["MRR"] - b["MRR"]:+6.3f}')

    print('\n--- TANG vs THONG KE, do bang MRR (lien tuc, khong phu thuoc nguong k) ---')
    for cv in T2.ung_vien.unique():
        for st in ['BARO', 'MEANSHIFT']:
            x = T2[(T2.ung_vien == cv) & (T2.thong_ke == st)]
            d = (x[x.tang == 'CHI cot _cpu'].MRR.iloc[0] - x[x.tang == 'moi cot metric'].MRR.iloc[0])
            print(f'  TANG   dong gop ({cv:18s}, {st:9s}): dMRR = {d:+.3f}')
    for la in T2.tang.unique():
        for cv in T2.ung_vien.unique():
            x = T2[(T2.ung_vien == cv) & (T2.tang == la)]
            d = (x[x.thong_ke == 'MEANSHIFT'].MRR.iloc[0] - x[x.thong_ke == 'BARO'].MRR.iloc[0])
            print(f'  THONGKE dong gop ({cv:18s}, {la:14s}): dMRR = {d:+.3f}')

    # ---------------- (T3) doi chieu voi CSV ba tang goc ----------------
    D = pd.read_csv(os.path.join(RES, 'rq6_three_layer_localization.csv'))
    D['nhom'] = np.where(D.fault.isin(RESOURCE), 'tai nguyen', 'mang')
    chk = (D.groupby(['system', 'nhom', 'layer'])
           .apply(lambda g: pd.Series(metrics(g['rank'])), include_groups=False).reset_index())
    m = T1[T1.tap_ung_vien == '(a) moi ung vien'].merge(
        chk, left_on=['he', 'nhom_loi', 'tang'], right_on=['system', 'nhom', 'layer'],
        suffixes=('_floor', '_3layer'))
    sai = (m['AC@1_floor'] - m['AC@1_3layer']).abs().max()
    print(f'\n[kiem doi chieu] lech AC@1 lon nhat giua hai duong tinh doc lap: {sai:.3f} diem')
    print('  (rq6_faultset_floor.py va rq6_three_layer_and_scope.py doc du lieu tho doc lap nhau)')

    print(f'\n-> {RES}/rq6_rcaeval_metrics_layers.csv')
    print(f'-> {RES}/rq6_rcaeval_metrics_baro_grid.csv')


if __name__ == '__main__':
    main()
