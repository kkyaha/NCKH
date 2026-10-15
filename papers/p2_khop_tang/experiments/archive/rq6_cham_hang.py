# -*- coding: utf-8 -*-
"""BO CHAM THEO HANG (rank scorer): dat duoc BAT BIEN THANG DO do CAU TAO.

Bai BARO phat bieu RobustScorer cua no la "distribution-free, scale-equivalent,
rotation-invariant". Nhung cai dat chia cho IQR:
    scaler = RobustScaler().fit(truoc);  diem = max(scaler.transform(sau))
Khi IQR = 0 thi thang do suy bien, cot do thuc chat KHONG duoc chuan hoa, va do
lech THO cua no (std trung vi ~8974) de moi cot chuan hoa dung. Do duoc: 86,9%
loi cua BARO la chon mot cot IQR=0.

Bo cham theo HANG khong co phep chia nao:
    U = thong ke Mann-Whitney giua cua so TRUOC va SAU
    diem = |U - EU| / sd(U)         (chuan hoa bang chinh so quan sat, khong phai thang do)
Tinh chat, do CAU TAO chu khong do du lieu:
  * bat bien voi MOI bien doi don dieu tang (hang khong doi) -> scale-equivalent THAT
  * khong phu thuoc phan phoi
  * khong the suy bien: khong chia cho uoc luong thang do nao
  * khong nghich dao ma tran tuong quan -> mien nhiem ca loi HANG
"""
import os
import sys
import gc
import importlib.util
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
_W = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rq6_wrapper_rcaeval.py')
_s = importlib.util.spec_from_file_location('w', _W)
w = importlib.util.module_from_spec(_s)
_a = sys.argv; sys.argv = ['x']; _s.loader.exec_module(w); sys.argv = _a


def cham_hang(d, inject_time=None, dataset=None, **kw):
    """Xep hang cot bang thong ke hang Mann-Whitney chuan hoa. Khong co phep chia thang do."""
    tr = d[d['time'] < inject_time]
    sa = d[d['time'] >= inject_time]
    diem = []
    for c in d.columns:
        if c == 'time':
            continue
        x = tr[c].dropna().to_numpy(dtype=float)
        y = sa[c].dropna().to_numpy(dtype=float)
        n, m = len(x), len(y)
        if n < 10 or m < 10:
            continue
        # U qua thu hang gop (xu ly dong hang bang hang trung binh)
        gop = np.concatenate([x, y])
        hang = pd.Series(gop).rank().to_numpy()
        R1 = hang[:n].sum()
        U = R1 - n * (n + 1) / 2.0
        EU = n * m / 2.0
        # phuong sai co hieu chinh dong hang
        _, dem = np.unique(gop, return_counts=True)
        N = n + m
        hc = (dem ** 3 - dem).sum()
        var = n * m / 12.0 * ((N + 1) - hc / (N * (N - 1))) if N > 1 else 0.0
        if var <= 0:
            continue
        diem.append((c, abs(U - EU) / np.sqrt(var)))
    diem.sort(key=lambda t: -t[1])
    return {'node_names': [c for c, _ in diem], 'ranks': [c for c, _ in diem]}


def main():
    import io, contextlib
    import RCAEval.e2e as e2e
    from RCAEval.e2e.baro import baro
    PP = {'cham_hang': cham_hang, 'baro': baro, 'nsigma': e2e.nsigma}

    def goi(fn, d, t):
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                out = fn(d.copy(), inject_time=t, dataset=None)
        except Exception:
            return None
        if 'failed' in err.getvalue():
            return None
        r = [c for c in (out.get('ranks') or []) if c != 'time']
        svc = []
        for c in r:
            s = w.col2svc(c)
            if s not in svc:
                svc.append(s)
        return svc

    def loc_iqr0(d, t):
        a = d[d['time'] < t]
        giu = ['time']
        for c in d.columns:
            if c == 'time':
                continue
            x = a[c].dropna().to_numpy(dtype=float)
            if len(x) < 5:
                continue
            q1, q3 = np.percentile(x, [25, 75])
            if q3 - q1 > 0:
                giu.append(c)
        return d[giu] if len(giu) > 3 else d

    rows = []
    for he in ('SockShop', 'OnlineBoutique', 'TrainTicket'):
        for inj, ft, run, draw, t in w.runs_of(he):
            d = w.lam_sach(draw); del draw
            dl = loc_iqr0(d, t)
            r = dict(he=he, injected=inj, fault=ft, run=run)
            for ten, fn in PP.items():
                for nhan, fr in (('', d), ('_loc', dl)):
                    xh = goi(fn, fr, t)
                    r[f'{ten}{nhan}_t1'] = (bool(xh) and xh[0] == inj) if xh is not None else np.nan
                    r[f'{ten}{nhan}_t3'] = (bool(xh) and inj in xh[:3]) if xh is not None else np.nan
            rows.append(r)
            del d, dl; gc.collect()
        print(f'  {he} xong ({len(rows)})', flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(w.RES, 'rq6_cham_hang.csv'), index=False)
    print(f'\n{len(D)} ca\n')
    print(f'{"bo cham":14s} {"AC@1 khong loc":>16s} {"AC@1 CO loc IQR0":>18s} {"chenh":>8s}   {"AC@3 khong loc":>15s}')
    for ten in PP:
        a = 100 * D[f'{ten}_t1'].mean(); b = 100 * D[f'{ten}_loc_t1'].mean()
        c3 = 100 * D[f'{ten}_t3'].mean()
        print(f'{ten:14s} {a:15.2f}% {b:17.2f}% {b-a:+8.2f}   {c3:14.2f}%')
    print()
    print('theo he, AC@1 khong loc:')
    for he, g in D.groupby('he'):
        print(f'  {he:15s} ' + '  '.join(f'{t}={100*g[f"{t}_t1"].mean():5.1f}%' for t in PP))


if __name__ == '__main__':
    main()
