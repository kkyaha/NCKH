# -*- coding: utf-8 -*-
"""RQ6 — DO DIEU KIEN HANG cua ma tran tuong quan: co so cho luan diem KHA THI.

Kiem dinh Fisher-z can tuong quan rieng rho(X,Y|Z), ma tinh no phai NGHICH DAO
ma tran tuong quan cua {X,Y} hop Z. Neu cac cot phu thuoc tuyen tinh thi det R = 0,
R^-1 khong ton tai, va tuong quan rieng KHONG XAC DINH -- do la luc PC nem
`ValueError: Data correlation matrix is singular. Cannot run fisherz test.`

Trong du lieu nay co hai nguon phu thuoc THEO CAU TAO, khong phai trung hop:
  * `svc_latency-50` va `svc_latency-90` la hai phan vi cua CUNG mot phan phoi
    do tre; duoi tai chung di cung nhau, r -> 1.
  * neu workload la bien dieu khien va latency/cpu deu la ham don dieu cua no thi
    co phu thuoc gan tuyen tinh theo tung tam phan.
Tron L tang x S service cho p = L*S cot nhung chi ~S huong doc lap thuc cong L
nhan to chung -> HANG THAP HON p rat nhieu. Giu mot tang: p = S, mot ho metric.

Script do, cho MOI ca, o CA HAI che do (toan khung / mot tang):
    p            so cot
    hang         hang so cua R (numpy matrix_rank)
    thieu_hang   p - hang
    kappa        so dieu kien cua R
    n_tri_nho    so tri rieng < 1e-10
    cap_99       so cap cot |r| > 0,99
    cap_99_noi   trong so do, bao nhieu cap NAM TRONG CUNG service
Roi ghep voi viec circa/pc co fallback o che do (0) hay khong -> thieu hang co
DU DOAN duoc ca nao vo khong.

Ra: data/processed/scm_results/rq6_dieu_kien_hang.csv
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
_spec = importlib.util.spec_from_file_location('w', _W)
w = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = ['x']
_spec.loader.exec_module(w)
sys.argv = _argv
from RCAEval.io.time_series import preprocess


def do(d):
    """Do dieu kien hang cua ma tran tuong quan tren DUNG khung ma Fisher-z gap."""
    x = preprocess(data=d.copy(), dataset=None, dk_select_useful=False)
    x = x.drop(columns=[c for c in x.columns if c == 'time'], errors='ignore')
    x = x.loc[:, x.std(numeric_only=True) > 0]
    p = x.shape[1]
    if p < 2:
        return None
    R = np.corrcoef(x.to_numpy(dtype=float), rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    ev = np.linalg.eigvalsh(R)
    ev = np.clip(ev, 0.0, None)
    hang = int(np.linalg.matrix_rank(R))
    nho, lon = float(ev.min()), float(ev.max())
    A = np.abs(np.triu(R, 1))
    cap = int((A > 0.99).sum())
    svc = [w.col2svc(c) for c in x.columns]
    noi = 0
    if cap:
        ii, jj = np.where(A > 0.99)
        noi = int(sum(1 for i, j in zip(ii, jj) if svc[i] == svc[j]))
    return dict(p=p, hang=hang, thieu_hang=p - hang,
                kappa=(lon / nho) if nho > 1e-15 else np.inf,
                tri_min=nho, n_tri_nho=int((ev < 1e-10).sum()),
                cap_99=cap, cap_99_noi=noi)


def main():
    rows = []
    for he in ('SockShop', 'OnlineBoutique', 'TrainTicket'):
        for inj, ft, run, draw, t in w.runs_of(he):
            d = w.lam_sach(draw)
            del draw
            per = {la: v for la, v in ((la, w.dich_chuyen(d, t, la)) for la in w.LAYERS)
                   if len(v) >= 3}
            P = {la: max(v.values()) / sum(v.values()) for la, v in per.items() if sum(v.values()) > 0}
            if not P:
                del d
                gc.collect()
                continue
            la = max(P, key=P.get)
            cot = ['time'] + [c for c in d.columns if c.endswith('_' + la)]
            a, b = do(d), do(d[cot])
            if a and b:
                rows.append(dict(he=he, injected=inj, fault=ft, run=run, tang_chon=la,
                                 **{f'{k}_0': v for k, v in a.items()},
                                 **{f'{k}_1': v for k, v in b.items()}))
            del d
            gc.collect()
        print(f'  {he} xong ({len(rows)} ca)', flush=True)
    D = pd.DataFrame(rows)
    out = os.path.join(w.RES, 'rq6_dieu_kien_hang.csv')
    D.to_csv(out, index=False)
    print(f'\n{len(D)} ca -> {out}')


if __name__ == '__main__':
    main()
