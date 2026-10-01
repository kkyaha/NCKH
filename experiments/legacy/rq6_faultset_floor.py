# -*- coding: utf-8 -*-
"""RQ6 — CHAM LAI VOI MUC SAN NGHIEM HON (tra loi arXiv 2609.27069).

Bai audit arXiv 2609.27069 chi ra: RCAEval chi tiem loi vao DUNG 5 service moi he,
nhung phoi 12-70 service trong telemetry. Vi vay muc san "ngau nhien deu tren toan
bo ung vien" (1/7, 1/11, 1/28) la muc san QUA DE cho bat ky bo xep hang DA HOC
duoc prior cua benchmark. Muc san trung thuc cho nhung bo do la 1/5 = 20%.

Script nay cham lai moi ket qua dinh vi tren CA HAI muc san:

  (a) deu tren TOAN BO ung vien xuat hien trong telemetry  -- san dung cho mot bo
      xep hang KHONG HOC gi tu benchmark (truong hop phuong phap nay: no chi doc
      do lon dich chuyen cua metric, khong he thay nhan nao).
  (b) deu tren DUNG 5 service co bi tiem                   -- san dung cho moi bo
      xep hang co the da hoc prior do (moi phuong phap co huan luyen tren RE2).

Tap 5 service bi tiem khong duoc khai bao tay: no duoc suy ra tu TEN THU MUC cua
chinh bo du lieu, nen khong co cho nao de lot gia dinh vao.

Chay:  python experiments/legacy/rq6_faultset_floor.py
Ra:    data/processed/scm_results/rq6_faultset_floor.csv        (1 dong / run / tang)
       data/processed/scm_results/rq6_faultset_floor_summary.csv (1 dong / he x nhom x tang)
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RES = os.path.join(_P, 'data', 'processed', 'scm_results')
LAYERS = ['workload', 'cpu', 'latency-50']
RESOURCE = ['cpu', 'mem', 'disk', 'socket']


def shifts(d, t, layer):
    """|dich chuyen trung binh| / std truoc, cho moi service tren MOT tang."""
    tc = 'imte' if 'imte' in d.columns else 'time'   # RE2-OB/TT co cot go sai chinh ta
    b, a = d[d[tc] < t], d[d[tc] >= t]
    if len(b) < 20 or len(a) < 20:
        return {}
    out = {}
    for c in d.columns:
        if not c.endswith('_' + layer):
            continue
        s = c[:-(len(layer) + 1)]
        nb, na = b[c].dropna(), a[c].dropna()
        if len(nb) < 10 or len(na) < 10 or nb.std() == 0:
            continue
        out[s] = abs(na.mean() - nb.mean()) / nb.std()
    return out


def runs():
    """(he, service bi tiem, loai loi, metrics, thoi diem tiem) -- dap an tu TEN THU MUC."""
    ss = os.path.join(_P, 'data', 'raw', 'RE2-SS')
    for scen in sorted(os.listdir(ss)):
        sp = os.path.join(ss, scen)
        if not os.path.isdir(sp):
            continue
        inj, _, ft = scen.rpartition('_')
        for r in sorted(os.listdir(sp)):
            mp, ip = os.path.join(sp, r, 'simple_metrics.csv'), os.path.join(sp, r, 'inject_time.txt')
            if os.path.exists(mp) and os.path.exists(ip):
                yield 'SockShop', inj, ft, pd.read_csv(mp), int(open(ip).read().strip())
    for pref, root, name in (('re2ob_', 'RE2-OB', 'OnlineBoutique'),
                             ('re2tt_', 'trainticket', 'TrainTicket')):
        for sp in sorted(glob.glob(os.path.join(_P, 'data', 'raw', root, pref + '*'))):
            base = os.path.basename(sp)[len(pref):].rsplit('_', 1)[0]
            inj, _, ft = base.rpartition('_')
            mp, ip = os.path.join(sp, 'metrics.parquet'), os.path.join(sp, 'inject_time.txt')
            if os.path.exists(mp) and os.path.exists(ip):
                yield name, inj, ft, pd.read_parquet(mp), int(open(ip).read().strip())


def rank_in(v, keep, inj):
    """Hang cua `inj` khi CHI xet ung vien trong `keep`."""
    vv = {k: x for k, x in v.items() if k in keep}
    if inj not in vv:
        return None, None
    return sorted(vv, key=vv.get, reverse=True).index(inj) + 1, len(vv)


def main():
    rows = []
    for sysname, inj, ft, d, t in runs():
        for layer in LAYERS:
            v = shifts(d, t, layer)
            if len(v) < 4 or inj not in v:
                continue
            rows.append(dict(system=sysname, injected=inj, fault=ft, layer=layer,
                             n_all=len(v), rank_all=sorted(v, key=v.get, reverse=True).index(inj) + 1,
                             r_inj=v[inj], r_max=max(v.values()),
                             top_svc=max(v, key=v.get), vmap=v))
    D = pd.DataFrame(rows)

    # TAP SERVICE BI TIEM -- suy tu ten thu muc, khong khai bao tay
    faultset = {s: set(g.injected.unique()) for s, g in D.groupby('system')}
    print('tap service BI TIEM (suy tu ten thu muc):')
    for s, f in sorted(faultset.items()):
        print(f'  {s:15s} {len(f)} service: {sorted(f)}')

    out = []
    for r in D.itertuples():
        k, n = rank_in(r.vmap, faultset[r.system], r.injected)
        out.append(dict(system=r.system, injected=r.injected, fault=r.fault, layer=r.layer,
                        n_all=r.n_all, rank_all=r.rank_all,
                        r_inj=r.r_inj, r_max=r.r_max, top_svc=r.top_svc,
                        t1_all=r.rank_all == 1, t3_all=r.rank_all <= 3,
                        n_fs=n, rank_fs=k, t1_fs=(k == 1), t3_fs=(k <= 3)))
    O = pd.DataFrame(out)
    O['nhom'] = np.where(O.fault.isin(RESOURCE), 'tai nguyen', 'mang')
    O.to_csv(os.path.join(RES, 'rq6_faultset_floor.csv'), index=False)

    srows = []
    for (s, nh, la), g in O.groupby(['system', 'nhom', 'layer']):
        na, nb = g.n_all.median(), g.n_fs.median()
        srows.append(dict(
            system=s, nhom=nh, layer=la, n=len(g),
            ac1_all=100 * g.t1_all.mean(), ac3_all=100 * g.t3_all.mean(),
            n_cand_all=na, floor_all=100 / na,
            p_all=binomtest(int(g.t1_all.sum()), len(g), 1 / na).pvalue,
            ac1_fs=100 * g.t1_fs.mean(), ac3_fs=100 * g.t3_fs.mean(),
            n_cand_fs=nb, floor_fs=100 / nb,
            p_fs=binomtest(int(g.t1_fs.sum()), len(g), 1 / nb).pvalue))
    S = pd.DataFrame(srows)
    S.to_csv(os.path.join(RES, 'rq6_faultset_floor_summary.csv'), index=False)

    print(f"\n{'he':15s} {'nhom':11s} {'tang':11s} | {'AC@1 all':>9s} {'san(a)':>7s} |"
          f" {'AC@1 5svc':>10s} {'san(b)':>7s} {'p (b)':>10s}")
    print('-' * 92)
    for r in S.sort_values(['system', 'nhom', 'layer']).itertuples():
        print(f'{r.system:15s} {r.nhom:11s} {r.layer:11s} | {r.ac1_all:8.1f}% {r.floor_all:6.1f}% |'
              f' {r.ac1_fs:9.1f}% {r.floor_fs:6.1f}% {r.p_fs:10.3g}')
    print(f'\n-> {RES}/rq6_faultset_floor.csv  ({len(O)} dong)')


if __name__ == '__main__':
    main()
