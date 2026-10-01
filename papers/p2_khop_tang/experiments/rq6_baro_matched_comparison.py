# -*- coding: utf-8 -*-
"""RQ6 — SO KHOP VOI BARO: thang do la THONG KE hay la TANG?

Ket qua truoc: xep hang theo dich chuyen tang CPU dinh vi service bi tiem
98.3% top-1 tren loi tai nguyen, so voi BARO 26.7% da bao trong draft. NHUNG
hai con so do KHONG so duoc: BARO cham tren ~14 service voi MOI cot metric;
phep cua toi cham tren 7 service ung dung voi CHI cot _cpu.

Bai nay tach ba yeu to tren CUNG 90 run, CUNG dap an (service bi tiem, tu ten
thu muc), CUNG cua so truoc/sau inject_time:

  thong ke : BARO (RobustScaler median/IQR, max|z|, verbatim tu e2e/baro.py)
             MEANSHIFT (|trung binh sau - trung binh truoc| / std truoc)
  tang     : MOI cot metric   vs   CHI cot _cpu
  ung vien : 7 service ung dung  vs  MOI service co trong cot (~14)

Neu CHI-CPU thang MOI-COT o CA HAI thong ke -> yeu to quyet dinh la TANG.
Neu BARO thang MEANSHIFT tren cung tang -> yeu to quyet dinh la THONG KE.

Chay:  python papers/p2_khop_tang/experiments/rq6_baro_matched_comparison.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RAW = os.path.join(_P, 'data', 'raw', 'RE2-SS')
RES = os.path.join(_P, 'data', 'processed', 'scm_results')
APP7 = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
ENV = {'time', 'imte', 'vm_cpu_util', 'vm_mem_avail_mb'}


def rank_baro(normal, anomal, cols):
    """Luat cua RCAEval e2e/baro.py, dung nguyen van."""
    out = []
    for c in cols:
        n, a = normal[c].to_numpy(float), anomal[c].to_numpy(float)
        n, a = n[np.isfinite(n)], a[np.isfinite(a)]
        if len(n) < 5 or len(a) < 5 or np.all(n == n[0]):
            continue
        z = RobustScaler().fit(n.reshape(-1, 1)).transform(a.reshape(-1, 1))
        out.append((c, float(np.max(np.abs(z)))))
    return out


def rank_meanshift(normal, anomal, cols):
    """|trung binh sau - trung binh truoc| / std truoc."""
    out = []
    for c in cols:
        n, a = normal[c].dropna(), anomal[c].dropna()
        if len(n) < 5 or len(a) < 5 or n.std() == 0:
            continue
        out.append((c, abs(a.mean() - n.mean()) / n.std()))
    return out


STATS = {'BARO': rank_baro, 'MEANSHIFT': rank_meanshift}


def main():
    rows = []
    for scen in sorted(os.listdir(RAW)):
        sp = os.path.join(RAW, scen)
        if not os.path.isdir(sp):
            continue
        inj_svc, _, ftype = scen.rpartition('_')
        if not inj_svc:
            continue
        for run in sorted(os.listdir(sp)):
            mp, ip = os.path.join(sp, run, 'simple_metrics.csv'), os.path.join(sp, run, 'inject_time.txt')
            if not (os.path.exists(mp) and os.path.exists(ip)):
                continue
            try:
                t = int(open(ip).read().strip())
                d = pd.read_csv(mp)
            except Exception:
                continue
            tc = 'imte' if 'imte' in d.columns else 'time'
            normal, anomal = d[d[tc] < t], d[d[tc] >= t]
            if len(normal) < 20 or len(anomal) < 20:
                continue
            metric_cols = [c for c in d.columns if c not in ENV and '_' in c]
            for cand_lab, cand in (('app7', set(APP7)),
                                   ('all', {c.rsplit('_', 1)[0] for c in metric_cols})):
                if inj_svc not in cand:
                    continue
                for layer_lab, sel in (('moi_cot', [c for c in metric_cols if c.rsplit('_', 1)[0] in cand]),
                                       ('chi_cpu', [f'{s}_cpu' for s in cand if f'{s}_cpu' in d.columns])):
                    if not sel:
                        continue
                    for stat_lab, fn in STATS.items():
                        r = fn(normal, anomal, sel)
                        if not r:
                            continue
                        r.sort(key=lambda x: x[1], reverse=True)
                        svc_rank = []
                        for c, _ in r:
                            s = c.rsplit('_', 1)[0]
                            if s not in svc_rank:
                                svc_rank.append(s)
                        if inj_svc not in svc_rank:
                            continue
                        rows.append(dict(scenario=scen, run=run, injected=inj_svc, fault=ftype,
                                         cand=cand_lab, layer=layer_lab, stat=stat_lab,
                                         n_cand=len(svc_rank), rank=svc_rank.index(inj_svc) + 1,
                                         top1=svc_rank[0] == inj_svc, top3=inj_svc in svc_rank[:3]))
    D = pd.DataFrame(rows)
    out = os.path.join(RES, 'rq6_baro_matched_comparison.csv')
    D.to_csv(out, index=False)
    print(f'{D.scenario.nunique()} kich ban | {len(D)} (run x dieu kien) | da ghi {out}\n')

    print('=== LUOI 2x2x2: top-1 % (ngoac = so ung vien trung binh, ngau nhien = 100/n) ===')
    g = D.groupby(['cand', 'layer', 'stat']).agg(n=('top1', 'size'), top1=('top1', 'mean'),
                                                 top3=('top3', 'mean'), nc=('n_cand', 'mean'),
                                                 hang=('rank', 'mean'))
    g[['top1', 'top3']] *= 100
    g['ngau_nhien'] = 100.0 / g.nc
    print(g.round(1).to_string())

    print('\n=== chi loi TAI NGUYEN (cpu/mem/disk/socket) ===')
    R = D[D.fault.isin(['cpu', 'mem', 'disk', 'socket'])]
    g2 = R.groupby(['cand', 'layer', 'stat']).agg(n=('top1', 'size'), top1=('top1', 'mean'),
                                                  top3=('top3', 'mean'), nc=('n_cand', 'mean'))
    g2[['top1', 'top3']] *= 100
    g2['ngau_nhien'] = 100.0 / g2.nc
    print(g2.round(1).to_string())

    print('\n=== TACH YEU TO (chi loi tai nguyen, ung vien app7) ===')
    A = R[R.cand == 'app7']
    for stat in STATS:
        s = A[A.stat == stat]
        a = 100 * s[s.layer == 'chi_cpu'].top1.mean()
        b = 100 * s[s.layer == 'moi_cot'].top1.mean()
        print(f'  thong ke {stat:10s}: chi_cpu {a:5.1f}%  vs  moi_cot {b:5.1f}%  -> tang dong gop {a-b:+5.1f} d')
    for layer in ('chi_cpu', 'moi_cot'):
        s = A[A.layer == layer]
        a = 100 * s[s.stat == 'MEANSHIFT'].top1.mean()
        b = 100 * s[s.stat == 'BARO'].top1.mean()
        print(f'  tang {layer:8s}    : MEANSHIFT {a:5.1f}%  vs  BARO {b:5.1f}%  -> thong ke dong gop {a-b:+5.1f} d')


if __name__ == '__main__':
    main()
