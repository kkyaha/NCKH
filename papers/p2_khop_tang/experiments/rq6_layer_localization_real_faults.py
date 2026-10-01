# -*- coding: utf-8 -*-
"""RQ6 — DINH VI TREN TANG NAO? Kiem bang NHAN TIEM LOI THAT (RE2-SS).

Ket qua truoc do ("tang workload dinh vi tot hon tang CPU") duoc cham diem bang
tap reachable SUY TU DO THI GOI -- co vong tron cau truc. Bai nay dung dap an
KHONG suy tu do thi nao ca: RCAEval RE2-SS tiem mot loi THAT vao mot service cu
the (ten thu muc = `{service}_{loai loi}`), 5 service x 6 loai x 3 lan lap = 90 run,
kem `inject_time.txt`.

Cau hoi sac hon: khi nguyen nhan nam o tang nao thi dinh vi duoc tren tang nao?
  - loi TAI NGUYEN (cpu/mem/disk/socket) dat TRUC TIEP len tang CPU
  - loi MANG (delay/loss) tac dong len duong goi
Du doan tu co che: dinh vi duoc tren TANG MA NGUYEN NHAN NAM. Neu dung, thi that
bai trong thiet lap goc khong phai loi cua Shapley ma la do TIEM tren tang workload
roi QUY GAN tren tang CPU.

Chi so: service bi tiem co duoc xep hang top-1 / top-3 khong, tren tung tang.
Muc ngau nhien: 1/7 = 14.3% (top-1), 3/7 = 42.9% (top-3).

Chay:  python papers/p2_khop_tang/experiments/rq6_layer_localization_real_faults.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RAW = os.path.join(_P, 'data', 'raw', 'RE2-SS')
RES = os.path.join(_P, 'data', 'processed', 'scm_results')
SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']


def shifts(d, inj_t):
    """sigma-shift = (trung binh SAU - trung binh TRUOC) / std TRUOC, cho tung tang."""
    tc = 'imte' if 'imte' in d.columns else 'time'
    before, after = d[d[tc] < inj_t], d[d[tc] >= inj_t]
    if len(before) < 20 or len(after) < 20:
        return None
    out = {}
    for s in SERVICES:
        for kind in ('workload', 'cpu'):
            c = f'{s}_{kind}'
            if c not in d.columns:
                out[(s, kind)] = np.nan; continue
            b, a = before[c].dropna(), after[c].dropna()
            sd = b.std()
            out[(s, kind)] = abs(a.mean() - b.mean()) / sd if (len(b) > 10 and len(a) > 10 and sd > 0) else np.nan
    return out


def main():
    rows = []
    for scen in sorted(os.listdir(RAW)):
        sp = os.path.join(RAW, scen)
        if not os.path.isdir(sp):
            continue
        parts = scen.rsplit('_', 1)
        if len(parts) != 2:
            continue
        inj_svc, ftype = parts
        for run in sorted(os.listdir(sp)):
            mp, ip = os.path.join(sp, run, 'simple_metrics.csv'), os.path.join(sp, run, 'inject_time.txt')
            if not (os.path.exists(mp) and os.path.exists(ip)):
                continue
            try:
                t = int(open(ip).read().strip())
                sh = shifts(pd.read_csv(mp), t)
            except Exception:
                continue
            if sh is None:
                continue
            for kind in ('workload', 'cpu'):
                v = {s: sh[(s, kind)] for s in SERVICES if np.isfinite(sh.get((s, kind), np.nan))}
                if len(v) < 4 or inj_svc not in v:
                    continue
                rank = sorted(v, key=v.get, reverse=True)
                rows.append(dict(scenario=scen, run=run, injected=inj_svc, fault=ftype, layer=kind,
                                 n_cand=len(v), rank=rank.index(inj_svc) + 1,
                                 top1=rank[0] == inj_svc, top3=inj_svc in rank[:3],
                                 shift_injected=v[inj_svc], shift_max=max(v.values())))
    D = pd.DataFrame(rows)
    out = os.path.join(RES, 'rq6_layer_localization_real_faults.csv')
    D.to_csv(out, index=False)
    print(f'{D.scenario.nunique()} kich ban, {len(D)//2} run x 2 tang | da ghi {out}')
    print(f'muc ngau nhien: top-1 = {100/7:.1f}%, top-3 = {300/7:.1f}%\n')

    print('=== TONG: dinh vi service bi tiem, theo tang ===')
    g = D.groupby('layer').agg(n=('top1', 'size'), top1=('top1', 'mean'), top3=('top3', 'mean'),
                               hang_tb=('rank', 'mean'))
    g[['top1', 'top3']] *= 100
    print(g.round(1).to_string())

    print('\n=== theo LOAI LOI (top-1 %) ===')
    p = D.pivot_table(index='fault', columns='layer', values='top1', aggfunc='mean') * 100
    p['n_run'] = D[D.layer == 'cpu'].groupby('fault').size()
    p['tang thang'] = np.where(p.cpu > p.workload, 'CPU', np.where(p.workload > p.cpu, 'workload', '='))
    print(p.round(1).to_string())

    print('\n=== theo LOAI LOI (top-3 %) ===')
    p3 = D.pivot_table(index='fault', columns='layer', values='top3', aggfunc='mean') * 100
    print(p3.round(1).to_string())

    print('\n=== nhom loi TAI NGUYEN (cpu/mem/disk/socket) vs MANG (delay/loss) ===')
    D['nhom'] = np.where(D.fault.isin(['cpu', 'mem', 'disk', 'socket']), 'tai nguyen', 'mang')
    q = D.groupby(['nhom', 'layer']).agg(n=('top1', 'size'), top1=('top1', 'mean'), top3=('top3', 'mean'))
    q[['top1', 'top3']] *= 100
    print(q.round(1).to_string())


if __name__ == '__main__':
    main()
