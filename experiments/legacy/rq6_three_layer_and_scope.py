# -*- coding: utf-8 -*-
"""RQ6 — BA TANG, va KIEM DIEU KIEN PHAM VI tren HAI he.

Ba viec trong mot script:

  (1) DINH VI TREN BA TANG. Truoc day chi thu tang `workload` va `cpu`. Nhung bang
      mau canh (scm_graph_builder.py:70-79) co ca `latency-50 -> latency-50`, tuc mo
      hinh CO tang do tre. Tien doan cua ly thuyet: loi `delay`/`loss` phai dinh vi
      duoc tren TANG DO TRE. Chua ai kiem.

  (2) KIEM NHI THUC cho tien doan diem. Ly thuyet noi tang lech phai dat DUNG muc
      ngau nhien, khong phai "kem hon". So hai con so bang mat la khong du -- can p-value
      cho H0: ti le = 1/k.

  (3) DIEU KIEN PHAM VI tren HAI he. Tien de thuc nghiem (loi tai nguyen khong lam
      dich workload) hien chi do tren Sock Shop. Lap lai tren Train Ticket.

Dap an: service bi tiem, lay tu TEN THU MUC. Khong suy tu do thi nao.
Chay:  python experiments/legacy/rq6_three_layer_and_scope.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RES = os.path.join(_P, 'data', 'processed', 'scm_results')
LAYERS = ['workload', 'cpu', 'latency-50']
RESOURCE = ['cpu', 'mem', 'disk', 'socket']
NETWORK = ['delay', 'loss']


def load_sockshop():
    for scen in sorted(os.listdir(os.path.join(_P, 'data', 'raw', 'RE2-SS'))):
        sp = os.path.join(_P, 'data', 'raw', 'RE2-SS', scen)
        if not os.path.isdir(sp):
            continue
        svc, _, ft = scen.rpartition('_')
        for run in sorted(os.listdir(sp)):
            mp, ip = os.path.join(sp, run, 'simple_metrics.csv'), os.path.join(sp, run, 'inject_time.txt')
            if os.path.exists(mp) and os.path.exists(ip):
                try:
                    yield 'SockShop', svc, ft, run, pd.read_csv(mp), int(open(ip).read().strip())
                except Exception:
                    continue


def load_trainticket():
    for scen in sorted(os.listdir(os.path.join(_P, 'data', 'raw', 'trainticket'))):
        sp = os.path.join(_P, 'data', 'raw', 'trainticket', scen)
        if not os.path.isdir(sp):
            continue
        base = scen[len('re2tt_'):] if scen.startswith('re2tt_') else scen
        base = base.rsplit('_', 1)[0]            # bo so lan lap
        svc, _, ft = base.rpartition('_')
        mp, ip = os.path.join(sp, 'metrics.parquet'), os.path.join(sp, 'inject_time.txt')
        if os.path.exists(mp) and os.path.exists(ip):
            try:
                yield 'TrainTicket', svc, ft, scen[-1], pd.read_parquet(mp), int(open(ip).read().strip())
            except Exception:
                continue


def sigma_shifts(d, t, layer):
    """|trung binh sau - trung binh truoc| / std truoc, cho tung service tren MOT tang."""
    tc = 'imte' if 'imte' in d.columns else 'time'
    b, a = d[d[tc] < t], d[d[tc] >= t]
    if len(b) < 20 or len(a) < 20:
        return {}
    out = {}
    for c in d.columns:
        if not c.endswith('_' + layer):
            continue
        svc = c[: -(len(layer) + 1)]
        nb, na = b[c].dropna(), a[c].dropna()
        if len(nb) < 10 or len(na) < 10 or nb.std() == 0:
            continue
        out[svc] = abs(na.mean() - nb.mean()) / nb.std()
    return out


def load_onlineboutique():
    """RE2-OB: cung dinh dang parquet nhu Train Ticket, ten thu muc `re2ob_<svc>_<fault>_<n>`."""
    root = os.path.join(_P, 'data', 'raw', 'RE2-OB')
    if not os.path.isdir(root):
        return
    for scen in sorted(os.listdir(root)):
        sp = os.path.join(root, scen)
        if not os.path.isdir(sp):
            continue
        base = scen[len('re2ob_'):] if scen.startswith('re2ob_') else scen
        base = base.rsplit('_', 1)[0]
        svc, _, ft = base.rpartition('_')
        mp, ip = os.path.join(sp, 'metrics.parquet'), os.path.join(sp, 'inject_time.txt')
        if os.path.exists(mp) and os.path.exists(ip):
            try:
                yield 'OnlineBoutique', svc, ft, scen[-1], pd.read_parquet(mp), int(open(ip).read().strip())
            except Exception:
                continue


def main():
    rows, scope = [], []
    for sysname, svc, ft, run, d, t in (list(load_sockshop()) + list(load_onlineboutique())
                                        + list(load_trainticket())):
        # (3) dieu kien pham vi: DAU cua dich chuyen workload o service bi tiem
        tc = 'imte' if 'imte' in d.columns else 'time'
        b, a = d[d[tc] < t], d[d[tc] >= t]
        wc = f'{svc}_workload'
        if wc in d.columns and len(b) >= 20 and len(a) >= 20:
            nb, na = b[wc].dropna(), a[wc].dropna()
            if len(nb) >= 10 and len(na) >= 10 and nb.std() > 0:
                scope.append(dict(system=sysname, svc=svc, fault=ft,
                                  w_sigma=(na.mean() - nb.mean()) / nb.std()))
        # (1) dinh vi tren ba tang
        for layer in LAYERS:
            v = sigma_shifts(d, t, layer)
            if len(v) < 4 or svc not in v:
                continue
            rank = sorted(v, key=v.get, reverse=True)
            rows.append(dict(system=sysname, injected=svc, fault=ft, run=run, layer=layer,
                             n_cand=len(v), rank=rank.index(svc) + 1,
                             top1=rank[0] == svc, top3=svc in rank[:3]))
    D, S = pd.DataFrame(rows), pd.DataFrame(scope)
    D.to_csv(os.path.join(RES, 'rq6_three_layer_localization.csv'), index=False)
    S.to_csv(os.path.join(RES, 'rq6_scope_condition_two_systems.csv'), index=False)
    print(f'{len(D)} (run x tang) | {S.system.nunique()} he | da ghi 2 CSV\n')

    # ---------------- (1) ba tang ----------------
    print('=' * 74)
    print('  (1) DINH VI TREN BA TANG — top-1 %, theo nhom loi')
    print('=' * 74)
    D['nhom'] = np.where(D.fault.isin(RESOURCE), 'tai nguyen', 'mang')
    for sysname, g in D.groupby('system'):
        print(f'\n--- {sysname} (ngau nhien = 100/{g.n_cand.median():.0f} = {100/g.n_cand.median():.1f}%) ---')
        p = g.pivot_table(index='nhom', columns='layer', values='top1', aggfunc='mean') * 100
        p3 = g.pivot_table(index='nhom', columns='layer', values='top3', aggfunc='mean') * 100
        print('top-1:'); print(p.round(1).to_string())
        print('top-3:'); print(p3.round(1).to_string())
        print('\ntheo tung loai loi (top-1):')
        print((g.pivot_table(index='fault', columns='layer', values='top1', aggfunc='mean') * 100).round(1).to_string())

    # ---------------- (2) kiem nhi thuc ----------------
    print('\n' + '=' * 74)
    print('  (2) KIEM NHI THUC — tang lech co DUNG muc ngau nhien khong?')
    print('=' * 74)
    print(f"{'he':12s} {'nhom loi':11s} {'tang':11s} {'n':>4s} {'hit':>4s} {'ti le':>7s} {'ngau nhien':>10s} {'p':>9s}  ket luan")
    for (sysname, nhom, layer), g in D.groupby(['system', 'nhom', 'layer']):
        k = int(g.n_cand.median())
        hits, n = int(g.top1.sum()), len(g)
        if n < 10:
            continue
        r = binomtest(hits, n, 1.0 / k)
        obs, exp = 100 * hits / n, 100.0 / k
        verdict = ('= ngau nhien' if r.pvalue > 0.05 else
                   ('TREN ngau nhien' if obs > exp else 'DUOI ngau nhien'))
        print(f'{sysname:12s} {nhom:11s} {layer:11s} {n:4d} {hits:4d} {obs:6.1f}% {exp:9.1f}% {r.pvalue:9.4g}  {verdict}')

    # ---------------- (3) dieu kien pham vi, hai he ----------------
    print('\n' + '=' * 74)
    print('  (3) DIEU KIEN PHAM VI — loi tai nguyen co lam dich WORKLOAD khong?')
    print('=' * 74)
    print(f"{'he':12s} {'nhom':11s} {'n':>4s} {'dich tv':>9s} {'tang/giam':>10s} {'Wilcoxon p':>11s}  ket luan")
    for (sysname, nhom), g in S.assign(nhom=np.where(S.fault.isin(RESOURCE), 'tai nguyen', 'mang')
                                       ).groupby(['system', 'nhom']):
        if len(g) < 8:
            continue
        try:
            p = wilcoxon(g.w_sigma).pvalue
        except Exception:
            p = np.nan
        v = 'D DONG (khong dich)' if p > 0.05 else 'D RO RI (co dich)'
        print(f'{sysname:12s} {nhom:11s} {len(g):4d} {g.w_sigma.median():+8.3f} '
              f'{(g.w_sigma>0).sum():4d}/{(g.w_sigma<0).sum():<5d} {p:11.4g}  {v}')


if __name__ == '__main__':
    main()
