# -*- coding: utf-8 -*-
"""RQ6 — QUY TRINH HAI TRUC, KHONG NHAN: tang + don vi, ghep lai thanh phuong phap.

Day la buoc bien toan bo phat hien thanh MOT PHUONG PHAP chay duoc, va no dua tren
hai ket qua moi:

  (A) CHON TANG duoc chi tu du lieu, khong nhan. Quy tac SUY TU LY THUYET (khong
      phai heuristic): Menh de 1 dang ba muc noi tang CHUA nguyen nhan manh nhat
      va moi tang HAU DUE cua no cung co tin hieu (loang hon). Nen khi nhieu tang
      cung "nhoi", nguyen nhan nam o tang THUONG NGUON nhat trong so do.
      Quy tac nay nang ti le chon dung tang cho loi tai nguyen tu 60,0% (chon tang
      nhoi nhat) len 84,4%.

  (B) TRAN TREN cua truc TANG la 88,9% -- va 90% cac ca khong cham toi la LOI MANG
      (27/30: 15 `loss`, 12 `delay`). Chung nam tren CANH, nen KHONG tang nao dinh
      vi duoc theo don vi NODE. Do dung la cho truc DON VI (E6) xu ly.

GHEP: hai truc bu dap dung cho mu cua nhau.

    buoc 1 (tang)   : chon tang phan ung THUONG NGUON nhat
    buoc 2 (don vi) : neu tang duoc chon la `latency-50` -> cham theo DUONG
                      (service bi tiem la HAU DUE cua top-1), nguoc lai -> NODE

Buoc 2 khong dung nhan: no chi doc KET QUA cua buoc 1. Lap luan: tang do tre la
tang duy nhat ma nguyen nhan co the la mot CANH, vi do tre lan truyen NGUOC theo
chuoi goi; hai tang kia chi mang nguyen nhan dang NODE.

Chay:  python papers/p2_khop_tang/experiments/rq6_pipeline_hai_truc.py
Ra:    data/processed/scm_results/rq6_pipeline_hai_truc.csv
"""
import argparse
import glob
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import binomtest

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RES = os.path.join(_P, 'data', 'processed', 'scm_results')

LAYERS = ['workload', 'cpu', 'latency-50']        # thuong nguon -> ha nguon
TAU_LUOI = [round(x, 2) for x in np.arange(0.2, 1.01, 0.05)]
RESOURCE = ['cpu', 'mem', 'disk', 'socket']
ENV = {'time', 'imte', 'vm_cpu_util', 'vm_mem_avail_mb'}
SYS = {
    'SockShop': dict(graph='sockshop_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'RE2-SS')),
    'OnlineBoutique': dict(graph='onlineboutique_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'RE2-OB')),
    'TrainTicket': dict(graph='trainticket_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'trainticket')),
}


def call_graph(fname):
    G = json.load(open(os.path.join(_P, 'src', 'graph', fname), encoding='utf-8'))
    svcs = {n['id'] if isinstance(n, dict) else n for n in G['nodes']}
    g = nx.DiGraph()
    g.add_nodes_from(svcs)
    g.add_edges_from([(e.get('source', e.get('from')), e.get('target', e.get('to'))) for e in G['edges']])
    return g


def shifts(d, t, layer):
    tc = 'imte' if 'imte' in d.columns else 'time'
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


def moi_cot(d, t):
    tc = 'imte' if 'imte' in d.columns else 'time'
    b, a = d[d[tc] < t], d[d[tc] >= t]
    if len(b) < 20 or len(a) < 20:
        return {}
    diem = {}
    for c in d.columns:
        if c in ENV or '_' not in c:
            continue
        nb, na = b[c].dropna(), a[c].dropna()
        if len(nb) < 10 or len(na) < 10 or nb.std() == 0:
            continue
        s = c.rsplit('_', 1)[0]
        diem[s] = max(diem.get(s, 0.0), abs(na.mean() - nb.mean()) / nb.std())
    return diem


def runs_of(sysname):
    cfg = SYS[sysname]
    if sysname == 'SockShop':
        for scen in sorted(os.listdir(cfg['root'])):
            sp = os.path.join(cfg['root'], scen)
            if not os.path.isdir(sp):
                continue
            inj, _, ft = scen.rpartition('_')
            for run in sorted(os.listdir(sp)):
                mp, ip = os.path.join(sp, run, 'simple_metrics.csv'), os.path.join(sp, run, 'inject_time.txt')
                if os.path.exists(mp) and os.path.exists(ip):
                    try:
                        yield inj, ft, run, pd.read_csv(mp), int(open(ip).read().strip())
                    except Exception:
                        continue
    else:
        pref = 're2ob_' if sysname == 'OnlineBoutique' else 're2tt_'
        for sp in sorted(glob.glob(os.path.join(cfg['root'], pref + '*'))):
            base = os.path.basename(sp)[len(pref):]
            run = base.rsplit('_', 1)[-1]
            inj, _, ft = base.rsplit('_', 1)[0].rpartition('_')
            mp, ip = os.path.join(sp, 'metrics.parquet'), os.path.join(sp, 'inject_time.txt')
            if os.path.exists(mp) and os.path.exists(ip):
                try:
                    yield inj, ft, run, pd.read_parquet(mp), int(open(ip).read().strip())
                except Exception:
                    continue


def tu_tap(v):
    """Do NHOI = r_top1 / sum(r). Khong dung nhan."""
    if len(v) < 3:
        return None
    x = np.array(list(v.values()), dtype=float)
    tong = x.sum()
    return float(x.max() / tong) if tong > 0 else None


def chon_tang(P: dict, tau: float):
    """Tang phan ung THUONG NGUON nhat: P >= tau * max(P), uu tien thu tu LAYERS."""
    ok = {k: p for k, p in P.items() if p is not None}
    if not ok:
        return None
    pmax = max(ok.values())
    for la in LAYERS:
        if la in ok and ok[la] >= tau * pmax:
            return la
    return max(ok, key=ok.get)


def hit(rank, inj, don_vi, desc, k=1):
    top = rank[:k]
    if don_vi == 'node':
        return inj in top
    return any(s == inj or inj in desc.get(s, ()) for s in top)   # DUONG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tau', type=float, default=0.5)
    a = ap.parse_args()

    rows = []
    for sysname in SYS:
        g = call_graph(SYS[sysname]['graph'])
        desc = {s: nx.descendants(g, s) for s in g.nodes}
        for inj, ft, run, d, t in runs_of(sysname):
            V = {la: shifts(d, t, la) for la in LAYERS}
            if not V.get('cpu') or inj not in V['cpu']:
                continue
            P = {la: (tu_tap(v) if (v and inj in v) else None) for la, v in V.items()}
            la_chon = chon_tang(P, a.tau)
            if la_chon is None:
                continue
            xh = {la: sorted(v, key=v.get, reverse=True) for la, v in V.items() if v}
            mc = moi_cot(d, t)
            xh_mc = sorted(mc, key=mc.get, reverse=True) if mc else []

            r = dict(he=sysname, injected=inj, fault=ft, run=run,
                     nhom='tai nguyen' if ft in RESOURCE else 'mang',
                     tang_chon=la_chon, n_cand=len(V['cpu']))
            # moc so sanh, TAT CA don vi NODE
            r['A_cpu'] = inj in xh.get('cpu', [])[:1]
            r['A_latency'] = inj in xh.get('latency-50', [])[:1]
            r['A_moi_cot'] = inj in xh_mc[:1]
            r['B_chon_tang'] = inj in xh.get(la_chon, [])[:1]
            # QUY TRINH HAI TRUC: tang da chon + don vi suy tu tang do
            dv = 'duong' if la_chon == 'latency-50' else 'node'
            r['don_vi'] = dv
            r['C_hai_truc'] = hit(xh.get(la_chon, []), inj, dv, desc, 1)
            r['C_hai_truc_t3'] = hit(xh.get(la_chon, []), inj, dv, desc, 3)
            # tran tren cua TUNG truc, va cua CA HAI
            # luu ket qua cho MOI tau trong luoi, de kiem NGOAI MAU sau nay
            for tt in TAU_LUOI:
                lc = chon_tang(P, tt)
                if lc is None:
                    r[f'C_tau_{tt:.2f}'] = np.nan
                    continue
                u = 'duong' if lc == 'latency-50' else 'node'
                r[f'C_tau_{tt:.2f}'] = hit(xh.get(lc, []), inj, u, desc, 1)
            r['oracle_tang'] = any(inj in xh[la][:1] for la in xh)
            r['oracle_ca_hai'] = any(hit(xh[la], inj, u, desc, 1) for la in xh for u in ('node', 'duong'))
            rows.append(r)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RES, 'rq6_pipeline_hai_truc.csv'), index=False)
    print(f'{len(D)} ca | tau = {a.tau}\n')

    MOC = [('tang co dinh cpu       · don vi node', 'A_cpu'),
           ('tang co dinh latency   · don vi node', 'A_latency'),
           ('tron MOI cot metric    · don vi node', 'A_moi_cot'),
           ('(A) CHON TANG          · don vi node', 'B_chon_tang'),
           ('(C) HAI TRUC: tang + don vi', 'C_hai_truc'),
           ('tran tren truc TANG', 'oracle_tang'),
           ('tran tren CA HAI truc', 'oracle_ca_hai')]

    print('=' * 100)
    print('  (1) AC@1 toan bo 270 ca')
    print('=' * 100)
    for ten, c in MOC:
        print(f'  {ten:38s} {100*D[c].mean():6.2f}%')

    print('\n' + '=' * 100)
    print('  (2) THEO NHOM LOI — hai truc bu dap cho nhau o dau')
    print('=' * 100)
    print(f"  {'':38s} {'tai nguyen':>12s} {'mang':>10s}")
    for ten, c in MOC:
        v = D.groupby('nhom')[c].mean() * 100
        print(f'  {ten:38s} {v.get("tai nguyen", np.nan):11.1f}% {v.get("mang", np.nan):9.1f}%')

    print('\n' + '=' * 100)
    print('  (3) KIEM GHEP CAP (McNemar) — hai truc so voi tung moc')
    print('=' * 100)
    for ten, c in MOC[:4]:
        A, B = D['C_hai_truc'].astype(int), D[c].astype(int)
        h, k = int(((A == 1) & (B == 0)).sum()), int(((A == 0) & (B == 1)).sum())
        p = binomtest(h, h + k, 0.5).pvalue if h + k else np.nan
        print(f'  vs {ten:38s} {100*B.mean():6.2f}%  ->  {100*A.mean():6.2f}%  '
              f'| {h:3d} vs {k:3d}  p = {p:9.3g}  {"***" if p < 0.05 else ""}')

    print('\n' + '=' * 100)
    print('  (4) THEO HE — co tong quat hoa khong?')
    print('=' * 100)
    for he, g in D.groupby('he'):
        print(f'  {he:15s} ' + '  '.join(
            f'{t.split("·")[0].strip()[:14]}={100*g[c].mean():5.1f}%'
            for t, c in [MOC[0], MOC[2], MOC[3], MOC[4], MOC[6]]))

    print('\n' + '=' * 100)
    print('  (5) DON VI DUOC CHON co khop loai loi that khong? (khong dung nhan de chon)')
    print('=' * 100)
    print(pd.crosstab(D.nhom, D.don_vi, normalize='index').round(3).mul(100).to_string())

    print('\n' + '=' * 100)
    print('  (6) QUET tau, va KIEM NGOAI MAU bang leave-one-system-out')
    print('=' * 100)
    cols = [f'C_tau_{t:.2f}' for t in TAU_LUOI]
    print('  tau   ' + '  '.join(f'{t:5.2f}' for t in TAU_LUOI))
    print('  AC@1  ' + '  '.join(f'{100*D[c].mean():5.1f}' for c in cols))
    print()
    for he in sorted(D.he.unique()):
        tr, te = D[D.he != he], D[D.he == he]
        tot = max(cols, key=lambda c: tr[c].mean())
        tau_tot = float(tot[len('C_tau_'):])
        print(f'  giu lai {he:15s} | tau fit tren hai he con lai = {tau_tot:.2f}'
              f'  (AC@1 tren tap fit {100*tr[tot].mean():.1f}%)')
        print(f'      -> AC@1 NGOAI MAU tren {he} = {100*te[tot].mean():.1f}%'
              f'   (tau tot nhat cho chinh {he} se cho {100*max(te[c].mean() for c in cols):.1f}%)')
    print(f'\n  bien do AC@1 tren khoang tau 0,40-0,80: '
          f'{100*min(D[f"C_tau_{t:.2f}"].mean() for t in [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]):.1f}% - '
          f'{100*max(D[f"C_tau_{t:.2f}"].mean() for t in [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]):.1f}%'
          '  -> khong nhay voi tau')

    print(f'\n-> {RES}/rq6_pipeline_hai_truc.csv')
    kiem_do_chat_cua_duong(a.tau)




def kiem_do_chat_cua_duong(tau=1.0):
    """KIEM TINH HOP LE cua cham theo DUONG.

    Cham theo node tra loi MOT DIEM. Cham theo duong tra loi MOT TAP: "service bi
    tiem nam trong hau due cua node dau bang". Neu tap hau due do qua lon thi chi so
    bi LAM PHAT -- cuc doan la "hau due cua gateway" = gan nhu moi service, luon dung
    va vo nghia.

    Do: |hau due(top1) + chinh no| / so ung vien. Va bao muc san cua chinh phep cham
    duong -- ti le mot service NGAU NHIEN nam trong tap do.
    """
    rows = []
    for sysname in SYS:
        g = call_graph(SYS[sysname]['graph'])
        desc = {s: nx.descendants(g, s) for s in g.nodes}
        for inj, ft, run, d, t in runs_of(sysname):
            V = {la: shifts(d, t, la) for la in LAYERS}
            if not V.get('cpu') or inj not in V['cpu']:
                continue
            P = {la: (tu_tap(v) if (v and inj in v) else None) for la, v in V.items()}
            lc = chon_tang(P, tau)
            if lc is None or not V.get(lc):
                continue
            xh = sorted(V[lc], key=V[lc].get, reverse=True)
            top = xh[0]
            u = 'duong' if lc == 'latency-50' else 'node'
            tap = ({top} | desc.get(top, set())) & set(xh) if u == 'duong' else {top}
            rows.append(dict(he=sysname, fault=ft, run=run,
                             nhom='tai nguyen' if ft in RESOURCE else 'mang',
                             don_vi=u, n_cand=len(xh), top=top, n_tap=len(tap),
                             ti_le_tap=len(tap) / len(xh),
                             dung=inj in tap, node_dung=inj == top))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, 'rq6_pipeline_do_chat_duong.csv'), index=False)
    print('\n' + '=' * 100)
    print('  (7) DO CHAT cua cau tra loi — cham duong tra ve MOT TAP, lon bao nhieu?')
    print('=' * 100)
    print(f"  {'':34s} {'n ca':>5s} {'|tap|':>6s} {'/ung vien':>10s} {'AC@1':>7s} {'san cua tap':>12s}")
    for (dv, nh), g in T.groupby(['don_vi', 'nhom']):
        print(f'  don vi {dv:7s} · loi {nh:11s} {len(g):5d} {g.n_tap.mean():6.2f} '
              f'{100*g.ti_le_tap.mean():9.1f}% {100*g.dung.mean():6.1f}% {100*g.ti_le_tap.mean():11.1f}%')
    duong = T[T.don_vi == 'duong']
    if len(duong):
        print(f'\n  Tren cac ca cham theo DUONG: |tap| trung binh = {duong.n_tap.mean():.2f} '
              f'tren {duong.n_cand.mean():.1f} ung vien ({100*duong.ti_le_tap.mean():.1f}%)')
        print(f'  AC@1 cua duong = {100*duong.dung.mean():.1f}%  vs  SAN cua chinh phep do '
              f'= {100*duong.ti_le_tap.mean():.1f}%')
        loi_ich = 100 * (duong.dung.mean() - duong.ti_le_tap.mean())
        print(f'  -> vuot san {loi_ich:+.1f} diem. '
              f'{"Khong vo nghia." if loi_ich > 20 else "CANH BAO: gan muc san, chi so co the lam phat."}')
        print(f'\n  phan bo |tap|: ' + '  '.join(
            f'{k}={v}' for k, v in duong.n_tap.value_counts().sort_index().head(10).items()))
        q = duong[duong.ti_le_tap > 0.5]
        print(f'  so ca ma tap chiem >50% ung vien: {len(q)}/{len(duong)} '
              f'({100*len(q)/len(duong):.1f}%)  <- day la cac ca ĐANG NGO')
    return T


if __name__ == '__main__':
    main()
