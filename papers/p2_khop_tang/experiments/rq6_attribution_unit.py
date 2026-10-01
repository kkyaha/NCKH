# -*- coding: utf-8 -*-
"""RQ6 — DON VI QUY GAN: loi MANG nam o CANH, khong o NODE.

PHAT HIEN. Loi tai nguyen (cpu/mem/disk/socket) dinh vi duoc tren tang CPU voi don vi
NODE (98.3% top-1). Loi mang (delay/loss) thi khong: tren Train Ticket chi 6.7-60%.
Chan doan: mat goi / do tre xay ra tren CANH giua hai service, nen tin hieu do tre lon
nhat o node QUAN SAT canh do (caller / to tien), KHONG o service bi tiem -- service do
khong hong, DUONG TOI no hong.

Do duoc tren Train Ticket, tang latency, loi loss: 9/15 lan node dan dau la TO TIEN hoac
CALLER cua service bi tiem; service bi tiem xep hang 4/28; node dan dau dich 104.5 sigma
so voi 4.6 sigma cua service bi tiem (23x).

HE QUA VE GIAO THUC DANH GIA. Nhan cua RCAEval gan loi mang theo SERVICE, nhung thuc the
bi hong la CANH tới service do. Cham diem theo don vi dung thi ket qua doi han.

Chay:  python papers/p2_khop_tang/experiments/rq6_attribution_unit.py
"""
import glob
import json
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RES = os.path.join(_P, 'data', 'processed', 'scm_results')
NETWORK = ['delay', 'loss']
SYS = {
    'SockShop':    dict(graph='sockshop_agent_graph.json',    root=os.path.join(_P, 'data', 'raw', 'RE2-SS')),
    'OnlineBoutique': dict(graph='onlineboutique_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'RE2-OB')),
    'TrainTicket': dict(graph='trainticket_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'trainticket')),
}


def call_graph(fname):
    G = json.load(open(os.path.join(_P, 'src', 'graph', fname), encoding='utf-8'))
    svcs = {n['id'] if isinstance(n, dict) else n for n in G['nodes']}
    g = nx.DiGraph(); g.add_nodes_from(svcs)
    g.add_edges_from([(e.get('source', e.get('from')), e.get('target', e.get('to'))) for e in G['edges']])
    return g


def shifts(d, t, layer):
    """|trung binh sau - trung binh truoc| / std truoc, tung service, MOT tang."""
    tc = 'imte' if 'imte' in d.columns else 'time'
    b, a = d[d[tc] < t], d[d[tc] >= t]
    if len(b) < 20 or len(a) < 20:
        return {}
    out = {}
    for c in d.columns:
        if not c.endswith('_' + layer):
            continue
        s = c[: -(len(layer) + 1)]
        nb, na = b[c].dropna(), a[c].dropna()
        if len(nb) < 10 or len(na) < 10 or nb.std() == 0:
            continue
        out[s] = abs(na.mean() - nb.mean()) / nb.std()
    return out


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
                        yield inj, ft, pd.read_csv(mp), int(open(ip).read().strip())
                    except Exception:
                        continue
    else:
        pref = 're2ob_' if sysname == 'OnlineBoutique' else 're2tt_'
        for sp in sorted(glob.glob(os.path.join(cfg['root'], pref + '*'))):
            base = os.path.basename(sp)[len(pref):].rsplit('_', 1)[0]
            inj, _, ft = base.rpartition('_')
            mp, ip = os.path.join(sp, 'metrics.parquet'), os.path.join(sp, 'inject_time.txt')
            if os.path.exists(mp) and os.path.exists(ip):
                try:
                    yield inj, ft, pd.read_parquet(mp), int(open(ip).read().strip())
                except Exception:
                    continue


def main():
    rows = []
    for sysname in SYS:
        g = call_graph(SYS[sysname]['graph'])
        desc = {s: nx.descendants(g, s) for s in g.nodes}
        for inj, ft, d, t in runs_of(sysname):
            if ft not in NETWORK:
                continue
            v = shifts(d, t, 'latency-50')
            if len(v) < 4 or inj not in v:
                continue
            rank = sorted(v, key=v.get, reverse=True)

            def node_hit(k):   return inj in rank[:k]
            def edge_hit(k):   return any(s == inj or g.has_edge(s, inj) for s in rank[:k])
            def path_hit(k):   return any(s == inj or inj in desc.get(s, ()) for s in rank[:k])

            top = rank[0]
            rel = ('chinh no' if top == inj else
                   'CALLER truc tiep' if g.has_edge(top, inj) else
                   'to tien' if inj in desc.get(top, ()) else
                   'hau due' if top in desc.get(inj, ()) else 'khong lien quan')
            rows.append(dict(system=sysname, fault=ft, injected=inj, n_cand=len(v),
                             rank_injected=rank.index(inj) + 1, top1=top, quan_he=rel,
                             shift_injected=v[inj], shift_top=v[top],
                             node_t1=node_hit(1), node_t3=node_hit(3),
                             edge_t1=edge_hit(1), edge_t3=edge_hit(3),
                             path_t1=path_hit(1), path_t3=path_hit(3)))
    D = pd.DataFrame(rows)
    out = os.path.join(RES, 'rq6_attribution_unit.csv')
    D.to_csv(out, index=False)
    print(f'{len(D)} run loi mang (delay+loss), hai he | da ghi {out}\n')

    print('=' * 78)
    print('  NODE DAN DAU la ai? (tang latency, loi mang)')
    print('=' * 78)
    for sysname, g in D.groupby('system'):
        print(f'\n-- {sysname} (n={len(g)}, ~{g.n_cand.median():.0f} ung vien) --')
        print(g.quan_he.value_counts().to_string())
        print(f'   service bi tiem xep hang trung vi: {g.rank_injected.median():.0f}/{g.n_cand.median():.0f}')
        print(f'   dich chuyen: bi tiem {g.shift_injected.median():.1f} sigma  vs  dan dau {g.shift_top.median():.1f} sigma'
              f'  ({g.shift_top.median()/max(g.shift_injected.median(),1e-9):.0f}x)')

    print('\n' + '=' * 78)
    print('  CHAM THEO NODE vs CANH vs DUONG')
    print('=' * 78)
    for sysname, g in D.groupby('system'):
        print(f'\n-- {sysname} (ngau nhien theo node ~{100/g.n_cand.median():.1f}%) --')
        t = g.groupby('fault').agg(n=('node_t1', 'size'),
                                   **{k: (k, 'mean') for k in
                                      ('node_t1', 'node_t3', 'edge_t1', 'edge_t3', 'path_t1', 'path_t3')})
        for c in t.columns:
            if c != 'n':
                t[c] = (t[c] * 100).round(1)
        print(t.to_string())
        print('   gop:  ' + '  '.join(
            f'{lab}={100*g[c].mean():.1f}%' for lab, c in
            (('node_t1', 'node_t1'), ('canh_t1', 'edge_t1'), ('duong_t1', 'path_t1'),
             ('duong_t3', 'path_t3'))))

    print('\n=> Loi mang nam o CANH. Cham theo don vi NODE la cham sai don vi.')
    print('   Nhan cua RCAEval gan loi mang theo SERVICE; thuc the bi hong la DUONG TOI service do.')


if __name__ == '__main__':
    main()
