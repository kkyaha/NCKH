# -*- coding: utf-8 -*-
"""RQ6 — HAI CHUNG KIEM SOAT cho lop boc hai truc.

Doc CSV do `rq6_wrapper_rcaeval.py` sinh (phai co cot `xh0`/`xh1`/`ung`, tuc la
ban tu 2026-10-03 tro di). KHONG goi lai phuong phap nao.

CHUNG (a) -- TRUC DON VI: luat long co cuu HON NGAU NHIEN khong?
  Truc don vi doi luat tinh diem: che do (2) tinh dung ca khi `inj` la HAU DUE cua
  top-1. So M2-M1 voi san 1/n la KHONG cong bang: bo xep hang ngau nhien dung ~10%
  nen con ~90% du dia de luat long cuu, phuong phap tot thi gan nhu khong con.
  Nen chung phai CO DIEU KIEN -- chi tren cac ca phuong phap SAI o muc node:
      quan sat = ti le `inj in desc(top1)`
      ngau nhien = |{s in xh1 : s != inj, inj in desc(s)}| / (|xh1| - 1)
  tuc: neu cai top-1 sai ay duoc rut deu tu dung tap ung vien do, no co tinh cach
  la to tien cua `inj` thuong xuyen den the khong?

CHUNG (b) -- SO UNG VIEN: tap ung vien cua lop boc co hon mot tap NGAU NHIEN
  CUNG CO khong? Co dang dong chinh xac, khong can mo phong. Voi mot ca:
      n = |xh0| (toan bo service duoc xep hang o che do 0)
      k = |ung| (co tap ung vien lop boc chon)
      a = so service xep TREN `inj` trong xh0
  Han che xh0 vao mot tap ngau nhien R co k thi top-1 la `inj` <=> `inj` in R VA
  khong service nao trong a service tren no thuoc R:
      P = C(n-1-a, k-1) / C(n, k)
  So P nay voi M0b quan sat duoc.

Chay:  python papers/p2_khop_tang/experiments/rq6_chung_kiem_soat.py <file.csv> [...]
"""
import json
import os
import sys
from math import comb

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

_P = os.path.dirname(os.path.abspath(__file__))
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
RES = os.path.join(_P, 'data', 'processed', 'scm_results')
GRAPH = {'SockShop': 'sockshop_agent_graph.json',
         'OnlineBoutique': 'onlineboutique_agent_graph.json',
         'TrainTicket': 'trainticket_agent_graph.json'}


def desc_of(he):
    G = json.load(open(os.path.join(_P, 'src', 'graph', GRAPH[he]), encoding='utf-8'))
    svcs = {n['id'] if isinstance(n, dict) else n for n in G['nodes']}
    g = nx.DiGraph()
    g.add_nodes_from(svcs)
    g.add_edges_from([(e.get('source', e.get('from')), e.get('target', e.get('to'))) for e in G['edges']])
    return {s: nx.descendants(g, s) for s in g.nodes}


def main():
    files = sys.argv[1:]
    assert files, 'can it nhat mot CSV'
    D = pd.concat([pd.read_csv(os.path.join(RES, f) if not os.path.isabs(f) else f)
                   for f in files], ignore_index=True)
    for c in ('fallback_0', 'fallback_1', 'xh0', 'xh1', 'ung'):
        assert c in D.columns, f'CSV thieu cot `{c}` -- chay lai bang ban wrapper moi'
        D[c] = D[c].fillna('')
    D = D[(D.fallback_0 == '') & (D.fallback_1 == '')].copy()
    DESC = {he: desc_of(he) for he in D.he.unique()}
    print(f'{len(D)} luot hop le | {D.groupby(["he","injected","fault","run"]).ngroups} ca'
          f' | {D.phuong_phap.nunique()} phuong phap\n')

    print('=' * 100)
    print('  CHUNG (a) TRUC DON VI — chi tren ca SAI o muc node, don vi = DUONG')
    print('=' * 100)
    for pp, g in D.groupby('phuong_phap'):
        q = g[(g.don_vi == 'duong') & (g.xh1 != '')]
        obs, exp = [], []
        for _, r in q.iterrows():
            xh1 = r.xh1.split('|')
            if len(xh1) < 2 or xh1[0] == r.injected:
                continue                                  # dung o muc node -> khong co gi de cuu
            dsc = DESC[r.he]
            obs.append(1.0 if r.injected in dsc.get(xh1[0], ()) else 0.0)
            to_tien = sum(1 for s in xh1 if s != r.injected and r.injected in dsc.get(s, ()))
            exp.append(to_tien / (len(xh1) - 1))
        if len(obs) < 8:
            print(f'  {pp:14s} chi {len(obs)} ca sai-muc-node -> khong du de kiem dinh\n'); continue
        obs, exp = np.array(obs), np.array(exp)
        h = int(obs.sum())
        p_bin = binomtest(h, len(obs), float(exp.mean())).pvalue
        try:
            p_w = wilcoxon(obs - exp, zero_method='zsplit').pvalue
        except Exception:
            p_w = np.nan
        print(f'  {pp:14s} n={len(obs):3d} ca sai-muc-node')
        print(f'      cuu QUAN SAT   {100*obs.mean():6.2f}%   ({h}/{len(obs)})')
        print(f'      cuu NGAU NHIEN {100*exp.mean():6.2f}%   (top-1 sai rut deu tu ung vien)')
        print(f'      chenh {100*(obs.mean()-exp.mean()):+6.2f} d   binom p={p_bin:8.3g}'
              f'   wilcoxon p={p_w:8.3g}   {"***" if min(p_bin, p_w) < 0.05 else "KHONG y nghia"}')
        print()

    print('=' * 100)
    print('  CHUNG (b) SO UNG VIEN — tap ung vien lop boc vs tap NGAU NHIEN CUNG CO')
    print('=' * 100)
    for pp, g in D.groupby('phuong_phap'):
        q = g[(g.xh0 != '') & (g.ung != '')]
        obs, exp = [], []
        for _, r in q.iterrows():
            xh0, ung = r.xh0.split('|'), set(r.ung.split('|'))
            n, k = len(xh0), len(ung & set(xh0))
            if r.injected not in xh0 or k < 1 or k > n:
                continue
            a = xh0.index(r.injected)
            obs.append(1.0 if (lambda z: bool(z) and z[0] == r.injected)([s for s in xh0 if s in ung]) else 0.0)
            exp.append(comb(n - 1 - a, k - 1) / comb(n, k) if n - 1 - a >= k - 1 else 0.0)
        if len(obs) < 8:
            print(f'  {pp:14s} chi {len(obs)} ca -> bo qua\n'); continue
        obs, exp = np.array(obs), np.array(exp)
        h = int(obs.sum())
        p_bin = binomtest(h, len(obs), float(exp.mean())).pvalue
        try:
            p_w = wilcoxon(obs - exp, zero_method='zsplit').pvalue
        except Exception:
            p_w = np.nan
        print(f'  {pp:14s} n={len(obs):3d} ca')
        print(f'      M0b QUAN SAT   {100*obs.mean():6.2f}%   ({h}/{len(obs)})')
        print(f'      M0b NGAU NHIEN {100*exp.mean():6.2f}%   (tap ung vien ngau nhien cung co)')
        print(f'      chenh {100*(obs.mean()-exp.mean()):+6.2f} d   binom p={p_bin:8.3g}'
              f'   wilcoxon p={p_w:8.3g}   {"***" if min(p_bin, p_w) < 0.05 else "KHONG y nghia"}')
        print()


if __name__ == '__main__':
    main()
