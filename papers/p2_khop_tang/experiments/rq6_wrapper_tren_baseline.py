# -*- coding: utf-8 -*-
"""RQ6 — HAI TRUC LAM LOP BOC TREN MOT PHUONG PHAP DA XUAT BAN.

Cau hoi: quy trinh hai truc (chon tang + chon don vi) co phai mot PHUONG PHAP
RIENG, hay la mot LOP BOC nang duoc bat ky bo xep hang nao?

Neu la lop boc thi phat bieu manh hon han: khong phai "phuong phap cua chung toi
tot hon BARO" ma "BARO, va moi phuong phap cung dang, duoc loi N diem tu mot lop
boc khong ton gi".

Thiet ke. Giu NGUYEN bo xep hang, chi doi hai nut:

  (0) NHU DA XUAT BAN : cham MOI cot metric, gop ve service bang max, don vi NODE.
                        Day dung la cach RCAEval chay cac baseline cua no.
  (1) + TRUC TANG     : cham rieng tung tang, chon tang co diem TAP TRUNG nhat
                        (P = r_top1 / sum r), don vi NODE.
  (2) + CA HAI TRUC   : nhu (1), roi neu tang duoc chon la `latency-50` -> cham
                        theo DUONG, nguoc lai -> NODE.

Hai bo xep hang:
  MEANSHIFT = |trung binh sau - trung binh truoc| / std truoc
  BARO      = RobustScaler median/IQR tren cua so truoc, roi max|z| tren cua so sau
              (luat cua RCAEval `e2e/baro.py`, FSE'24)

LUU Y VE TAP UNG VIEN. (0) xep hang moi service co BAT KY cot metric nao; (1)/(2)
chi xep hang service co cot CUA TANG DO. Nen tap ung vien khong hoan toan trung
nhau. Script bao them cot `n_cand` cua tung che do de doc duoc chenh lech nay, va
mot bang chi tinh tren GIAO cua ba tap de so sanh that khop.

Chay:  python papers/p2_khop_tang/experiments/rq6_wrapper_tren_baseline.py
Ra:    data/processed/scm_results/rq6_wrapper_tren_baseline.csv
"""
import glob
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import RobustScaler

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RES = os.path.join(_P, 'data', 'processed', 'scm_results')

LAYERS = ['workload', 'cpu', 'latency-50']
RESOURCE = ['cpu', 'mem', 'disk', 'socket']
ENV = {'time', 'imte', 'vm_cpu_util', 'vm_mem_avail_mb'}
SYS = {
    'SockShop': dict(graph='sockshop_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'RE2-SS')),
    'OnlineBoutique': dict(graph='onlineboutique_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'RE2-OB')),
    'TrainTicket': dict(graph='trainticket_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'trainticket')),
}


# ----------------------------------------------------------------- bo xep hang
def diem_meanshift(nb, na):
    if len(nb) < 5 or len(na) < 5 or nb.std() == 0:
        return None
    return abs(na.mean() - nb.mean()) / nb.std()


def diem_baro(nb, na):
    n, a = nb.to_numpy(float), na.to_numpy(float)
    n, a = n[np.isfinite(n)], a[np.isfinite(a)]
    if len(n) < 5 or len(a) < 5 or np.all(n == n[0]):
        return None
    z = RobustScaler().fit(n.reshape(-1, 1)).transform(a.reshape(-1, 1))
    return float(np.max(np.abs(z)))


THONG_KE = {'MEANSHIFT': diem_meanshift, 'BARO': diem_baro}


# ----------------------------------------------------------------- du lieu
def call_graph(fname):
    G = json.load(open(os.path.join(_P, 'src', 'graph', fname), encoding='utf-8'))
    svcs = {n['id'] if isinstance(n, dict) else n for n in G['nodes']}
    g = nx.DiGraph()
    g.add_nodes_from(svcs)
    g.add_edges_from([(e.get('source', e.get('from')), e.get('target', e.get('to'))) for e in G['edges']])
    return g


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


def cua_so(d, t):
    tc = 'imte' if 'imte' in d.columns else 'time'
    b, a = d[d[tc] < t], d[d[tc] >= t]
    return (b, a) if len(b) >= 20 and len(a) >= 20 else (None, None)


def main():
    rows = []
    for sysname in SYS:
        g = call_graph(SYS[sysname]['graph'])
        desc = {s: nx.descendants(g, s) for s in g.nodes}
        for inj, ft, run, d, t in runs_of(sysname):
            b, a = cua_so(d, t)
            if b is None:
                continue
            cot = [c for c in d.columns if c not in ENV and '_' in c]
            for ten_tk, fn in THONG_KE.items():
                # diem tung COT
                dc = {}
                for c in cot:
                    v = fn(b[c].dropna(), a[c].dropna())
                    if v is not None and np.isfinite(v):
                        dc[c] = v
                if not dc:
                    continue
                # (0) NHU DA XUAT BAN: gop moi cot ve service bang max
                g0 = {}
                for c, v in dc.items():
                    s = c.rsplit('_', 1)[0]
                    g0[s] = max(g0.get(s, -np.inf), v)
                xh0 = sorted(g0, key=g0.get, reverse=True)
                # diem theo TUNG TANG
                per = {}
                for la in LAYERS:
                    v = {c[:-(len(la) + 1)]: dc[c] for c in dc if c.endswith('_' + la)}
                    if len(v) >= 3:
                        per[la] = v
                if not per:
                    continue
                P = {la: max(v.values()) / sum(v.values()) for la, v in per.items() if sum(v.values()) > 0}
                if not P:
                    continue
                la_chon = max(P, key=P.get)
                xh1 = sorted(per[la_chon], key=per[la_chon].get, reverse=True)
                dv = 'duong' if la_chon == 'latency-50' else 'node'

                def trung(xh, u, k=1):
                    top = xh[:k]
                    if u == 'node':
                        return inj in top
                    return any(s == inj or inj in desc.get(s, ()) for s in top)

                # (0b) KHOP TAP UNG VIEN: van tron MOI cot, nhung CHI tren dung
                # nhung service co trong tap ung vien cua tang da chon. Tach hieu
                # ung TANG khoi hieu ung SO UNG VIEN -- neu khong co dong nay thi
                # muc nang cua lop boc lan voi viec tap ung vien co lai
                # (Train Ticket: 68 -> 27 service).
                ung = set(per[la_chon])
                g0b = {s_: v for s_, v in g0.items() if s_ in ung}
                xh0b = sorted(g0b, key=g0b.get, reverse=True) if g0b else []

                rows.append(dict(
                    he=sysname, injected=inj, fault=ft, run=run, thong_ke=ten_tk,
                    nhom='tai nguyen' if ft in RESOURCE else 'mang',
                    tang_chon=la_chon, don_vi=dv,
                    n_cand_0=len(xh0), n_cand_0b=len(xh0b), n_cand_1=len(xh1),
                    M0_nhu_xuat_ban=trung(xh0, 'node'),
                    M0b_khop_ung_vien=trung(xh0b, 'node') if xh0b else np.nan,
                    M1_them_truc_tang=trung(xh1, 'node'),
                    M2_ca_hai_truc=trung(xh1, dv),
                    M0_t3=trung(xh0, 'node', 3), M2_t3=trung(xh1, dv, 3)))

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RES, 'rq6_wrapper_tren_baseline.csv'), index=False)
    MOC = [('(0) NHU DA XUAT BAN (moi cot, node)', 'M0_nhu_xuat_ban'),
           ('(0b) moi cot, KHOP tap ung vien', 'M0b_khop_ung_vien'),
           ('(1) + truc TANG (khop ung vien)', 'M1_them_truc_tang'),
           ('(2) + CA HAI TRUC', 'M2_ca_hai_truc')]
    print(f'{len(D)} dong = (ca x thong ke) | {D.groupby(["he","injected","fault","run"]).ngroups} ca\n')

    print('=' * 102)
    print('  (1) LOP BOC CO NANG DUOC MOT PHUONG PHAP DA XUAT BAN KHONG?  (AC@1)')
    print('=' * 102)
    print(f"  {'thong ke':11s} {'che do':36s} {'AC@1':>7s} {'tai nguyen':>11s} {'mang':>8s}")
    for tk, gt in D.groupby('thong_ke'):
        for ten, c in MOC:
            v = gt.groupby('nhom')[c].mean() * 100
            print(f'  {tk:11s} {ten:36s} {100*gt[c].mean():6.2f}% '
                  f'{v.get("tai nguyen", np.nan):10.1f}% {v.get("mang", np.nan):7.1f}%')
        print()

    print('=' * 102)
    print('  (2) MUC NANG, va kiem ghep cap McNemar')
    print('=' * 102)
    for tk, gt in D.groupby('thong_ke'):
        A = gt['M2_ca_hai_truc'].astype(int)
        for ten, c in MOC[:3]:
            B = gt[c].astype(int)
            h, k = int(((A == 1) & (B == 0)).sum()), int(((A == 0) & (B == 1)).sum())
            p = binomtest(h, h + k, 0.5).pvalue if h + k else np.nan
            print(f'  {tk:11s} {ten:36s} {100*B.mean():6.2f}% -> {100*A.mean():6.2f}%'
                  f'  ({100*(A.mean()-B.mean()):+6.2f} diem) | {h:3d} vs {k:3d}  p = {p:9.3g}'
                  f'  {"***" if p < 0.05 else ""}')
        print()

    print('=' * 102)
    print('  (3) THEO HE')
    print('=' * 102)
    for (tk, he), gt in D.groupby(['thong_ke', 'he']):
        print(f'  {tk:11s} {he:15s} ' + '  '.join(
            f'{t[:7]}={100*gt[c].mean():5.1f}%' for t, c in MOC))

    print('\n' + '=' * 102)
    print('  (4) AC@3')
    print('=' * 102)
    for tk, gt in D.groupby('thong_ke'):
        print(f'  {tk:11s} nhu xuat ban {100*gt.M0_t3.mean():5.1f}%  ->  ca hai truc {100*gt.M2_t3.mean():5.1f}%')

    print('\n' + '=' * 102)
    print('  (5) TACH HAI HIEU UNG: so ung vien  vs  tang')
    print('=' * 102)
    for tk, gt in D.groupby('thong_ke'):
        m0, m0b, m1, m2 = (100 * gt[c].mean() for _, c in MOC)
        print(f'  {tk}')
        print(f'    (0)  moi cot, MOI ung vien          {m0:6.2f}%')
        print(f'    (0b) moi cot, khop tap ung vien     {m0b:6.2f}%   <- do SO UNG VIEN dong gop {m0b-m0:+6.2f} diem')
        print(f'    (1)  loc TANG, khop tap ung vien    {m1:6.2f}%   <- do TANG      dong gop {m1-m0b:+6.2f} diem')
        print(f'    (2)  + truc DON VI                  {m2:6.2f}%   <- do DON VI    dong gop {m2-m1:+6.2f} diem')
        print(f'    tong muc nang cua lop boc           {m2-m0:+6.2f} diem\n')
    print(D.groupby(['thong_ke', 'he'])[['n_cand_0', 'n_cand_0b', 'n_cand_1']].median().to_string())

    print(f'\n-> {RES}/rq6_wrapper_tren_baseline.csv')


if __name__ == '__main__':
    main()
