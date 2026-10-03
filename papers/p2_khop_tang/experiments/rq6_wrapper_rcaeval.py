# -*- coding: utf-8 -*-
"""RQ6 — LOP BOC HAI TRUC TREN CODE CUA CHINH TAC GIA (repo RCAEval).

Day la ban thay cho `rq6_wrapper_tren_baseline.py`: thay vi CAI LAI luat cua BARO,
script nay goi THUC SU cac ham trong repo `phamquiluan/RCAEval`. Do la phan bien
manh nhat con lai -- "co the day chi la tao tac cua cach cac tac gia cai lai BARO".

BON PHUONG PHAP chay duoc tren may nay (python 3.9):
  baro          BARO, FSE'24            -- code cua tac gia
  nsigma        N-sigma baseline        -- code cua tac gia
  circa         CIRCA, KDD'22           -- code cua tac gia
  pc_randomwalk PC + Random Walk        -- code cua tac gia

Cac ham khac trong repo khong chay duoc o day, da ghi ro ly do:
  e_diagnosis, ht      thieu goi `pyrca`
  microcause, easyrca, cloudranger  thieu `pingouin`
  *_pagerank, causalrca             thieu `sknetwork`
  rcd                  causallearn khong tuong thich phien ban
  run                  can CUDA
  mscred               NaN trong luc train
  tracerca, microrank  can du lieu trace dang khac ('methodName')

BON CHE DO, giu NGUYEN phuong phap, chi doi du lieu dua vao:
  (0)  NHU DA XUAT BAN : toan bo khung metric, don vi NODE
  (0b) KHOP UNG VIEN   : cung thu hang (0), nhung chi giu service thuoc tap ung
                         vien cua tang da chon -- tach hieu ung SO UNG VIEN
  (1)  + TRUC TANG     : goi lai phuong phap voi khung CHI co cot cua tang da chon
  (2)  + TRUC DON VI   : nhu (1), neu tang da chon la latency -> cham theo DUONG

Tang duoc chon tu DU LIEU THO bang do tap trung P = r_top1 / sum(r) cua dich
chuyen trung binh -- KHONG dung nhan, va KHONG dung noi that cua phuong phap.
Nen lop boc la mot BO TIEN XU LY, cam vao truoc bat ky phuong phap nao.

Loi `inj in v` trong dieu kien chon tang DA BI XOA (2026-10-03). Truoc do tang chi
duoc coi la hop le neu service bi tiem co mat trong do, va ca nao khong tang nao
chua ground truth thi bi loai han -- tuc cau "KHONG dung nhan" o tren la SAI.
Do lai: 0/178 ca bi loai boi dieu kien ay, 7/178 (3,9%) doi tang khi bo nhan, nen
anh huong nho; nhung day la ro ri nhan that va reviewer doc code se thay.

CHUNG KIEM SOAT con thieu -- vi sao CSV phai luu thu hang:
  (a) truc DON VI doi LUAT TINH DIEM (tinh dung ca khi inj la hau due cua top-1),
      nen san ngau nhien cua no cao hon 1/n. Do duoc: lam phat co hoc, pha loang
      tren toan bo ca, la +7,38 d (SS) / +11,14 d (OB) / +14,50 d (TT) -- NGANG
      hoac HON muc quan sat (+7,84 baro / +8,21 nsigma / +9,09 circa / +0,00 pc).
      Phep so sanh tho ay nghieng bat loi qua muc (bo xep hang ngau nhien con
      ~90% du dia de luat long cuu, phuong phap tot thi khong), nen chung DUNG
      phai co dieu kien: CHI tren cac ca phuong phap sai o muc node, ti le inj la
      hau due cua top-1 co cao hon ngau nhien khong?
  (b) phan ung vien cua BARO (+25,00 d) vuot san 1/n tan +22,31 d, nhung 1/n la
      proxy yeu; chung dung la tap ung vien NGAU NHIEN cung co.
Ca hai chung chi can thu hang + do thi goi, nen luu `xh0`/`xh1`/`ung` la du.

BAT FALLBACK AM THAM. Decorator `rca` cua RCAEval bat moi ngoai le va tra ve THU
TU COT thay vi bao loi. Mot ket qua nhu vay KHONG phai ket qua cua phuong phap.
Script bat stderr va danh dau `fallback_*`; moi bang deu loai cac luot do. Day
dung lop bug da tung pha huy 600 dong Gemini trong du an nay.

Chay:  PYTHONPATH=<duong dan RCAEval> python papers/p2_khop_tang/experiments/rq6_wrapper_rcaeval.py
Ra:    data/processed/scm_results/rq6_wrapper_rcaeval.csv   (ghi dan, dung giua van dung duoc)
"""
import argparse
import contextlib
import glob
import importlib
import io
import json
import os
import sys
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RES = os.path.join(_P, 'data', 'processed', 'scm_results')

LAYERS = ['workload', 'cpu', 'latency-50']
RESOURCE = ['cpu', 'mem', 'disk', 'socket']
DUOI_METRIC = ['latency-50', 'latency-90', 'workload', 'cpu', 'mem', 'diskio', 'disk', 'socket']
SYS = {
    'SockShop': dict(graph='sockshop_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'RE2-SS')),
    'OnlineBoutique': dict(graph='onlineboutique_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'RE2-OB')),
    'TrainTicket': dict(graph='trainticket_agent_graph.json', root=os.path.join(_P, 'data', 'raw', 'trainticket')),
}


def nap_phuong_phap():
    """Nap cac ham e2e cua RCAEval. Bao ro cai nao khong co, khong doan."""
    import RCAEval.e2e as e2e
    pp = {}
    for ten in ('nsigma',):
        if hasattr(e2e, ten):
            pp[ten] = getattr(e2e, ten)
    for mod, ten in (('baro', 'baro'), ('circa', 'circa'), ('pc_randomwalk', 'pc_randomwalk')):
        try:
            m = importlib.import_module(f'RCAEval.e2e.{mod}')
            if hasattr(m, ten):
                pp[ten] = getattr(m, ten)
        except Exception as e:
            print(f'[nap] bo qua {ten}: {type(e).__name__}: {str(e)[:60]}', file=sys.stderr)
    return pp


def col2svc(c):
    for d in DUOI_METRIC:
        if c.endswith('_' + d):
            return c[:-(len(d) + 1)]
    return c.rsplit('_', 1)[0] if '_' in c else c


def lam_sach(d):
    """Bo cot NaN va cot hang so, giu `time`.

    Can thiet: circa/pc_randomwalk NEM AssertionError('Input data contains NaN')
    roi bi decorator `rca` bien thanh thu tu cot. Ap DONG NHAT cho MOI che do nen
    khong phai mot bien gay lan.
    """
    d = d.rename(columns={'imte': 'time'})
    tc = d['time']
    d = d.drop(columns=['time']).dropna(axis=1)
    d = d.loc[:, d.nunique() > 1]
    d.insert(0, 'time', tc.values)
    return d


def dich_chuyen(d, t, layer):
    b, a = d[d['time'] < t], d[d['time'] >= t]
    if len(b) < 20 or len(a) < 20:
        return {}
    out = {}
    for c in d.columns:
        if not c.endswith('_' + layer):
            continue
        nb, na = b[c].dropna(), a[c].dropna()
        if len(nb) < 10 or len(na) < 10 or nb.std() == 0:
            continue
        out[c[:-(len(layer) + 1)]] = abs(na.mean() - nb.mean()) / nb.std()
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


def call_graph(fname):
    G = json.load(open(os.path.join(_P, 'src', 'graph', fname), encoding='utf-8'))
    svcs = {n['id'] if isinstance(n, dict) else n for n in G['nodes']}
    g = nx.DiGraph()
    g.add_nodes_from(svcs)
    g.add_edges_from([(e.get('source', e.get('from')), e.get('target', e.get('to'))) for e in G['edges']])
    return g


def goi(fn, d, t):
    """Goi mot phuong phap, tra (thu hang service, co_fallback, giay)."""
    err = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            out = fn(d, inject_time=t, dataset=None)
    except Exception as e:
        return None, f'NEM:{type(e).__name__}', time.time() - t0
    dt = time.time() - t0
    r = out.get('ranks') if isinstance(out, dict) else out
    if not r:
        return None, 'RONG', dt
    fb = 'failed' in err.getvalue()
    svc = []
    for c in r:
        if c == 'time':
            continue
        s = col2svc(c)
        if s not in svc:
            svc.append(s)
    return svc, ('FALLBACK' if fb else ''), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--he', default=None, help='chi chay mot he (SockShop/OnlineBoutique/TrainTicket)')
    ap.add_argument('--pp', default=None, help='danh sach phuong phap, cach nhau bang dau phay')
    ap.add_argument('--ra', default='rq6_wrapper_rcaeval.csv')
    a = ap.parse_args()

    PP = nap_phuong_phap()
    if a.pp:
        PP = {k: v for k, v in PP.items() if k in a.pp.split(',')}
    print(f'[nap] {len(PP)} phuong phap: {", ".join(PP)}', flush=True)
    assert PP, 'khong nap duoc phuong phap nao -- dat PYTHONPATH tro vao repo RCAEval'

    out_path = os.path.join(RES, a.ra)
    rows, n_ca = [], 0
    hes = [a.he] if a.he else list(SYS)
    for sysname in hes:
        g = call_graph(SYS[sysname]['graph'])
        desc = {s: nx.descendants(g, s) for s in g.nodes}
        for inj, ft, run, draw, t in runs_of(sysname):
            d = lam_sach(draw)
            per = {la: v for la, v in ((la, dich_chuyen(d, t, la)) for la in LAYERS)
                   if len(v) >= 3}
            if not per:
                continue
            P = {la: max(v.values()) / sum(v.values()) for la, v in per.items() if sum(v.values()) > 0}
            if not P:
                continue
            la_chon = max(P, key=P.get)
            ung = set(per[la_chon])
            cot_tang = ['time'] + [c for c in d.columns if c.endswith('_' + la_chon)]
            dv = 'duong' if la_chon == 'latency-50' else 'node'
            n_ca += 1

            for ten, fn in PP.items():
                xh0, fb0, s0 = goi(fn, d.copy(), t)
                xh1, fb1, s1 = goi(fn, d[cot_tang].copy(), t)

                def trung(xh, u, k=1):
                    if not xh:
                        return np.nan
                    top = xh[:k]
                    if u == 'node':
                        return inj in top
                    return any(s == inj or inj in desc.get(s, ()) for s in top)

                xh0b = [s for s in xh0 if s in ung] if xh0 else None
                rows.append(dict(
                    he=sysname, injected=inj, fault=ft, run=run, phuong_phap=ten,
                    nhom='tai nguyen' if ft in RESOURCE else 'mang',
                    tang_chon=la_chon, don_vi=dv,
                    # THU HANG THAT, de moi chung kiem soat ve sau tinh duoc ma
                    # KHONG phai goi lai phuong phap (xem docstring, phan CHUNG).
                    xh0='|'.join(xh0) if xh0 else '',
                    xh1='|'.join(xh1) if xh1 else '',
                    ung='|'.join(sorted(ung)),
                    fallback_0=fb0, fallback_1=fb1, giay_0=s0, giay_1=s1,
                    n_cand_0=len(xh0) if xh0 else np.nan,
                    n_cand_1=len(xh1) if xh1 else np.nan,
                    M0_nhu_xuat_ban=trung(xh0, 'node'),
                    M0b_khop_ung_vien=trung(xh0b, 'node'),
                    M1_truc_tang=trung(xh1, 'node'),
                    M2_ca_hai_truc=trung(xh1, dv),
                    M0_t3=trung(xh0, 'node', 3), M2_t3=trung(xh1, dv, 3)))
            if n_ca % 10 == 0:
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f'  [{sysname}] {n_ca} ca xong, da ghi {len(rows)} dong', flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(out_path, index=False)
    print(f'\n{n_ca} ca | {len(D)} dong -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
