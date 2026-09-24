# -*- coding: utf-8 -*-
"""
Lan truyen Tier-1 tren Alibaba: do thi co gia tri gi hon gia dinh delta deu?
=============================================================================
Day la phien ban Alibaba cua RQ4, nhung chay tren du lieu CO tin hieu va CO
day du chot an toan -- khac ban goc tren Sock Shop, noi n=6 service va khoang
tin cay bootstrap chua 0.

Cau hoi tach roi khoi cau hoi ve co che: khi mot requirement lam tang tai o
service THUONG NGUON, tai cua service HA NGUON thay doi the nao? Hai cau tra
loi canh tranh:

  delta deu (naive) : W_C thay doi CUNG TI LE % voi W_P
                      W_C_pred = mean(W_C_train) * (W_P_test / mean(W_P_train))
  lan truyen do thi : W_C = f(W_P) voi f fit tu du lieu

Topology lay tu MSCallGraph (cot um -> dm). Chi giu rpctype la loi goi giua
SERVICE ('rpc', 'http'); bo 'db', 'mq', 'mc' (ha tang) va 'userDefined'.

Chot an toan (giong alibaba_forecast_eval):
  - baseline hang so lam moc skill
  - negative control: hoan vi W_P -> skill phai sup ve ~0
  - tach in-distribution va OOD
  - bao cao phan phoi per-edge, khong chi trung vi

Output: data/processed/scm_results/alibaba_tier1_propagation.csv
"""

import os
import sys
import glob
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
sys.path.insert(0, os.path.dirname(__file__))
from alibaba_signal_check import load_mcr, DATA_DIR   # noqa: E402

SERVICE_RPC = ('rpc', 'http')      # loi goi giua service; bo db/mq/mc/userDefined
MIN_EDGE_CALLS = 5000              # edge phai du pho bien moi dang tin
MIN_POINTS = 60
N_FIT = 2000
SEED = 42
RNG = np.random.RandomState(SEED)
CHUNK = 3_000_000


def build_topology():
    """Edge list (um -> dm) tu MSCallGraph, chi giu loi goi giua service."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'MSCallGraph_*.csv')))
    if not files:
        return None
    counts = {}
    for f in files:
        for ch in pd.read_csv(f, usecols=['um', 'rpctype', 'dm'],
                              chunksize=CHUNK,
                              dtype={'um': 'str', 'rpctype': 'category',
                                     'dm': 'str'}):
            ch = ch[ch.rpctype.isin(SERVICE_RPC)].dropna(subset=['um', 'dm'])
            ch = ch[ch.um != ch.dm]
            vc = ch.groupby(['um', 'dm']).size()
            for k, v in vc.items():
                counts[k] = counts.get(k, 0) + int(v)
    edges = pd.DataFrame(
        [{'parent': p, 'child': c, 'n_calls': n} for (p, c), n in counts.items()])
    return edges[edges.n_calls >= MIN_EDGE_CALLS].sort_values(
        'n_calls', ascending=False).reset_index(drop=True)


def skill(y, pred, base):
    mm = float(np.mean((y - pred) ** 2))
    mb = float(np.mean((y - base) ** 2))
    return np.nan if mb <= 1e-15 else 1.0 - mm / mb


def mape(y, pred):
    m = (y != 0) & np.isfinite(y) & np.isfinite(pred)
    return float(np.mean(np.abs((y[m] - pred[m]) / y[m])) * 100) if m.sum() else np.nan


def _split(g, mode):
    if mode == 'ood':
        g = g.sort_values('wl_parent')
        k = int(len(g) * 0.67)
        return g.iloc[:k], g.iloc[k:]
    idx = RNG.permutation(len(g))
    k = int(len(g) * 0.7)
    return g.iloc[idx[:k]], g.iloc[idx[k:]]


def eval_edge(g, mode, shuffled):
    tr, te = _split(g, mode)
    if len(tr) < 30 or len(te) < 15:
        return []
    if len(tr) > N_FIT:
        tr = tr.sample(N_FIT, random_state=SEED)
    Xtr, Xte = tr[['wl_parent']].values, te[['wl_parent']].values
    ytr, yte = tr.wl_child.values, te.wl_child.values
    if np.std(Xtr) < 1e-12 or np.std(ytr) < 1e-12:
        return []

    const = np.full(len(yte), float(np.mean(ytr)))

    # delta deu: ti le % cua parent ap y nguyen sang child
    mp = float(np.mean(Xtr))
    uniform = (np.full(len(yte), float(np.mean(ytr))) *
               (Xte.ravel() / mp)) if mp > 1e-12 else const

    out = []
    base = dict(mode=mode, shuffled=shuffled, n_train=len(tr), n_test=len(te))
    out.append(dict(base, model='constant', skill=0.0, mape=mape(yte, const)))
    out.append(dict(base, model='uniform_delta',
                    skill=skill(yte, uniform, const), mape=mape(yte, uniform)))
    for name, m in (('graph_LinearReg', LinearRegression()),
                    ('graph_NNLS', LinearRegression(positive=True))):
        try:
            p = m.fit(Xtr, ytr).predict(Xte)
        except Exception:
            continue
        out.append(dict(base, model=name, skill=skill(yte, p, const),
                        mape=mape(yte, p)))
    return out


def run():
    print("=== building topology from MSCallGraph ===")
    edges = build_topology()
    if edges is None or edges.empty:
        print("  khong dung duoc topology")
        return pd.DataFrame()
    print(f"  {len(edges):,} edge (>= {MIN_EDGE_CALLS} calls) | "
          f"{edges.parent.nunique():,} parent, {edges.child.nunique():,} child")

    print("=== loading workload ===")
    wl = load_mcr()
    w = wl.rename(columns={'msname': 'svc'})
    print(f"  {len(w):,} dong, {w.svc.nunique():,} service")

    rows = []
    for shuffled in (False, True):
        for _, e in edges.iterrows():
            p = w[w.svc == e.parent][['timestamp', 'workload']].rename(
                columns={'workload': 'wl_parent'})
            c = w[w.svc == e.child][['timestamp', 'workload']].rename(
                columns={'workload': 'wl_child'})
            if len(p) < MIN_POINTS or len(c) < MIN_POINTS:
                continue
            g = p.merge(c, on='timestamp', how='inner')
            g = g[(g.wl_parent > 0) & (g.wl_child > 0)]
            if len(g) < MIN_POINTS:
                continue
            if shuffled:
                g = g.copy()
                g['wl_parent'] = RNG.permutation(g.wl_parent.values)
            for mode in ('indist', 'ood'):
                for r in eval_edge(g, mode, shuffled):
                    r.update(parent=e.parent[:12], child=e.child[:12],
                             n_calls=int(e.n_calls))
                    rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT_DIR, 'alibaba_tier1_propagation.csv'), index=False)
    return d


def report(d):
    if d.empty:
        print("khong co du lieu")
        return
    d = d.replace([np.inf, -np.inf], np.nan)
    n_edges = d[['parent', 'child']].drop_duplicates().shape[0]
    print(f"\n{'='*76}\n  LAN TRUYEN TIER-1  ({n_edges} edge)\n{'='*76}")
    print(f"  {'model':18s} {'mode':7s} {'shuf':6s} {'skill_med':>10s} "
          f"{'%>0':>7s} {'MAPE_med':>9s}")
    for mode in ('indist', 'ood'):
        for shuf in (False, True):
            for m in ('constant', 'uniform_delta', 'graph_LinearReg', 'graph_NNLS'):
                x = d[(d['mode'] == mode) & (d.shuffled == shuf) & (d.model == m)]
                if x.empty:
                    continue
                print(f"  {m:18s} {mode:7s} {str(shuf):6s} {x.skill.median():10.4f} "
                      f"{100*(x.skill > 0).mean():6.1f}% {x.mape.median():9.2f}")

    print(f"\n{'='*76}\n  DO THI vs DELTA DEU (chi tren shuf=False)\n{'='*76}")
    for mode in ('indist', 'ood'):
        s = d[(d['mode'] == mode) & (~d.shuffled)]
        piv = s.pivot_table(index=['parent', 'child'], columns='model',
                            values='skill')
        if 'uniform_delta' not in piv or 'graph_NNLS' not in piv:
            continue
        piv = piv.dropna(subset=['uniform_delta', 'graph_NNLS'])
        win = int((piv.graph_NNLS > piv.uniform_delta).sum())
        print(f"  {mode:7s} n={len(piv):3d} | graph_NNLS thang delta deu tren "
              f"{win}/{len(piv)} edge ({100*win/max(len(piv),1):.1f}%)")
        print(f"          skill trung vi: graph={piv.graph_NNLS.median():+.4f} "
              f"uniform={piv.uniform_delta.median():+.4f}")


if __name__ == '__main__':
    report(run())
