# -*- coding: utf-8 -*-
"""
!!! THI NGHIEM NAY KHONG HOP LE -- GIU LAI DE GHI NHAN, KHONG DUOC TRICH DAN !!!
================================================================================
Hai loi da chan doan duoc sau khi chay:

1. TIEN DE SAI. Fault injection cua RCAEval KHONG phai mot thay doi TAI. Do
   workload tai service bi tiem: median -0.9%, va 62.2% scenario GIAM tai
   (fault 'loss': -46.4%). Fault lam SUY GIAM service, khong nap tai vao no.
   Nen script dua vao mo hinh mot tin hieu can thiep ~1% roi so voi hau qua cua
   mot thu hoan toan khac (CPU stress, memory pressure tac dong truc tiep).

2. KIEM SOAT AM HONG. Baseline 'random' hoan vi chinh vector du doan, ma vector
   do gan nhu hang so (vi Delta-w ~ 1%), nen khong sinh ra thu hang ngau nhien
   deu. Mo phong dung cho median Spearman = +0.0286 voi n=6; thi nghiem quan sat
   -0.1014. Khi kiem soat am cung lech thi khong dong nao doc duoc.

Ket luan: KHONG phai ket qua am ve framework. La thi nghiem duoc thiet ke sai.

Y tuong goc (dung can thiep da biet lam ground truth cho ATTRIBUTION, theo logic
Budhathoki et al. ICML 2022) VAN HOP LE, nhung can mot can thiep LEN TAI -- tuc
mot load sweep tu chay -- chu khong phai fault injection. RCAEval can thiep o
tang CO CHE, con do thi cua framework can thiep o tang WORKLOAD: sai tang.

Day cung la ly do THU BA, doc lap, khien RCAEval khong hop voi bai toan nay:
  (1) khong co bien thien tai      -> R^2 = 0.003
  (2) khong co confounding         -> effect 0.498 (production: 0.873)
  (3) can thiep sai tang           -> Delta-w median -0.9%
"""


import os
import sys
import json
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
GRAPH = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
from rq7_interventional_validity import _iter_runs, SERVICES, MIN_PRE, MIN_POST

SEED = 42
RNG = np.random.RandomState(SEED)


def load_graph():
    g = json.load(open(GRAPH))
    G = nx.DiGraph()
    for n in g['nodes']:
        G.add_node(n['id'])
    for e in g['edges']:
        G.add_edge(e['source'], e['target'])
    return G


def rel_change(post, pre):
    return (post - pre) / pre if abs(pre) > 1e-9 else np.nan


def run():
    G = load_graph()
    und = G.to_undirected()
    rows = []

    for scenario, target, fault, run_id, df, it in _iter_runs():
        pre, post = df[df['time'] < it], df[df['time'] >= it]
        if len(pre) < MIN_PRE or len(post) < MIN_POST:
            continue
        wt = f'{target}_workload'
        if wt not in df.columns:
            continue

        for metric in ('cpu', 'mem'):
            others = [s for s in SERVICES if s != target
                      and f'{s}_workload' in df.columns
                      and f'{s}_{metric}' in df.columns]
            if len(others) < 4:
                continue

            # gia tri can thiep: workload THUC TE cua service dich sau inject
            w0 = float(pre[wt].mean())
            w1 = float(post[wt].mean())
            if not (np.isfinite(w0) and np.isfinite(w1)) or w0 <= 1e-9:
                continue

            pred, actual = {}, {}
            for s in others:
                ws, rs = f'{s}_workload', f'{s}_{metric}'
                sp = pre[[wt, ws, rs]].dropna()
                so = post[[ws, rs]].dropna()
                if len(sp) < MIN_PRE or len(so) < MIN_POST:
                    continue
                # Tier-1: W_s ~ W_target   |   Tier-2: R_s ~ W_s
                t1 = LinearRegression(positive=True).fit(sp[[wt]].values, sp[ws].values)
                t2 = LinearRegression(positive=True).fit(sp[[ws]].values, sp[rs].values)
                r_base = float(t2.predict(t1.predict(np.array([[w0]])).reshape(-1, 1))[0])
                r_new = float(t2.predict(t1.predict(np.array([[w1]])).reshape(-1, 1))[0])
                pred[s] = rel_change(r_new, r_base)
                actual[s] = rel_change(float(so[rs].mean()), float(sp[rs].mean()))

            common = [s for s in others
                      if s in pred and s in actual
                      and np.isfinite(pred[s]) and np.isfinite(actual[s])]
            if len(common) < 4:
                continue

            p = np.array([pred[s] for s in common])
            a = np.array([actual[s] for s in common])
            # baseline: gan hon -> anh huong nhieu hon (nen lay am cua hop)
            hop = np.array([-nx.shortest_path_length(und, target, s)
                            if nx.has_path(und, target, s) else -99
                            for s in common], dtype=float)
            rnd = RNG.permutation(p)

            def sp_corr(x):
                if np.std(x) < 1e-12 or np.std(a) < 1e-12:
                    return np.nan
                return float(stats.spearmanr(x, a).correlation)

            # precision@2: trong 2 service bi anh huong manh nhat thuc te,
            # bao nhieu nam trong top-2 du doan
            k = 2
            top_a = set(np.argsort(-a)[:k])
            prec = {name: len(top_a & set(np.argsort(-x)[:k])) / k
                    for name, x in (('model', p), ('hop', hop), ('random', rnd))}

            rows.append(dict(
                scenario=scenario, target=target, fault=fault, run_id=run_id,
                metric=metric, n_services=len(common),
                delta_w_pct=100 * (w1 - w0) / w0,
                rho_model=sp_corr(p), rho_hop=sp_corr(hop), rho_random=sp_corr(rnd),
                prec_model=prec['model'], prec_hop=prec['hop'],
                prec_random=prec['random']))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'rq12_prospective_attribution.csv'), index=False)
    return out


def report(d):
    if d.empty:
        print("khong co du lieu")
        return
    d = d.replace([np.inf, -np.inf], np.nan)
    print(f"n = {len(d)} (scenario x run x metric), "
          f"{d.n_services.median():.0f} service duoc xep hang moi lan\n")

    print("=" * 74)
    print("  Tuong quan THU HANG Spearman voi thay doi THUC DO")
    print("=" * 74)
    print(f"  {'predictor':14s} {'median rho':>11s} {'mean':>8s} {'%rho>0':>8s} "
          f"{'prec@2':>8s}")
    for name in ('model', 'hop', 'random'):
        r = d[f'rho_{name}'].dropna()
        pr = d[f'prec_{name}'].dropna()
        if r.empty:
            continue
        print(f"  {name:14s} {r.median():11.4f} {r.mean():8.4f} "
              f"{100*(r > 0).mean():7.1f}% {pr.mean():8.3f}")

    print("\n" + "=" * 74)
    print("  Kiem dinh")
    print("=" * 74)
    for a, b in (('model', 'random'), ('model', 'hop')):
        x = (d[f'rho_{a}'] - d[f'rho_{b}']).dropna()
        if len(x) < 10:
            continue
        st, p = stats.wilcoxon(x)
        med = float(np.median(x))
        win = float((d[f'rho_{a}'] > d[f'rho_{b}']).mean())
        verdict = ('TOT HON' if p < 0.05 and med > 0 and win >= 0.6 else
                   'TE HON' if p < 0.05 and med < 0 and win <= 0.4 else
                   'khong khac biet dang ke')
        print(f"  {a} vs {b:8s}: median diff={med:+.4f}  p={p:.3g}  "
              f"win={100*win:.1f}%  -> {verdict}")

    print("\n  Theo metric:")
    print(d.groupby('metric')[['rho_model', 'rho_hop', 'rho_random']]
          .median().round(4).to_string())
    print("\n  Theo loai fault (rho_model trung vi):")
    print(d.groupby('fault')['rho_model'].median().round(4).to_string())


if __name__ == '__main__':
    report(run())
