# -*- coding: utf-8 -*-
"""
AN TOAN NGOAI SUY: canh backpressure moi them vao capacity_agent.py::train_accurate_path
co lam TANG so ca sign-inversion (delta AM khi workload TANG) o extreme delta
(+150%/+300%) khong, so voi Global DAG cu (khong co canh nay)?

Dung lai chinh xac cach xay dung graph cua capacity_agent.py::train_accurate_path,
chi bat/tat 3 canh backpressure de so sanh CU vs MOI, quet delta tu +5% den +300%
tren nhieu injection_service, kiem tra dau (sign) cua *_cpu_change_pct cho tat ca
service — dac biet shipping/carts/user (noi vua them canh).

Output: data/processed/scm_results/backpressure_edge_ood_safety.csv
"""

import os
import sys
import json
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data, SERVICES

GRAPH_PATH = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')
BACKPRESSURE_EDGES = [('orders_cpu', 'shipping_cpu'), ('orders_cpu', 'carts_cpu'), ('front-end_cpu', 'user_cpu')]
DELTAS = [5, 20, 50, 100, 150, 300]
INJECTION_SERVICES = ['front-end', 'orders']
N_PROJ = 200


class QueueingLatencyRegressor:
    """Ban rut gon chi de fit duoc do thi day du (khong dung ket qua latency o day)."""
    from sklearn.base import BaseEstimator, RegressorMixin


def build_model(df_data, with_backpressure: bool):
    with open(GRAPH_PATH, 'r', encoding='utf-8') as f:
        graph_json = json.load(f)
    g = nx.DiGraph()
    for edge in graph_json.get('edges', []):
        src, tgt = edge['source'], edge['target']
        if src in SERVICES and tgt in SERVICES:
            g.add_edge(f"{src}_workload", f"{tgt}_workload")
    for s in SERVICES:
        for metric_col in [f'{s}_cpu', f'{s}_mem']:  # bo latency cho gon (khong lien quan test nay)
            if f'{s}_workload' in df_data.columns and metric_col in df_data.columns:
                g.add_edge(f"{s}_workload", metric_col)
    if with_backpressure:
        for a, b in BACKPRESSURE_EDGES:
            if a in df_data.columns and b in df_data.columns:
                g.add_edge(a, b)

    valid_nodes = [n for n in g.nodes() if n in df_data.columns]
    g_sub = g.subgraph(valid_nodes).copy()
    df_sub = df_data[valid_nodes].dropna()

    df_fit = df_sub.sample(min(2000, len(df_sub)), random_state=42) if len(df_sub) > 2000 else df_sub
    model = gcm.InvertibleStructuralCausalModel(g_sub)
    gcm.auto.assign_causal_mechanisms(model, df_fit)
    for node in g_sub.nodes():
        if node.endswith('_cpu') or node.endswith('_mem'):
            model.set_causal_mechanism(node, AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))
        elif node.endswith('_workload') and g_sub.in_degree(node) > 0:
            model.set_causal_mechanism(node, AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))
    gcm.fit(model, df_fit)
    return model, df_sub


def sweep(model, df_sub, injection_service, tag):
    rows = []
    inj_col = f"{injection_service}_workload"
    base_wl = float(df_sub[inj_col].mean())
    for delta in DELTAS:
        target_wl = base_wl * (1 + delta / 100)
        samples = gcm.interventional_samples(
            model, interventions={inj_col: lambda x, w=target_wl: w}, num_samples_to_draw=N_PROJ)
        for svc in SERVICES:
            col = f"{svc}_cpu"
            if col not in df_sub.columns or col not in samples.columns:
                continue
            base_v = float(df_sub[col].mean())
            pred_v = float(samples[col].mean())
            chg = (pred_v - base_v) / abs(base_v) * 100 if base_v != 0 else 0.0
            rows.append({'variant': tag, 'injection_service': injection_service, 'delta_pct': delta,
                         'service': svc, 'cpu_change_pct': round(chg, 2), 'sign_inverted': chg < 0})
    return rows


if __name__ == '__main__':
    df_data = load_multi_service_data(None, system_type='sockshop')

    all_rows = []
    for with_bp, tag in [(False, 'OLD_no_backpressure'), (True, 'NEW_with_backpressure')]:
        print(f"\n=== Fit {tag} ===")
        model, df_sub = build_model(df_data, with_backpressure=with_bp)
        for inj in INJECTION_SERVICES:
            print(f"  Sweep injection={inj} ...")
            all_rows += sweep(model, df_sub, inj, tag)

    out = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, 'backpressure_edge_ood_safety.csv')
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu: {out_path}")

    print("\n" + "=" * 80)
    print("  SO SANH SO CA SIGN-INVERSION (delta AM khi workload TANG), theo delta_pct")
    print("=" * 80)
    piv = out.groupby(['variant', 'delta_pct'])['sign_inverted'].sum().unstack('variant')
    print(piv)

    print("\nChi tiet cac ca sign-inverted (neu co):")
    inv = out[out['sign_inverted']]
    if inv.empty:
        print("  KHONG CO ca sign-inversion nao trong ca 2 phien ban (OLD va NEW), tren moi delta/injection/service da quet.")
    else:
        print(inv.to_string(index=False))

    print("\nRieng shipping/carts/user (noi vua them canh backpressure), o delta +150%/+300%:")
    focus = out[(out['service'].isin(['shipping', 'carts', 'user'])) & (out['delta_pct'].isin([150, 300]))]
    print(focus.pivot_table(index=['injection_service', 'service', 'delta_pct'], columns='variant', values='cpu_change_pct'))
