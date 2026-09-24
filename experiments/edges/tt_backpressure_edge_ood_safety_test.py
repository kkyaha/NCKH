# -*- coding: utf-8 -*-
"""
TRAIN TICKET: an toan ngoai suy khi them 50 canh backpressure (gain>0.03) vao
Global DAG that. Dung DUNG phuong phap backpressure_edge_ood_safety_test.py
(Sock Shop) -- so sanh OLD (khong co canh) vs NEW (co 50 canh), quet delta,
dem so ca sign-inversion.
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
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data, TRAINTICKET_SERVICES

GRAPH_PATH = os.path.join(BASE_DIR, 'src', 'graph', 'trainticket_agent_graph.json')
DELTAS = [100, 150, 300]
INJECTION_SERVICES = ['ts-ui-dashboard', 'ts-preserve-service']
N_PROJ = 150
GAIN_THRESHOLD = 0.03


def build_model(df_data, backpressure_edges):
    with open(GRAPH_PATH, 'r', encoding='utf-8') as f:
        graph_json = json.load(f)
    svc_set = set(TRAINTICKET_SERVICES)
    g = nx.DiGraph()
    for edge in graph_json.get('edges', []):
        src, tgt = edge['source'], edge['target']
        if src in svc_set and tgt in svc_set:
            g.add_edge(f"{src}_workload", f"{tgt}_workload")
    for s in svc_set:
        for metric_col in [f'{s}_cpu', f'{s}_mem']:
            if f'{s}_workload' in df_data.columns and metric_col in df_data.columns:
                g.add_edge(f"{s}_workload", metric_col)
    for caller, callee in backpressure_edges:
        a, b = f'{caller}_cpu', f'{callee}_cpu'
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
    if inj_col not in df_sub.columns:
        return rows
    base_wl = float(df_sub[inj_col].mean())
    for delta in DELTAS:
        target_wl = base_wl * (1 + delta / 100)
        samples = gcm.interventional_samples(
            model, interventions={inj_col: lambda x, w=target_wl: w}, num_samples_to_draw=N_PROJ)
        for s in TRAINTICKET_SERVICES:
            col = f"{s}_cpu"
            if col not in df_sub.columns or col not in samples.columns:
                continue
            base_v = float(df_sub[col].mean())
            pred_v = float(samples[col].mean())
            chg = (pred_v - base_v) / abs(base_v) * 100 if base_v != 0 else 0.0
            rows.append({'variant': tag, 'injection_service': injection_service, 'delta_pct': delta,
                         'service': s, 'cpu_change_pct': round(chg, 2), 'sign_inverted': chg < 0})
    return rows


if __name__ == '__main__':
    diag = pd.read_csv(os.path.join(OUT_DIR, 'tt_call_chain_neighbor_diagnostic.csv'))
    strong_edges = list(diag[diag['gain'] > GAIN_THRESHOLD][['caller', 'callee']].itertuples(index=False, name=None))
    print(f"So canh backpressure se them: {len(strong_edges)}")

    df_data = load_multi_service_data(os.path.join(BASE_DIR, 'data', 'raw', 'trainticket'), system_type='trainticket')

    all_rows = []
    for edges, tag in [([], 'OLD_no_backpressure'), (strong_edges, 'NEW_with_backpressure')]:
        print(f"\n=== Fit {tag} ===")
        model, df_sub = build_model(df_data, edges)
        print(f"  nodes={model.graph.number_of_nodes()} edges={model.graph.number_of_edges()}")
        for inj in INJECTION_SERVICES:
            print(f"  Sweep injection={inj} ...")
            all_rows += sweep(model, df_sub, inj, tag)

    out = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, 'tt_backpressure_edge_ood_safety.csv')
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu: {out_path}")

    print("\n" + "=" * 70)
    print("  SO SANH SO CA SIGN-INVERSION theo delta_pct")
    print("=" * 70)
    piv = out.groupby(['variant', 'delta_pct'])['sign_inverted'].sum().unstack('variant')
    print(piv)
    print(f"\nTong: OLD={out[out.variant=='OLD_no_backpressure']['sign_inverted'].sum()}  "
          f"NEW={out[out.variant=='NEW_with_backpressure']['sign_inverted'].sum()}  "
          f"(tren {len(out[out.variant=='OLD_no_backpressure'])} to hop moi ben)")
