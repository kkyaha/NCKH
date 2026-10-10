# -*- coding: utf-8 -*-
"""
CHON CANH SCM (Tier 2.5) TREN DU LIEU SS-TRAIN (testbed tu dung cua nguoi
dung), KHONG phai RCAEval -- de kiem xem quy trinh chon canh (R2-gain +
knee-point + OOD-safety) co chon dung 3 canh da biet khong khi chay tren
MOT nguon du lieu khac, co kiem soat, co can thiep that (ramp).

Day la BAN SAO CO CHU DICH cua
papers/p1_du_phong/experiments/edges/select_scm_edges.py, voi hai doi khac
DUY NHAT:
  1. data_dir = 'data/raw/SS-TRAIN' thay vi None (mac dinh RCAEval RE2-SS).
  2. Ghi ra data/processed/scm_results/sockshop_scm_edges_from_sstrain.json
     -- KHONG GHI DE src/graph/sockshop_scm_edges.json (file production,
     dang duoc CapacityAgent doc cho moi ket qua RQ1-4 da cong bo tren
     RCAEval). Hai nguon du lieu khac nhau phai cho ra hai file khac nhau.

Dung:
    python experiments/chua_phan_loai/select_scm_edges_sstrain.py
"""

import json
import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import networkx as nx
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))

from data_processor import load_multi_service_data, SERVICES  # noqa: E402
from scm_edge_selector import select_scm_edges  # noqa: E402


class QueueingLatencyRegressor:
    """Placeholder toi thieu -- giong ban goc, chi de fit duoc do thi day du."""
    from sklearn.base import BaseEstimator, RegressorMixin


def make_build_model_fn(graph_path: str, services: list, metric: str = 'cpu'):
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_json = json.load(f)

    def build_model_fn(df_data, extra_edges):
        g = nx.DiGraph()
        for edge in graph_json.get('edges', []):
            src, tgt = edge['source'], edge['target']
            if src in services and tgt in services:
                g.add_edge(f'{src}_workload', f'{tgt}_workload')
        for s in services:
            for metric_col in [f'{s}_cpu', f'{s}_mem']:
                if f'{s}_workload' in df_data.columns and metric_col in df_data.columns:
                    g.add_edge(f'{s}_workload', metric_col)
        for caller_col, callee_col in extra_edges:
            if caller_col in g.nodes() and callee_col in g.nodes():
                g.add_edge(caller_col, callee_col)

        valid_nodes = [n for n in g.nodes() if n in df_data.columns]
        g_sub = g.subgraph(valid_nodes).copy()
        df_sub = df_data[valid_nodes].dropna()

        while not nx.is_directed_acyclic_graph(g_sub):
            cycle = nx.find_cycle(g_sub, orientation='original')
            g_sub.remove_edge(cycle[-1][0], cycle[-1][1])

        df_fit = df_sub.sample(min(2000, len(df_sub)), random_state=42) if len(df_sub) > 2000 else df_sub

        model = gcm.InvertibleStructuralCausalModel(g_sub)
        gcm.auto.assign_causal_mechanisms(model, df_fit)
        for node in g_sub.nodes():
            if node.endswith('_cpu') or node.endswith('_mem'):
                model.set_causal_mechanism(
                    node, AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))
            elif node.endswith('_workload') and g_sub.in_degree(node) > 0:
                model.set_causal_mechanism(
                    node, AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))
        gcm.fit(model, df_fit)
        return model, df_sub

    return build_model_fn


def main():
    graph_path = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')
    services = SERVICES
    injection_services = ['front-end', 'orders']
    data_dir = os.path.join(BASE_DIR, 'data', 'raw', 'SS-TRAIN')  # KHAC BAN GOC: SS-TRAIN, khong phai RCAEval
    metric = 'cpu'
    min_gain = 0.01

    print(f"[select_scm_edges_sstrain] Nguon du lieu: {data_dir}")
    df_data = load_multi_service_data(data_dir, system_type='sockshop')
    print(f"[select_scm_edges_sstrain] Da nap telemetry: {len(df_data)} dong")

    build_model_fn = make_build_model_fn(graph_path, services, metric)

    result = select_scm_edges(
        graph_path, df_data, services, metric=metric, min_gain=min_gain,
        build_model_fn=build_model_fn, injection_services=injection_services,
    )

    print(f"\n[select_scm_edges_sstrain] Candidate: {len(result['candidate_edges'])} canh")
    print(f"[select_scm_edges_sstrain] Sau R2-gain + knee-point: {len(result['selected_edges'])} canh")
    for c, callee in result['selected_edges']:
        print(f"    - {c} -> {callee}")

    if result['ood_safety'] is not None:
        n_dropped = len(result['selected_edges']) - len(result['final_edges'])
        print(f"[select_scm_edges_sstrain] Sau OOD-safety gate: {len(result['final_edges'])} canh "
              f"({n_dropped} canh bi loai vi gay sign-inversion moi)")

    out_path = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results',
                             'sockshop_scm_edges_from_sstrain.json')
    out_data = {
        'system': 'sockshop',
        'source_data': 'data/raw/SS-TRAIN (KHONG phai RCAEval)',
        'metric': metric,
        'min_gain': min_gain,
        'ood_safety_checked': result['ood_safety'] is not None,
        'final_edges': [list(e) for e in result['final_edges']],
        'scores': result['scores'].to_dict(orient='records') if not result['scores'].empty else [],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] -> {out_path}")

    scores_csv = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results',
                               'scm_edge_selection_sockshop_from_sstrain.csv')
    if not result['scores'].empty:
        result['scores'].to_csv(scores_csv, index=False)
        print(f"[OK] -> {scores_csv}")


if __name__ == '__main__':
    main()
