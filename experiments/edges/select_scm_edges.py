# -*- coding: utf-8 -*-
"""
TIEN TINH TOAN canh SCM (Tier 2.5 "backpressure") TU DONG, TU LOGS
======================================================================
Chay 4 pha cua src/scm/scm_edge_selector.py cho MOT he thong, dung CHINH
logic build DAG/gan co che nhan qua cua CapacityAgent.train_accurate_path
(Tier 1 + Tier 2 + gan mechanism CPU/Mem/Latency/Workload -- xem
src/agents/capacity_agent.py:389-543), roi ghi ket qua ra
src/graph/<system>_scm_edges.json.

Muc dich: thay THE HAI DANH SACH BACKPRESSURE_EDGES go tay trong
capacity_agent.py (Sock Shop 3 canh, Train Ticket ~50 canh, moi he mot
nguong khac nhau) bang MOT quy trinh tu dong duy nhat, chay lai duoc tren
BAT KY he thong nao chi can (a) graph JSON dung schema chung va (b) du lieu
telemetry co cot '<service>_workload'/'<service>_<metric>'.

Day la buoc TIEN TINH (giong cach graph topology duoc trich xuat 1 lan roi
luu JSON, khong tu suy lai moi lan agent khoi dong) -- vi pha 4 (OOD-safety)
can fit dowhy.gcm model, ton thoi gian hon mot lan doc file JSON.

Dung:
    python experiments/select_scm_edges.py --system sockshop
    python experiments/select_scm_edges.py --system trainticket
"""

import argparse
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))

from data_processor import load_multi_service_data, SERVICES, TRAINTICKET_SERVICES  # noqa: E402
from scm_edge_selector import select_scm_edges  # noqa: E402


class QueueingLatencyRegressor:
    """Placeholder toi thieu -- chi de fit duoc do thi day du giong
    CapacityAgent; ket qua latency khong duoc dung trong lua chon canh nay
    (chi CPU duoc dung, giong backpressure_edge_ood_safety_test.py goc)."""
    from sklearn.base import BaseEstimator, RegressorMixin


def make_build_model_fn(graph_path: str, services: list, metric: str = 'cpu'):
    """Tra ve build_model_fn(df_data, extra_edges) -> (model, df_sub), dung
    LAI dung logic Tier1+Tier2+mechanism-assignment cua
    CapacityAgent.train_accurate_path (xem src/agents/capacity_agent.py) --
    de pha 4 (OOD-safety) kiem chung dung tren mo hinh SE THUC SU duoc trien
    khai, khong phai mot ban rut gon lech hanh vi.
    """
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
            if caller_col in df_data.columns and callee_col in df_data.columns:
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--system', required=True, choices=['sockshop', 'trainticket'])
    ap.add_argument('--metric', default='cpu')
    ap.add_argument('--min-gain', type=float, default=0.01)
    ap.add_argument('--skip-ood-safety', action='store_true',
                     help='Chi chay pha 1-3 (nhanh), bo qua fit dowhy.gcm o pha 4.')
    args = ap.parse_args()

    if args.system == 'trainticket':
        graph_path = os.path.join(BASE_DIR, 'src', 'graph', 'trainticket_agent_graph.json')
        services = TRAINTICKET_SERVICES
        injection_services = ['ts-preserve-service', 'ts-travel-service']
        data_dir = None
    else:
        graph_path = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')
        services = SERVICES
        injection_services = ['front-end', 'orders']
        data_dir = None

    print(f"[select_scm_edges] He thong: {args.system} ({len(services)} service)")
    # KHONG dropna() tren toan bo df tho: he thong nhieu service (vd Train
    # Ticket ~68 node) co the co NaN rai rac o cot khong lien quan, xoa het
    # dong chi vi 1 cot lech -- dropna() phai lam TREN TUNG SUBSET cot can
    # dung (score_r2_gain/build_model_fn da tu lo phan nay), giong dung cach
    # CapacityAgent.train_accurate_path lam (df_sub = df_data[valid_nodes].dropna()).
    df_data = load_multi_service_data(data_dir, system_type=args.system)
    print(f"[select_scm_edges] Da nap telemetry: {len(df_data)} dong")

    build_model_fn = None if args.skip_ood_safety else make_build_model_fn(graph_path, services, args.metric)

    result = select_scm_edges(
        graph_path, df_data, services, metric=args.metric, min_gain=args.min_gain,
        build_model_fn=build_model_fn, injection_services=injection_services,
    )

    print(f"\n[select_scm_edges] Candidate: {len(result['candidate_edges'])} canh")
    print(f"[select_scm_edges] Sau R2-gain + knee-point: {len(result['selected_edges'])} canh")
    for c, callee in result['selected_edges']:
        print(f"    - {c} -> {callee}")

    if result['ood_safety'] is not None:
        n_dropped = len(result['selected_edges']) - len(result['final_edges'])
        print(f"[select_scm_edges] Sau OOD-safety gate: {len(result['final_edges'])} canh "
              f"({n_dropped} canh bi loai vi gay sign-inversion moi)")

    out_path = os.path.join(BASE_DIR, 'src', 'graph', f'{args.system}_scm_edges.json')
    out_data = {
        'system': args.system,
        'metric': args.metric,
        'min_gain': args.min_gain,
        'ood_safety_checked': result['ood_safety'] is not None,
        'final_edges': [list(e) for e in result['final_edges']],
        'scores': result['scores'].to_dict(orient='records') if not result['scores'].empty else [],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] -> {out_path}")

    scores_csv = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results', f'scm_edge_selection_{args.system}.csv')
    if not result['scores'].empty:
        result['scores'].to_csv(scores_csv, index=False)
        print(f"[OK] -> {scores_csv}")


if __name__ == '__main__':
    main()
