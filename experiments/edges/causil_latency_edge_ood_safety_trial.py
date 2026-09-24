# -*- coding: utf-8 -*-
"""
CAUSIL LATENCY EDGE -- PHA 4: OOD-SAFETY VALIDATION
============================================================
Tiep noi causil_latency_edge_trial.py (da xac nhan kha thi qua BIC + held-out
R2/MAPE): buoc nay ap dung PHA 4 cua scm_edge_selector.py (quet do() qua
nhieu delta, dem sign-inversion) len CHINH Global DAG that (qua CapacityAgent),
khong phai mo hinh 2-node rieng le -- vi da co bang chung cu the (SockShop
orders->payment) rang BIC/R2 mot minh co the chon nham canh lam hai kha nang
ngoai suy, dung y CausIL/paper deu KHONG the bo qua buoc nay.

Dung deterministic_forward() (khong Monte Carlo) cho ca 2 bien the (baseline
= DAG hien co, extended = DAG + candidate edges CausIL da qua BIC) -- tat
dinh, nhanh, dung dinh huong toi uu phep do() da thong nhat truoc do trong
phien nay.

Dung: python experiments/causil_latency_edge_ood_safety_trial.py --system trainticket
      python experiments/causil_latency_edge_ood_safety_trial.py --system sockshop
"""

import argparse
import copy
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.linear_model import LinearRegression

from capacity_agent import CapacityAgent, QueueingLatencyRegressor  # noqa: E402
from deterministic_forward import deterministic_forward  # noqa: E402
from causil_latency_edge_trial import candidate_latency_edges, evaluate_candidate  # noqa: E402


def _assign_mechanism(model, g: nx.DiGraph, node: str):
    """Dung LAI y het quy tac train_accurate_path() dung (cpu/mem -> Linear
    positive, latency-50 -> QueueingLatencyRegressor, workload CO CHA ->
    Linear positive) -- KHONG doi quy tac, chi ap dung lai cho DAG da them
    canh CausIL. Root workload (in-degree=0, vd gateway) GIU nguyen mechanism
    stochastic ma gcm.auto.assign_causal_mechanisms() da gan -- khong duoc
    ghi de bang AdditiveNoiseModel(Linear), se vi pham rang buoc cua dowhy
    (root node can StochasticModel, khong phai AdditiveNoiseModel)."""
    if node.endswith('_cpu') or node.endswith('_mem'):
        model.set_causal_mechanism(node, AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))
    elif node.endswith('_latency-50'):
        model.set_causal_mechanism(node, AdditiveNoiseModel(SklearnRegressionModel(QueueingLatencyRegressor())))
    elif node.endswith('_workload') and g.in_degree(node) > 0:
        model.set_causal_mechanism(node, AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))


def build_extended_model(agent: CapacityAgent, extra_edges: list):
    """Sao chep dag_graph cua agent, them extra_edges (callee_latency ->
    caller_latency), pha cycle neu co, gan mechanism DUNG quy tac hien co,
    fit tren CUNG baseline df ma agent that da dung."""
    g = agent.dag_graph.copy()
    for src, tgt in extra_edges:
        g.add_edge(src, tgt)

    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g, orientation='original')
        g.remove_edge(cycle[-1][0], cycle[-1][1])

    df_fit = agent.global_df_baseline
    model = gcm.InvertibleStructuralCausalModel(g)
    gcm.auto.assign_causal_mechanisms(model, df_fit)
    for node in g.nodes():
        _assign_mechanism(model, g, node)
    gcm.fit(model, df_fit)
    return g, model


def sweep_sign_inversions(dag_graph, model, baseline_df, injection_nodes: list,
                           deltas=(5, 20, 50, 100, 150, 300), min_abs_change_pct: float = 1.0):
    """Quet do(injection_node = base*(1+delta%)) qua nhieu delta, dem
    sign-inversion tren MOI node latency trong DAG -- dung deterministic_
    forward() (tat dinh, khong Monte Carlo)."""
    rows = []
    lat_nodes = [n for n in dag_graph.nodes() if n.endswith('_latency-50')]
    for inj in injection_nodes:
        if inj not in dag_graph.nodes():
            continue
        base_wl = float(baseline_df[inj].mean())
        for delta in deltas:
            target_wl = base_wl * (1 + delta / 100)
            vals, breached = deterministic_forward(dag_graph, model, baseline_df, {inj: target_wl})
            for node in lat_nodes:
                if node == inj:
                    continue
                base_v = float(baseline_df[node].mean())
                pred_v = float('inf') if node in breached else vals.get(node, base_v)
                if not np.isfinite(pred_v):
                    chg = float('inf')
                else:
                    chg = (pred_v - base_v) / abs(base_v) * 100.0 if base_v != 0 else 0.0
                rows.append({
                    'injection': inj, 'delta_pct': delta, 'node': node,
                    'change_pct': chg, 'sign_inverted': (chg < -abs(min_abs_change_pct)),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', choices=['sockshop', 'trainticket'], default='trainticket')
    ap.add_argument('--bic-only', action='store_true', help='Chi dung candidate BIC chon giu (mac dinh: tat ca candidate)')
    args = ap.parse_args()

    print(f"[1] Train CapacityAgent({args.system}) -- DAG baseline (chua co canh CausIL).")
    agent = CapacityAgent(system_type=args.system, auto_train=True)

    graph_path = os.path.join(BASE_DIR, 'src', 'graph', f'{"trainticket" if args.system=="trainticket" else "sockshop"}_agent_graph.json')
    candidates = candidate_latency_edges(graph_path, agent.global_df_baseline.columns)
    print(f"[2] {len(candidates)} candidate L^B->L^A tu graph phu thuoc.")

    bic_rows = []
    for caller, callee in candidates:
        r = evaluate_candidate(agent.global_df_baseline, caller, callee)
        if r is not None:
            bic_rows.append(r)
    bic_df = pd.DataFrame(bic_rows)
    selected = bic_df[bic_df['bic_selects_edge']] if args.bic_only else bic_df
    extra_edges = [(f"{r.callee}_latency-50", f"{r.caller}_latency-50") for r in selected.itertuples()]
    print(f"[3] {len(extra_edges)} canh dua vao DAG mo rong "
          f"({'chi BIC-selected' if args.bic_only else 'toan bo candidate'}).")

    print("[4] Fit lai Global DAG voi canh moi...")
    g_ext, model_ext = build_extended_model(agent, extra_edges)

    injection_nodes = [f"{s}_workload" for s in ([agent.default_injection] if isinstance(agent.default_injection, str) else agent.default_injection)]
    print(f"[5] Quet sign-inversion, injection={injection_nodes}, baseline vs extended...")

    safety_base = sweep_sign_inversions(agent.dag_graph, agent.global_dag_model, agent.global_df_baseline, injection_nodes)
    safety_ext = sweep_sign_inversions(g_ext, model_ext, agent.global_df_baseline, injection_nodes)

    n_inv_base = int(safety_base['sign_inverted'].sum())
    n_inv_ext = int(safety_ext['sign_inverted'].sum())
    print(f"\n[6] TOM TAT AN TOAN OOD (tren {len(safety_base)} phep do moi ben):")
    print(f"    Sign-inversion BASELINE (chua co canh CausIL): {n_inv_base}/{len(safety_base)}")
    print(f"    Sign-inversion EXTENDED (co canh CausIL):      {n_inv_ext}/{len(safety_ext)}")

    # So khop key (injection, delta, node) giua 2 ban -- tim inversion MOI
    key_cols = ['injection', 'delta_pct', 'node']
    base_inv = set(map(tuple, safety_base[safety_base['sign_inverted']][key_cols].values))
    ext_inv = set(map(tuple, safety_ext[safety_ext['sign_inverted']][key_cols].values))
    new_inversions = ext_inv - base_inv
    fixed_inversions = base_inv - ext_inv
    print(f"    Inversion MOI xuat hien (do canh CausIL gay ra): {len(new_inversions)}")
    print(f"    Inversion CU duoc sua (canh CausIL lam het):     {len(fixed_inversions)}")
    if new_inversions:
        print(f"    Chi tiet inversion moi (toi da 15): {sorted(new_inversions)[:15]}")

    if not args.bic_only:
        unsafe_nodes = set(n for _, _, n in new_inversions)
        safe_edges = [(cl, ca) for cl, ca in extra_edges if ca.rsplit('_', 1)[0] + '_latency-50' not in unsafe_nodes]
        print(f"\n[7] Neu loai canh gay inversion moi: con lai {len(safe_edges)}/{len(extra_edges)} canh an toan.")


if __name__ == '__main__':
    main()
