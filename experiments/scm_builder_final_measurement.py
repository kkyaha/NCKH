# -*- coding: utf-8 -*-
"""
DO LUONG CHOT: engine xay SCM tong quat (template + held-out + cache)
========================================================================
Chay MOT luot tren ca 2 he thong, bao cao:
  1. So canh moi template (candidate -> held-out/knee-point -> OOD-safety)
  2. Phan bo do tin cay tung loai node (_node_confidence)
  3. MAPE/R2 held-out trung binh theo nhom metric (evaluate_node_stability)
  4. Smoke test cac API chinh
  5. Thoi gian train (cache nguoi/nong)

Dung: python experiments/scm_builder_final_measurement.py
"""

import os
import sys
import time
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))

import pandas as pd
from capacity_agent import CapacityAgent  # noqa: E402


def suffix_of(node: str) -> str:
    return node.rsplit('_', 1)[-1]


def measure(system: str) -> dict:
    print('=' * 72)
    print(f'  HE THONG: {system}')
    print('=' * 72)
    t0 = time.time()
    agent = CapacityAgent(system_type=system, auto_train=True)
    train_time = time.time() - t0

    eb = agent._last_edge_build
    print(f'\n[1] Canh theo template (from_cache={eb.get("from_cache")}):')
    for label, rep in eb['reports'].items():
        print(f'    {label:<28} {rep["n_candidates"]:>3} cand -> {rep["n_selected"]:>3} '
              f'held-out/knee -> {rep["n_final"]:>3} final')
    print(f'    Tong: {len(eb["structural_edges"])} cau truc + {len(eb["learned_edges"])} hoc duoc '
          f'= {agent.get_dag_summary()["n_edges"]} canh / {agent.get_dag_summary()["n_nodes"]} node')

    print('\n[2] Do tin cay node theo loai metric:')
    by_suffix = {}
    for n in agent.dag_graph.nodes():
        by_suffix.setdefault(suffix_of(n), []).append(n)
    for suf in sorted(by_suffix):
        c = Counter(agent._node_confidence(n) for n in by_suffix[suf])
        print(f'    {suf:<12} {dict(c)}')

    print('\n[3] Do chinh xac held-out (evaluate_node_stability) theo loai metric:')
    stab = agent.evaluate_node_stability()
    if not stab.empty:
        stab['suffix'] = stab['node'].map(suffix_of)
        agg = stab.groupby('suffix').agg(n=('node', 'count'),
                                          mape_mean=('mape_pct', 'mean'),
                                          mape_median=('mape_pct', 'median'),
                                          r2_median=('r2', 'median'))
        print(agg.round(3).to_string())

    print('\n[4] Smoke test:')
    env = agent.certified_envelope(agent.default_injection)
    sim = agent.simulate_intervention(agent.default_injection, 25.0)
    sel = agent.select_key_nodes()
    m = agent.get_metrics_for_service(agent.services[0], workload_delta_pct=20.0)
    print(f'    certified_envelope={len(env)} keys, simulate_intervention={bool(sim)}, '
          f'get_metrics={bool(m)}')
    print(f'    select_key_nodes recommended={sel["recommended"]}')
    print(f'    unstable_but_impactful={sel["unstable_but_impactful"]}')

    t1 = time.time()
    CapacityAgent(system_type=system, auto_train=True)
    warm_time = time.time() - t1
    print(f'\n[5] Train time: cold/cache-hit ban dau {train_time:.1f}s, lan sau {warm_time:.1f}s')
    return {'system': system, 'train_time': train_time, 'warm_time': warm_time,
            'stability': stab}


def main():
    pd.set_option('display.width', 160)
    results = [measure('sockshop'), measure('trainticket')]
    print('\n' + '=' * 72)
    print('  TONG KET')
    print('=' * 72)
    for r in results:
        s = r['stability']
        if s.empty:
            continue
        s = s.copy()
        s['suffix'] = s['node'].map(suffix_of)
        lat = s[s['suffix'] == 'latency-50']
        cpu = s[s['suffix'] == 'cpu']
        print(f"{r['system']:<12} latency MAPE median={lat['mape_pct'].median():.2f}% "
              f"(n={len(lat)}, POOR={int((lat['tag']=='POOR').sum())}) | "
              f"cpu MAPE median={cpu['mape_pct'].median():.2f}% "
              f"(n={len(cpu)}, POOR={int((cpu['tag']=='POOR').sum())}) | "
              f"train {r['train_time']:.0f}s -> {r['warm_time']:.0f}s")


if __name__ == '__main__':
    main()
