# -*- coding: utf-8 -*-
"""
EXTRACT TRAIN TICKET ARCHITECTURE GRAPH
=======================================
Extracts nodes (gateways, backends, databases) and directed edges
(HTTP / RPC relations) from Train Ticket telemetry & traces,
producing src/graph/trainticket_agent_graph.json.
"""

import os
import json
import pandas as pd
import networkx as nx

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SAMPLE_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'trainticket', 'sample')
OUT_FILE = os.path.join(BASE_DIR, 'src', 'graph', 'trainticket_agent_graph.json')

# Standard database and gateway mappings in Train Ticket architecture
DB_SUFFIXES = ['-mongo', '-mysql']
GATEWAYS = ['ts-ui-dashboard']


def extract_graph():
    metrics_path = os.path.join(SAMPLE_DIR, 'metrics.parquet')
    traces_path = os.path.join(SAMPLE_DIR, 'traces.parquet')

    if not os.path.exists(metrics_path) or not os.path.exists(traces_path):
        raise FileNotFoundError(f"Missing sample files in {SAMPLE_DIR}")

    df_metrics = pd.read_parquet(metrics_path)
    df_traces = pd.read_parquet(traces_path)

    # 1. Collect all services from metrics columns
    services = set()
    for col in df_metrics.columns:
        if col == 'time':
            continue
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            services.add(parts[0])

    # 2. Extract caller -> callee edges from traces (vectorized join)
    df_parents = df_traces[['spanID', 'serviceName']].rename(
        columns={'spanID': 'parentSpanID', 'serviceName': 'parentService'}
    )
    merged = df_traces[['parentSpanID', 'serviceName']].dropna().merge(df_parents, on='parentSpanID')
    cross_service = merged[merged['parentService'] != merged['serviceName']][
        ['parentService', 'serviceName']
    ].drop_duplicates()

    edges_set = set(zip(cross_service['parentService'], cross_service['serviceName']))

    # 3. Add ts-ui-dashboard as the top-level frontend gateway
    services.add('ts-ui-dashboard')

    # Gateway entrypoint edges to primary user-facing services
    entrypoint_services = [
        'ts-preserve-service',
        'ts-preserve-other-service',
        'ts-travel-service',
        'ts-travel2-service',
        'ts-order-service',
        'ts-order-other-service',
        'ts-inside-payment-service',
        'ts-user-service',
        'ts-admin-basic-info-service',
        'ts-admin-travel-service',
        'ts-cancel-service',
        'ts-execute-service',
        'ts-contacts-service',
        'ts-food-service',
        'ts-consign-service'
    ]
    for ep in entrypoint_services:
        if ep in services:
            edges_set.add(('ts-ui-dashboard', ep))

    # 4. Connect database nodes to their corresponding backend services
    for s in list(services):
        for db_suffix in DB_SUFFIXES:
            if s.endswith(db_suffix):
                svc_name = s.replace(db_suffix, '-service')
                if svc_name in services:
                    edges_set.add((svc_name, s))

    # 4b. Break cycles to guarantee a strict Causal DAG (Directed Acyclic Graph)
    dg = nx.DiGraph()
    for u, v in edges_set:
        dg.add_edge(u, v)

    while not nx.is_directed_acyclic_graph(dg):
        cycle = nx.find_cycle(dg, orientation='original')
        # Prefer breaking upstream back-edges (e.g. ts-seat-service -> ts-travel-service)
        removed = False
        for edge in cycle:
            u, v = edge[0], edge[1]
            if u == 'ts-seat-service' and 'ts-travel' in v:
                dg.remove_edge(u, v)
                edges_set.discard((u, v))
                removed = True
                break
        if not removed:
            u, v = cycle[-1][0], cycle[-1][1]
            dg.remove_edge(u, v)
            edges_set.discard((u, v))

    print(f"  Verified DAG: nx.is_directed_acyclic_graph = {nx.is_directed_acyclic_graph(dg)}")

    # 5. Build nodes metadata
    nodes = []
    for s in sorted(services):
        if s in GATEWAYS or s == 'ts-ui-dashboard':
            node_type = 'gateway'
            desc = 'Web UI & API Gateway tiếp nhận yêu cầu từ người dùng'
        elif any(s.endswith(sfx) for sfx in DB_SUFFIXES):
            node_type = 'database'
            desc = f'Cơ sở dữ liệu của {s}'
        else:
            node_type = 'backend'
            desc = f'Microservice {s} (Java/Spring Boot)'
        nodes.append({'id': s, 'type': node_type, 'description': desc})

    # 6. Build edges metadata
    edges = []
    for u, v in sorted(edges_set):
        if any(v.endswith(sfx) for sfx in DB_SUFFIXES):
            rel = 'DB_QUERY'
        elif u == 'ts-ui-dashboard':
            rel = 'HTTP_GATEWAY'
        else:
            rel = 'REST_CALL'
        edges.append({'source': u, 'target': v, 'relation': rel})

    graph_data = {
        'system': 'TrainTicket',
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'nodes': nodes,
        'edges': edges
    }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Generated Train Ticket Graph: {len(nodes)} nodes, {len(edges)} edges -> {OUT_FILE}")
    return graph_data


if __name__ == '__main__':
    extract_graph()
