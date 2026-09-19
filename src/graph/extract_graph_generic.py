# -*- coding: utf-8 -*-
"""
GENERIC LOG/TRACE-BASED GRAPH EXTRACTOR (system-agnostic)
===========================================================
Xay do thi phu thuoc G tu span-level tracing (schema chung:
spanID, parentSpanID, serviceName -- dung format voi RCAEval/Jaeger/
OpenTelemetry export) MA KHONG ma cung ten dich vu nao. Moi phan loai
(gateway / database-ha tang / backend) va buoc pha cycle deu suy THUAN TUY
tu cau truc do thi, nen dung duoc cho bat ky he thong nao co du lieu tracing
cung schema -- khong rieng Sock Shop hay Train Ticket.

Doi chieu paper (Table "General Specification", sec:general-spec):
  - Dependency graph G <- tracing/service mesh: "yes, if available"
        -> hien thuc o day (build_weighted_edges).
  - Gateway set = {v : in-deg(v)=0}, minus infra: "fully automatic"
        -> classify_nodes() gan type='gateway' cho dung nhom node nay;
           src/agents/parser_agent.py::_get_gateways() ap dung LAI cung
           quy tac o runtime (doc truong 'type' nay) -- hai noi nhat quan.
  - Known services = V(G) minus infra: "fully automatic"
        -> danh sach `nodes` tra ve, loc type != 'database' o noi tieu thu.

So voi src/graph/extract_trainticket_graph.py (script cu, GIU LAI cho lich
su/doi chieu): script do ma cung GATEWAYS=['ts-ui-dashboard'],
DB_SUFFIXES=('-mongo','-mysql'), mot danh sach 15 entrypoint_services, va
mot luat pha cycle gan voi ten canh cu the ('ts-seat-service'). Khong cai
nao trong so do chuyen sang he thong khac duoc. Script nay thay the toan bo
phan do bang suy luan cau truc (in-degree/out-degree/tan suat canh), roi
xuat cung schema JSON {"nodes":[{"id","type","description"}],
"edges":[{"source","target","relation","call_count"}]} de tuong thich nguoc
voi ArchitectureAgent va taxonomy_builder.py hien co.

Dung:
    python src/graph/extract_graph_generic.py \\
        --traces path/to/traces.parquet --system TenHeThong \\
        --out src/graph/tenhethong_agent_graph.json
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd
import networkx as nx


def load_traces(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_weighted_edges(df_traces: pd.DataFrame) -> Counter:
    """Tra ve Counter[(caller, callee)] = so span quan sat duoc cho canh do.

    Chi dung 3 cot chung cho moi he thong dung schema tracing chuan
    (spanID/parentSpanID/serviceName) -- khong doc bat ky cot dac thu he
    thong nao khac, nen ham nay TU DONG tong quat.
    """
    required = {'spanID', 'parentSpanID', 'serviceName'}
    missing = required - set(df_traces.columns)
    if missing:
        raise ValueError(
            f"Traces thieu cot bat buoc: {missing} "
            f"(can schema chung spanID/parentSpanID/serviceName)")

    parents = df_traces[['spanID', 'serviceName']].rename(
        columns={'spanID': 'parentSpanID', 'serviceName': 'parentService'})
    merged = df_traces[['parentSpanID', 'serviceName']].dropna().merge(parents, on='parentSpanID')
    cross_service = merged[merged['parentService'] != merged['serviceName']]
    return Counter(zip(cross_service['parentService'], cross_service['serviceName']))


def break_cycles(weighted_edges: Counter):
    """Pha moi cycle bang cach xoa canh IT DUOC GOI NHAT trong cycle do.

    Heuristic du lieu-dan-dat (khong phu thuoc ten dich vu nao): canh xuat
    hien it nhat trong trace la canh it duoc ho tro boi bang chung nhat, nen
    la ung vien hop ly nhat de loai bo khi can pha vong lap de co DAG. Tie-
    break bang ten canh de deterministic (chay lai ra ket qua giong het).

    Tra ve (nx.DiGraph da la DAG, list cac canh da xoa kem weight).
    """
    dg = nx.DiGraph()
    for (u, v), w in weighted_edges.items():
        dg.add_edge(u, v, weight=w)

    removed = []
    while not nx.is_directed_acyclic_graph(dg):
        cycle = nx.find_cycle(dg, orientation='original')
        weakest = min(cycle, key=lambda e: (dg[e[0]][e[1]]['weight'], e[0], e[1]))
        u, v = weakest[0], weakest[1]
        removed.append((u, v, dg[u][v]['weight']))
        dg.remove_edge(u, v)

    return dg, removed


def classify_nodes(dg: nx.DiGraph) -> dict:
    """Suy loai node THUAN TUY tu cau truc do thi (khong doan ten):

      - sink   (out-degree=0, in-degree>0)  -> 'database' (ha tang/luu tru:
                                                khong dich vu nao goi ra ngoai
                                                minh ma chi bi goi den)
      - source (in-degree=0)                -> 'gateway'  (ung vien diem
                                                vao; co the co NHIEU gateway,
                                                _get_gateways()/
                                                derive_primary_gateway() o
                                                runtime se loc/chon tiep)
      - con lai                             -> 'backend'

    Node co ca in-degree=0 VA out-degree=0 (khong lien ket voi ai) duoc xep
    'gateway' (nhanh source thang truoc trong if/elif) vi no khong the la
    sink hop le (khong ai goi den) -- trong thuc te cac node co dinh se bi
    loai o buoc doc traces (chi giu node co it nhat 1 canh).
    """
    nodes = {}
    for n in dg.nodes():
        indeg, outdeg = dg.in_degree(n), dg.out_degree(n)
        if indeg == 0:
            nodes[n] = 'gateway'
        elif outdeg == 0:
            nodes[n] = 'database'
        else:
            nodes[n] = 'backend'
    return nodes


def build_graph_json(dg: nx.DiGraph, node_types: dict, system_name: str) -> dict:
    nodes = []
    for n in sorted(dg.nodes()):
        t = node_types[n]
        desc = {
            'database': f'Ha tang/luu tru suy tu cau truc (sink node, chi nhan goi): {n}',
            'gateway': f'Ung vien diem vao suy tu cau truc (in-degree=0): {n}',
            'backend': f'Microservice noi bo: {n}',
        }[t]
        nodes.append({'id': n, 'type': t, 'description': desc})

    edges = []
    for u, v, data in sorted(dg.edges(data=True)):
        if node_types[v] == 'database':
            rel = 'DB_QUERY'
        elif node_types[u] == 'gateway':
            rel = 'HTTP_GATEWAY'
        else:
            rel = 'REST_CALL'
        edges.append({'source': u, 'target': v, 'relation': rel, 'call_count': int(data['weight'])})

    return {
        'system': system_name,
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'nodes': nodes,
        'edges': edges,
    }


def extract_graph(traces_path: str, system_name: str, out_path: str) -> dict:
    df = load_traces(traces_path)
    weighted = build_weighted_edges(df)
    if not weighted:
        raise ValueError("Khong trich duoc canh nao tu traces -- kiem tra schema/du lieu dau vao.")

    dg, removed = break_cycles(weighted)
    if removed:
        print(f"[extract_graph_generic] Da xoa {len(removed)} canh de pha cycle "
              f"(canh it duoc goi nhat trong tung cycle):")
        for u, v, w in removed:
            print(f"    - {u} -> {v} (call_count={w})")

    node_types = classify_nodes(dg)
    graph_data = build_graph_json(dg, node_types, system_name)

    gateways = sorted(n for n, t in node_types.items() if t == 'gateway')
    databases = sorted(n for n, t in node_types.items() if t == 'database')
    print(f"[extract_graph_generic] {system_name}: {len(dg.nodes())} node, {len(dg.edges())} canh")
    print(f"  Gateway ({len(gateways)}): {gateways}")
    print(f"  Database/infra ({len(databases)}): {databases}")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] -> {out_path}")
    return graph_data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traces', required=True,
                     help='File traces (.parquet/.csv) voi cot spanID, parentSpanID, serviceName')
    ap.add_argument('--system', required=True, help='Ten he thong (vd: SockShop, TrainTicket, <he thong moi>)')
    ap.add_argument('--out', required=True, help='Duong dan JSON dau ra (vd: src/graph/<system>_agent_graph.json)')
    args = ap.parse_args()
    extract_graph(args.traces, args.system, args.out)


if __name__ == '__main__':
    main()
