# -*- coding: utf-8 -*-
"""
TAXONOMY BUILDER TONG QUAT (system-agnostic)
=============================================
Sinh CALL_CHAINS cho MOT HE THONG BAT KY tu (1) do thi phu thuoc da trich
xuat (vd boi src/graph/extract_graph_generic.py, hoac bat ky script nao xuat
dung schema {"nodes":[{"id","type",...}], "edges":[{"source","target",...}]})
va (2) mot dac ta archetype hand-authored toi thieu.

Theo dung phan chia trach nhiem trong paper (Table "General Specification",
sec:general-spec):
  - Dependency graph G                 -> tracing/service mesh, "yes if available"
  - Gateway set / known services       -> suy tu do thi, "fully automatic"
  - Archetype services (`services`)    -> bounded BFS tu seed, "semi-automatic"
    -> day la thu ham build_call_chains() nay lam.
  - keywords / description             -> "assisted" (LLM-drafted tu doc),
    VAN phai viet tay trong archetype_spec cua tung taxonomy.
  - expected_delta_pct                 -> "not automatable" (neo tu telemetry
    that), van viet tay.

Truoc khi co module nay, moi taxonomy he thong moi (vd trainticket.py) phai
tu viet lai _load_graph / _is_infra / _closure rieng -- moi ham do lai
thuong lai ma cung ten dich vu (vd DB_SUFFIXES = ('-mongo','-mysql')), nen
khong chuyen sang he thu ba duoc. Module nay tach phan THUAN TUY CAU TRUC
(load graph, suy gateway, BFS closure loai infra) ra dung MOT lan, dung
truong `type` da co san trong graph JSON (gateway/database/backend) thay vi
doan lai tu ten node.
"""

import json
from collections import deque


def load_graph(graph_json_path: str):
    """Doc graph JSON (schema {"nodes":[{"id","type"}], "edges":[{"source","target"}]}).

    Tra ve (adjacency: dict[str, list[str]], node_types: dict[str, str]).
    """
    with open(graph_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    adj = {n['id']: [] for n in data['nodes']}
    node_types = {n['id']: n['type'] for n in data['nodes']}
    for e in data['edges']:
        if e['source'] in adj:
            adj[e['source']].append(e['target'])
    return adj, node_types


def derive_primary_gateway(adj: dict, node_types: dict) -> str:
    """Chon MOT gateway 'chinh' khi taxonomy can mot gia tri default_gateway
    duy nhat de ghep vao dau moi call chain (vd Train Ticket co 14 node
    type=='gateway'/in-degree=0, nhung chi 1 la entry point nguoi dung that).

    Quy tac: trong cac node type=='gateway', lay node co out-degree lon nhat
    (entry point that su phuc vu nhieu request nhat). Hoa -> alphabet,
    deterministic. He mot-gateway (SockShop) tra ve dung node duy nhat do.
    """
    gateways = [n for n, t in node_types.items() if t == 'gateway']
    if not gateways:
        raise ValueError("Khong co node type='gateway' trong do thi -- kiem tra buoc trich xuat.")
    return sorted(gateways, key=lambda n: (-len(adj.get(n, [])), n))[0]


def bfs_closure(adj: dict, node_types: dict, seeds: list, max_hops: int,
                 gateway: str = None) -> list:
    """BFS huu han hop tu cac SEED cua mot archetype, bo qua node ha tang
    (type=='database') va node gateway (gateway duoc GHEP thu cong vao dau
    danh sach ket qua, vi moi request khach hang deu di qua no).

    Ham THUAN TUY CAU TRUC: khong doan ten dich vu nao, chi doc `type` va
    canh cua do thi -- nen dung duoc cho bat ky he thong nao xuat dung
    schema graph JSON, khong chi Train Ticket.

    seeds ngoai do thi bi bo qua lang le (cho phep archetype_spec viet tay
    hoi lech ten mot chut ma khong crash toan bo taxonomy).
    """
    seen = set()
    q = deque()
    for s in seeds:
        if s in adj and s != gateway:
            q.append((s, 0))
            seen.add(s)
    while q:
        node, hop = q.popleft()
        if hop >= max_hops:
            continue
        for nxt in adj.get(node, []):
            if (nxt not in seen and node_types.get(nxt) != 'database'
                    and nxt != gateway):
                seen.add(nxt)
                q.append((nxt, hop + 1))
    ordered = sorted(seen)
    return [gateway] + ordered if gateway else ordered


def build_call_chains(graph_json_path: str, archetype_spec: dict,
                       gateway: str = None) -> dict:
    """Sinh bang CALL_CHAINS day du cho MOT he thong tu graph + dac ta archetype.

    archetype_spec: {archetype_name: {
        'seeds': [...], 'hops': int,           # -> suy `services` (semi-automatic)
        'description': str, 'keywords': [...], # -> viet tay (assisted)
        'resource_profile': str,
        'expected_delta_pct': number,          # -> viet tay (not automatable)
    }}

    gateway: gia tri default_gateway. None -> tu suy bang derive_primary_gateway().

    Tra ve dict cung hinh dang SOCKSHOP_CALL_CHAINS / TRAINTICKET_CALL_CHAINS,
    nen la drop-in cho ParserAgent(call_chains=..., default_gateway=...).
    """
    adj, node_types = load_graph(graph_json_path)
    gw = gateway if gateway is not None else derive_primary_gateway(adj, node_types)

    out = {}
    for name, spec in archetype_spec.items():
        services = bfs_closure(adj, node_types, spec['seeds'], spec['hops'], gateway=gw)
        out[name] = {
            'services': services,
            'description': spec['description'],
            'keywords': spec['keywords'],
            'resource_profile': spec['resource_profile'],
            'expected_delta_pct': spec['expected_delta_pct'],
        }
    return out


__all__ = ['load_graph', 'derive_primary_gateway', 'bfs_closure', 'build_call_chains']
