# -*- coding: utf-8 -*-
"""Taxonomy hieu chinh cho Train Ticket (RCAEval / FudanSELab).

Muc dich: kiem chung tuyen bo "khung nay tong quat hoa duoc sang he thong
khac" bang cach THUC SU trien khai ParserAgent tren he thu hai, thay vi chi
tham so hoa co che roi de ngo.

Cach dung theo dung "General Specification" trong paper:

  * `services` cua moi archetype KHONG viet tay. Chung duoc suy ra tu do thi
    phu thuoc that (`src/graph/trainticket_agent_graph.json`) bang cach lay
    closure huu han hop (bounded BFS) tu gateway qua cac service "hat giong"
    dac trung cho archetype do -- dung phep "simple paths from gateway"
    ma bang General Specification danh dau la ban tu dong.
  * `keywords` va `description` viet tay tu tai lieu Train Ticket -- dung
    hang ma bang General Specification danh dau la "Assisted, needs human
    review". Day la phan KHONG tu dong hoa duoc, va chung toi khong gia vo
    nguoc lai.
  * `expected_delta_pct` la NEO GIA DINH, khong phai do tu telemetry thuc
    cua tung archetype. Bang General Specification xep hang nay vao "Not
    automatable". Vi vay moi ket qua Anchor-MAE tren Train Ticket chi do
    tinh NHAT QUAN cua parser voi bang nay, KHONG do tinh dung vat ly.
    Phai neu ro dieu nay o bat ky cho nao bao cao so lieu Train Ticket.

Gateway: khac SockShop (mot gateway `front-end` duy nhat), Train Ticket co
14 node in-degree=0. Entry point that cua nguoi dung cuoi la `ts-ui-dashboard`
(out-degree 15, cao nhat do thi). Cac node in-degree=0 con lai la dich vu
quan tri/ha tang (`ts-admin-*`, `ts-auth-service`, `ts-voucher-service`, ...)
KHONG phai duong vao cua yeu cau khach hang. Dieu nay lam cho bai toan
"gateway disambiguation" -- thu paper truoc day xep vao muc chua kiem chung
duoc vi topology chi co mot gateway -- tro nen kiem chung duoc.
"""

import json
import os
from collections import deque

GRAPH_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'graph', 'trainticket_agent_graph.json'))

# Entry point that su cho yeu cau khach hang (xem docstring ve 14 gateway).
TRAINTICKET_GATEWAY = 'ts-ui-dashboard'

# Ha tang, khong phai service ung dung -> loai khoi blast radius.
_INFRA_SUFFIXES = ('-mongo', '-mysql', '-redis')


def _load_graph():
    with open(GRAPH_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    adj = {n['id']: [] for n in data['nodes']}
    for e in data['edges']:
        if e['source'] in adj:
            adj[e['source']].append(e['target'])
    return adj


def _is_infra(node: str) -> bool:
    return node.endswith(_INFRA_SUFFIXES)


def _closure(adj, seeds, max_hops: int):
    """BFS huu han hop tu CAC SEED cua archetype, bo node ha tang.

    QUAN TRONG: BFS bat dau tu seed, KHONG tu gateway. Gateway
    `ts-ui-dashboard` co out-degree 15, nen mo rong tu no chi 2 hop la cham
    gan het do thi (28-30/68 node) va moi archetype se co blast radius gan
    nhu giong het nhau -- dung loi ma paper da neu: mot tap node chung khong
    phan biet duoc hai archetype khac y dinh nguoi dung. Gateway chi duoc
    GHEP vao dau danh sach vi moi yeu cau khach hang deu di qua no.

    Tra ve danh sach theo thu tu on dinh (gateway truoc, roi alphabet).
    """
    seen = set()
    q = deque()
    for s in seeds:
        if s in adj and s != TRAINTICKET_GATEWAY:
            q.append((s, 0))
            seen.add(s)
    while q:
        node, hop = q.popleft()
        if hop >= max_hops:
            continue
        for nxt in adj.get(node, []):
            if nxt not in seen and not _is_infra(nxt) and nxt != TRAINTICKET_GATEWAY:
                seen.add(nxt)
                q.append((nxt, hop + 1))
    return [TRAINTICKET_GATEWAY] + sorted(seen)


# Phan VIET TAY: seed service, mo ta, tu khoa, neo gia dinh.
# (Bang General Specification: hai hang cuoi khong tu dong hoa duoc.)
_ARCHETYPE_SPEC = {
    'SEARCH_TRAIN': {
        'seeds': ['ts-travel-service'],
        'hops': 1,
        'description': 'Customer searches for trains between two stations on a date',
        'keywords': ['search', 'tim', 'tra cuu', 'chuyen tau', 'train', 'timetable',
                     'lich trinh', 'tuyen', 'route', 'ga', 'station'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
    'QUERY_TICKET_PRICE': {
        'seeds': ['ts-basic-service'],
        'hops': 1,
        'description': 'Customer checks ticket price and seat availability',
        'keywords': ['price', 'gia', 've', 'ticket', 'cost', 'fare', 'bao nhieu', 'seat', 'cho ngoi'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
    'BOOK_TICKET': {
        'seeds': ['ts-preserve-service'],
        'hops': 1,
        'description': 'Customer books/reserves a train ticket (heaviest customer operation)',
        'keywords': ['book', 'dat ve', 'reserve', 'preserve', 'booking', 'mua ve',
                     'dat cho', 'purchase ticket'],
        'resource_profile': 'cpu-heavy',
        'expected_delta_pct': 25,
    },
    'PAY_ORDER': {
        'seeds': ['ts-inside-payment-service'],
        'hops': 1,
        'description': 'Customer pays for a booked order',
        'keywords': ['pay', 'payment', 'thanh toan', 'tra tien', 'checkout', 'money'],
        'resource_profile': 'cpu-heavy',
        'expected_delta_pct': 20,
    },
    'CANCEL_ORDER': {
        'seeds': ['ts-cancel-service'],
        'hops': 1,
        'description': 'Customer cancels an existing order and requests refund',
        'keywords': ['cancel', 'huy', 'refund', 'hoan tien', 'tra ve', 'huy ve'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 15,
    },
    'REBOOK_TICKET': {
        'seeds': ['ts-rebook-service', 'ts-travel-service'],
        'hops': 1,
        'description': 'Customer changes an existing booking to another train',
        'keywords': ['rebook', 'doi ve', 'change ticket', 'reschedule', 'doi chuyen'],
        'resource_profile': 'cpu-heavy',
        'expected_delta_pct': 20,
    },
    'ORDER_FOOD': {
        'seeds': ['ts-food-service'],
        'hops': 1,
        'description': 'Customer orders food to be delivered on board',
        'keywords': ['food', 'do an', 'meal', 'suat an', 'dat com', 'delivery'],
        'resource_profile': 'cpu-memory',
        'expected_delta_pct': 15,
    },
    'CONSIGN_LUGGAGE': {
        'seeds': ['ts-consign-service'],
        'hops': 1,
        'description': 'Customer consigns luggage for a trip',
        'keywords': ['consign', 'luggage', 'hanh ly', 'gui do', 'baggage'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
    'QUERY_ORDER': {
        'seeds': ['ts-order-service'],
        'hops': 1,
        'description': 'Customer views their existing orders and trip status',
        'keywords': ['order', 'don hang', 'my ticket', 'lich su', 'history', 'status', 'tra cuu don'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
    'USER_LOGIN': {
        'seeds': ['ts-user-service'],
        'hops': 1,
        'description': 'Customer logs in or registers an account',
        'keywords': ['login', 'dang nhap', 'register', 'dang ky', 'signup', 'account', 'tai khoan'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
}


def build_trainticket_call_chains() -> dict:
    """Sinh bang CALL_CHAINS cho Train Ticket tu do thi phu thuoc that."""
    adj = _load_graph()
    out = {}
    for name, spec in _ARCHETYPE_SPEC.items():
        services = _closure(adj, spec['seeds'], spec['hops'])
        out[name] = {
            'services': services,
            'description': spec['description'],
            'keywords': spec['keywords'],
            'resource_profile': spec['resource_profile'],
            'expected_delta_pct': spec['expected_delta_pct'],
        }
    return out


TRAINTICKET_CALL_CHAINS = build_trainticket_call_chains()

__all__ = ['TRAINTICKET_CALL_CHAINS', 'build_trainticket_call_chains',
           'TRAINTICKET_GATEWAY']
