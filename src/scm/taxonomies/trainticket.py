# -*- coding: utf-8 -*-
"""Taxonomy hieu chinh cho Train Ticket (RCAEval / FudanSELab).

Muc dich: kiem chung tuyen bo "khung nay tong quat hoa duoc sang he thong
khac" bang cach THUC SU trien khai ParserAgent tren he thu hai, thay vi chi
tham so hoa co che roi de ngo.

Cach dung theo dung "General Specification" trong paper:

  * `services` cua moi archetype KHONG viet tay. Chung duoc suy ra tu do thi
    phu thuoc that (`src/graph/trainticket_agent_graph.json`) bang cach lay
    closure huu han hop (bounded BFS) tu cac service "hat giong" dac trung
    cho archetype do -- dung phep ma bang General Specification danh dau la
    ban tu dong. Phan BFS/gateway-derivation nay nam trong
    `src/scm/taxonomy_builder.py` (system-agnostic), KHONG viet rieng o day
    nua -- xem module do de biet chi tiet thuat toan.
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
14 node type=='gateway' (in-degree=0, tru ha tang). Entry point that cua
nguoi dung cuoi la `ts-ui-dashboard` (out-degree 15, cao nhat do thi) --
dung `taxonomy_builder.derive_primary_gateway()` de suy ra gia tri nay tu
CHINH do thi thay vi ghi co dinh, va gan lai vao hang so ben duoi CHI DE
kiem chung/tai lieu hoa (assert bang nhau, xem duoi). Cac node type=='gateway'
con lai la dich vu quan tri/ha tang (`ts-admin-*`, `ts-auth-service`,
`ts-voucher-service`, ...) KHONG phai duong vao cua yeu cau khach hang. Dieu
nay lam cho bai toan "gateway disambiguation" -- thu paper truoc day xep vao
muc chua kiem chung duoc vi topology chi co mot gateway -- tro nen kiem
chung duoc.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from taxonomy_builder import load_graph, derive_primary_gateway, build_call_chains  # noqa: E402

GRAPH_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'graph', 'trainticket_agent_graph.json'))

# Entry point that su cho yeu cau khach hang (xem docstring ve 14 gateway).
# Ghi co dinh (khong goi derive_primary_gateway() truc tiep vao CALL_CHAINS)
# de mot lan trich xuat lai graph bi loi khong am tham doi gateway; module
# nay TU KIEM (xem __main__) rang gia tri suy ra tu do thi khop hang so nay.
TRAINTICKET_GATEWAY = 'ts-ui-dashboard'


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
    """Sinh bang CALL_CHAINS cho Train Ticket tu do thi phu thuoc that.

    Uy quyen toan bo cho taxonomy_builder.build_call_chains() (system-agnostic)
    -- Train Ticket chi cung graph path + TRAINTICKET_GATEWAY + _ARCHETYPE_SPEC
    viet tay (phan "assisted"/"not automatable" theo General Specification).
    """
    return build_call_chains(GRAPH_PATH, _ARCHETYPE_SPEC, gateway=TRAINTICKET_GATEWAY)


TRAINTICKET_CALL_CHAINS = build_trainticket_call_chains()

__all__ = ['TRAINTICKET_CALL_CHAINS', 'build_trainticket_call_chains',
           'TRAINTICKET_GATEWAY']

if __name__ == '__main__':
    # Tu kiem: TRAINTICKET_GATEWAY (hang so ghi co dinh o tren) phai khop
    # gateway suy tu CHINH do thi -- neu lech, graph da doi (vd trich xuat
    # lai) va hang so can duoc cap nhat thu cong.
    _adj, _node_types = load_graph(GRAPH_PATH)
    _derived = derive_primary_gateway(_adj, _node_types)
    assert _derived == TRAINTICKET_GATEWAY, (
        f"TRAINTICKET_GATEWAY='{TRAINTICKET_GATEWAY}' khong khop gateway suy tu "
        f"do thi hien tai ('{_derived}') -- graph co the da doi, kiem tra lai.")
    print(f"[OK] TRAINTICKET_GATEWAY khop gateway suy tu do thi: {_derived}")
    for _name, _chain in TRAINTICKET_CALL_CHAINS.items():
        print(f"  {_name}: {_chain['services']}")
