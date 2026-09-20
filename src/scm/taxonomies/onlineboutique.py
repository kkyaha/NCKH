# -*- coding: utf-8 -*-
"""Taxonomy hieu chinh cho Online Boutique (RCAEval RE2-OB).

He thu 3 kiem chung "General Specification" (sau SockShop, Train Ticket).
services cua moi archetype suy tu do thi that (bounded BFS tu seed, xem
taxonomy_builder.py) -- KHONG viet tay.

PHAT HIEN THAT khi trich xuat do thi (src/graph/onlineboutique_agent_graph.json):
extract_graph_generic.py phan loai node bang heuristic THUAN CAU TRUC (sink
out-degree=0 => 'database'). Heuristic nay dung cho SockShop/Train Ticket (sink
la Mongo/MySQL that), nhung SAI cho Online Boutique: trace RE2-OB khong ghi
lai loi goi toi datastore that (vd redis-cart khong xuat hien trong span nao),
nen 4 service ung dung THAT (currencyservice/emailservice/paymentservice/
productcatalogservice) bi gan nham thanh 'database' -- neu khong sua, bfs_
closure() se AM THAM loai ca 4 node nay khoi moi archetype (no bo qua
type=='database'), lam sai blast-radius cho hau het tinh nang (vd CHECKOUT se
mat het currency/email/payment/catalog). Da sua tay truc tiep trong file JSON
(xem field 'description' cua 4 node do) truoc khi dung file nay -- ghi nhan
day la GIOI HAN THAT cua heuristic sink=>database, khong phai loi trien khai
rieng cua he nay.

Chi 7/11 service cua Online Boutique that (thieu cartservice, shippingservice,
adservice, redis-cart) xuat hien trong trace RE2-OB -- da kiem tren ca 5
scenario (1/root-cause-service) cho ket qua giong het nhau, nen day la pham
vi that cua bo du lieu (cac service do khong duoc instrument tracing), khong
phai thieu sot khi trich xuat.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from taxonomy_builder import load_graph, derive_primary_gateway, build_call_chains  # noqa: E402

GRAPH_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'graph', 'onlineboutique_agent_graph.json'))

# Entry point that duy nhat (in-degree=0, out-degree lon nhat -- Online
# Boutique chi co 1 gateway trong du lieu trace nay, khac Train Ticket).
ONLINEBOUTIQUE_GATEWAY = 'frontend'


# Phan VIET TAY: seed service, mo ta, tu khoa, neo gia dinh.
# (Bang General Specification: hai hang cuoi khong tu dong hoa duoc.)
_ARCHETYPE_SPEC = {
    'BROWSE_PRODUCTS': {
        'seeds': ['productcatalogservice'],
        'hops': 1,
        'description': 'Customer browses or views product catalogue',
        'keywords': ['browse', 'catalogue', 'catalog', 'product', 'xem', 'san pham', 'danh sach'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
    'GET_RECOMMENDATIONS': {
        'seeds': ['recommendationservice'],
        'hops': 1,
        'description': 'Customer sees personalised product recommendations (calls catalogue for details)',
        'keywords': ['recommend', 'goi y', 'suggestion', 'related products', 'phu hop'],
        'resource_profile': 'cpu-memory',
        'expected_delta_pct': 20,
    },
    'CONVERT_CURRENCY': {
        'seeds': ['currencyservice'],
        'hops': 1,
        'description': 'Customer views price converted to their local currency',
        'keywords': ['currency', 'tien te', 'ty gia', 'convert', 'quy doi', 'price', 'gia'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
    'CHECKOUT': {
        'seeds': ['checkoutservice'],
        'hops': 1,
        'description': 'Customer completes checkout (payment, email confirmation, currency conversion, catalogue lookup)',
        'keywords': ['checkout', 'thanh toan', 'dat hang', 'mua hang', 'order', 'buy', 'purchase'],
        'resource_profile': 'cpu-heavy',
        'expected_delta_pct': 25,
    },
}


def build_onlineboutique_call_chains() -> dict:
    """Sinh bang CALL_CHAINS cho Online Boutique tu do thi phu thuoc that.

    Uy quyen toan bo cho taxonomy_builder.build_call_chains() (system-agnostic)
    -- Online Boutique chi cung graph path + ONLINEBOUTIQUE_GATEWAY +
    _ARCHETYPE_SPEC viet tay (phan "assisted"/"not automatable" theo General
    Specification), giong het cach Train Ticket da lam.
    """
    return build_call_chains(GRAPH_PATH, _ARCHETYPE_SPEC, gateway=ONLINEBOUTIQUE_GATEWAY)


ONLINEBOUTIQUE_CALL_CHAINS = build_onlineboutique_call_chains()

__all__ = ['ONLINEBOUTIQUE_CALL_CHAINS', 'build_onlineboutique_call_chains',
           'ONLINEBOUTIQUE_GATEWAY']

if __name__ == '__main__':
    # Tu kiem: ONLINEBOUTIQUE_GATEWAY phai khop gateway suy tu CHINH do thi.
    _adj, _node_types = load_graph(GRAPH_PATH)
    _derived = derive_primary_gateway(_adj, _node_types)
    assert _derived == ONLINEBOUTIQUE_GATEWAY, (
        f"ONLINEBOUTIQUE_GATEWAY='{ONLINEBOUTIQUE_GATEWAY}' khong khop gateway suy tu "
        f"do thi hien tai ('{_derived}') -- graph co the da doi, kiem tra lai.")
    print(f"[OK] ONLINEBOUTIQUE_GATEWAY khop gateway suy tu do thi: {_derived}")
    all_services = set()
    for _name, _chain in ONLINEBOUTIQUE_CALL_CHAINS.items():
        print(f"  {_name}: {_chain['services']}")
        all_services.update(_chain['services'])
    print(f"\nTong service phu duoc boi {len(ONLINEBOUTIQUE_CALL_CHAINS)} archetype: "
          f"{sorted(all_services)} ({len(all_services)}/{len(_node_types)} node cua do thi)")
