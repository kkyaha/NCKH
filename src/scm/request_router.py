# -*- coding: utf-8 -*-
"""
Module 1: Request Router
Map natural language request -> Request Type -> Blast Radius (affected services).
Based on SockShop architecture and log templates from cluster_info.json.
"""

import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Call chain per request type (from SockShop architecture + log template analysis).
# NOTE: day la taxonomy CUA SOCKSHOP. Moi ham trong module nay deu nhan
# call_chains lam tham so; hang so nay chi la GIA TRI MAC DINH de khong pha
# code cu. Taxonomy he khac: xem src/scm/taxonomies/.
SOCKSHOP_CALL_CHAINS = {
    'GET_CATALOGUE': {
        'services': ['front-end', 'catalogue'],
        'description': 'Customer views product list or detail',
        'keywords': ['catalogue', 'browse', 'product', 'xem', 'san pham', 'hang hoa', 'danh sach'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
    'ADD_TO_CART': {
        'services': ['front-end', 'catalogue', 'carts'],
        'description': 'Customer adds item to cart',
        'keywords': ['cart', 'add', 'them', 'gio hang', 'them vao gio', 'add to cart'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 15,
    },
    'VIEW_CART': {
        'services': ['front-end', 'carts'],
        'description': 'Customer views their cart',
        'keywords': ['view cart', 'xem gio', 'gio hang cua toi', 'my cart'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
    'REGISTER': {
        'services': ['front-end', 'user'],
        'description': 'New customer registration',
        'keywords': ['register', 'signup', 'dang ky', 'tao tai khoan', 'new account'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
    'LOGIN': {
        'services': ['front-end', 'user'],
        'description': 'Customer login',
        'keywords': ['login', 'signin', 'dang nhap', 'vao tai khoan', 'authenticate'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
    },
    'PLACE_ORDER': {
        'services': ['front-end', 'user', 'catalogue', 'carts', 'orders', 'payment', 'shipping'],
        'description': 'Customer places a full order (heaviest operation)',
        'keywords': ['order', 'checkout', 'buy', 'purchase', 'dat hang', 'mua hang',
                     'thanh toan', 'dat mua', 'payment', 'place order'],
        'resource_profile': 'cpu-heavy',
        'expected_delta_pct': 25,
    },
    # --- NEW HYPOTHETICAL FEATURES (NOT YET IN SOCKSHOP) ---
    'APPLY_PROMO_CODE': {
        'services': ['front-end', 'carts', 'orders', 'payment'],
        'description': 'NEW FEATURE: Customer applies discount promo code or voucher during checkout',
        'keywords': ['promo', 'discount', 'voucher', 'ap ma', 'giam gia', 'khuyen mai', 'coupon', 'code', 'km', 'ma'],
        'resource_profile': 'cpu-heavy',
        'expected_delta_pct': 20,
    },
    'RECOMMEND_PRODUCTS': {
        'services': ['front-end', 'user', 'catalogue', 'orders'],
        'description': 'NEW FEATURE: Smart product recommendations based on shopping history',
        'keywords': ['recommend', 'goi y', 'phu hop', 'recommendation', 'ai'],
        'resource_profile': 'cpu-memory',
        'expected_delta_pct': 30,
    },
    'TRACK_PACKAGE': {
        'services': ['front-end', 'orders', 'shipping'],
        'description': 'NEW FEATURE: Real-time package tracking and delivery status',
        'keywords': ['track', 'package', 'shipping', 'theo doi', 'don hang', 'tinh trang', 'giao hang', 'status', 'vi tri'],
        'resource_profile': 'socket-latency',
        'expected_delta_pct': 15,
    },
    'WRITE_PRODUCT_REVIEW': {
        'services': ['front-end', 'user', 'catalogue'],
        'description': 'NEW FEATURE: Customer writes a review and rates a purchased product',
        'keywords': ['review', 'rate', 'danh gia', 'nhan xet', 'rating', 'comment'],
        'resource_profile': 'disk-memory',
        'expected_delta_pct': 10,
    },
}

# Alias giu tuong thich nguoc: code cu `from request_router import CALL_CHAINS`
# van chay, va van tro dung taxonomy SockShop.
CALL_CHAINS = SOCKSHOP_CALL_CHAINS

# Archetype tra ve khi KHONG tu khoa nao khop. Truoc day viet cung trong
# classify_request. Suy ra tu bang (archetype nhe nhat) se doi gia tri nay
# thanh VIEW_CART tren SockShop -- mot thay doi hanh vi lam lech RQ3 -- nen
# moi taxonomy phai TU KHAI BAO default cua no.
SOCKSHOP_DEFAULT_TYPE = 'GET_CATALOGUE'

# Log templates per service that signal the service was called
SERVICE_KEY_TEMPLATES = {
    'payment':      [29],       # method=Authorise
    'orders':       [28, 30],   # Sending/Received payment request
    'shipping':     [31],       # Adding shipment to queue
    'user':         [20, 26, 27],
    'catalogue':    [8, 14],
    'carts':        [49],
    'front-end':    [4, 7, 17, 38],
}


def get_blast_radius(request_type: str, call_chains: dict = None,
                     service_templates: dict = None) -> dict:
    """Return affected services, templates, and expected workload delta for a request type."""
    call_chains = call_chains if call_chains is not None else CALL_CHAINS
    service_templates = service_templates if service_templates is not None else SERVICE_KEY_TEMPLATES
    if request_type not in call_chains:
        raise ValueError(f"Unknown: {request_type}. Choose from {list(call_chains.keys())}")

    chain = call_chains[request_type]
    log_templates = {
        svc: service_templates[svc]
        for svc in chain['services']
        if svc in service_templates
    }
    
    return {
        'request_type':       request_type,
        'affected_services':  chain['services'],
        'n_services':         len(chain['services']),
        'description':        chain['description'],
        'resource_profile':   chain['resource_profile'],
        'expected_delta_pct': chain['expected_delta_pct'],
        'log_templates':      log_templates,
    }


import re
import unicodedata

def remove_accents(input_str: str) -> str:
    nfkd_form = unicodedata.normalize('NFD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).replace('đ', 'd').replace('Đ', 'D')

def default_priority_order(call_chains: dict) -> list:
    """Thu tu uu tien pha hoa, suy ra TU BANG thay vi liet ke cung.

    Quy tac: yeu cau "nang"/cu the hon thang khi hoa diem tu khoa -- do o day
    bang so service trong call chain (giam dan), roi alphabet de deterministic.
    Nho vay mot taxonomy moi khong can khai bao thu tu bang tay.
    """
    return sorted(call_chains.keys(),
                  key=lambda rt: (-len(call_chains[rt].get('services', [])), rt))


def classify_request(text: str, call_chains: dict = None,
                     priority_order: list = None,
                     default_type: str = None) -> str:
    """Keyword-based classifier with Vietnamese accent normalization.

    call_chains / priority_order / default_type deu tham so hoa de bo phan
    phu thuoc cung vao taxonomy SockShop (xem SOCKSHOP_CALL_CHAINS). Khong
    truyen gi -> giu nguyen hanh vi cu.
    """
    # Kiem tra DINH DANH, khong chi None: ParserAgent truyen tuong minh
    # self.call_chains, voi SockShop chinh la object nay.
    using_default_table = (call_chains is None or call_chains is SOCKSHOP_CALL_CHAINS)
    call_chains = call_chains if call_chains is not None else CALL_CHAINS
    if not call_chains:
        raise ValueError("call_chains rong: khong the phan loai.")
    if priority_order is None:
        priority_order = default_priority_order(call_chains)
    if default_type is None:
        if using_default_table:
            # Tuong thich nguoc tuyet doi voi moi so lieu SockShop da cong bo.
            default_type = SOCKSHOP_DEFAULT_TYPE
        else:
            # Taxonomy moi khong khai bao -> archetype nhe nhat lam fallback.
            default_type = priority_order[-1]

    text_clean = remove_accents(text.lower())
    scores = {}
    for rtype, info in call_chains.items():
        score = 0
        for kw in info.get('keywords', []):
            kw_clean = remove_accents(kw.lower())
            pattern = r'\b' + re.escape(kw_clean) + r'\b'
            if re.search(pattern, text_clean):
                score += 1
        scores[rtype] = score

    max_score = max(scores.values())
    if max_score == 0:
        return default_type
    for rtype in priority_order:
        if scores.get(rtype, 0) == max_score:
            return rtype
    return max(scores, key=scores.get)


if __name__ == '__main__':
    tests = [
        "toi muon dat hang mua san pham",
        "toi muon ap ma giam gia khuyen mai voucher 20%",
        "goi y san pham thong minh cho toi",
        "theo doi hanh trinh giao hang don hang",
        "viet danh gia va nhan xet san pham",
        "add item to cart",
        "xem san pham",
        "dang ky tai khoan moi",
    ]
    for t in tests:
        rtype = classify_request(t)
        blast = get_blast_radius(rtype)
        print(f"'{t}' -> {rtype}: {blast['affected_services']}")

