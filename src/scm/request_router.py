# -*- coding: utf-8 -*-
"""
Module 1: Request Router
Map natural language request -> Request Type -> Blast Radius (affected services).
Based on SockShop architecture and log templates from cluster_info.json.
"""

import os
import json

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'

# Call chain per request type (from SockShop architecture + log template analysis)
CALL_CHAINS = {
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
        'description': 'TÍNH NĂNG MỚI: Khách hàng áp mã giảm giá / voucher khuyến mãi khi thanh toán',
        'keywords': ['promo', 'discount', 'voucher', 'ap ma', 'giam gia', 'khuyen mai', 'coupon', 'code', 'km', 'ma'],
        'resource_profile': 'cpu-heavy',
        'expected_delta_pct': 20,
    },
    'RECOMMEND_PRODUCTS': {
        'services': ['front-end', 'user', 'catalogue', 'orders'],
        'description': 'TÍNH NĂNG MỚI: Gợi ý sản phẩm thông minh dựa trên lịch sử mua sắm',
        'keywords': ['recommend', 'goi y', 'phu hop', 'recommendation', 'ai'],
        'resource_profile': 'cpu-memory',
        'expected_delta_pct': 30,
    },
    'TRACK_PACKAGE': {
        'services': ['front-end', 'orders', 'shipping'],
        'description': 'TÍNH NĂNG MỚI: Theo dõi hành trình giao hàng và trạng thái vận chuyển real-time',
        'keywords': ['track', 'tracking', 'hanh trinh', 'giao hang', 'don hang o dau', 'van chuyen', 'shipping status', 'theo doi', 'vi tri'],
        'resource_profile': 'socket-latency',
        'expected_delta_pct': 15,
    },
    'WRITE_PRODUCT_REVIEW': {
        'services': ['front-end', 'user', 'catalogue'],
        'description': 'TÍNH NĂNG MỚI: Khách hàng viết đánh giá và chấm điểm sao cho sản phẩm',
        'keywords': ['review', 'danh gia', 'binh luan', 'nhan xet', 'rating', 'comment'],
        'resource_profile': 'disk-memory',
        'expected_delta_pct': 10,
    },
}

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


def get_blast_radius(request_type: str) -> dict:
    """Return affected services, templates, and expected workload delta for a request type."""
    if request_type not in CALL_CHAINS:
        raise ValueError(f"Unknown: {request_type}. Choose from {list(CALL_CHAINS.keys())}")
    
    chain = CALL_CHAINS[request_type]
    log_templates = {
        svc: SERVICE_KEY_TEMPLATES[svc]
        for svc in chain['services']
        if svc in SERVICE_KEY_TEMPLATES
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

def classify_request(text: str) -> str:
    """Simple keyword-based classifier with Vietnamese accent normalization."""
    text_clean = remove_accents(text.lower())
    scores = {}
    for rtype, info in CALL_CHAINS.items():
        score = 0
        for kw in info['keywords']:
            kw_clean = remove_accents(kw.lower())
            pattern = r'\b' + re.escape(kw_clean) + r'\b'
            if re.search(pattern, text_clean):
                score += 1
        scores[rtype] = score

    max_score = max(scores.values())
    if max_score == 0:
        return 'GET_CATALOGUE'
    # On tie, heavier/specific request wins
    priority_order = [
        'PLACE_ORDER', 'APPLY_PROMO_CODE', 'RECOMMEND_PRODUCTS',
        'TRACK_PACKAGE', 'WRITE_PRODUCT_REVIEW', 'ADD_TO_CART',
        'VIEW_CART', 'REGISTER', 'LOGIN', 'GET_CATALOGUE'
    ]
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

