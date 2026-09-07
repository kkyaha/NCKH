# -*- coding: utf-8 -*-
"""
TRAIN TICKET REQUEST ROUTER
===========================
Maps natural language requests -> Train Ticket Request Type -> Blast Radius.
Supports 8 core and extended Train Ticket call chains spanning 40+ microservices.
"""

import os
import re
import unicodedata
from typing import Dict, Any

CALL_CHAINS: Dict[str, Dict[str, Any]] = {
    'SEARCH_TRIP': {
        'services': [
            'ts-ui-dashboard',
            'ts-travel-service',
            'ts-route-service',
            'ts-ticketinfo-service',
            'ts-basic-service',
            'ts-station-service',
            'ts-train-service',
            'ts-price-service'
        ],
        'description': 'Customer searches for train routes, schedules, and ticket prices',
        'keywords': ['search', 'trip', 'travel', 'route', 'train', 'tim ve', 'tra cuu', 'hanh trinh', 'gia ve', 'chuyen tau', 'gio chay'],
        'resource_profile': 'cpu-memory',
        'expected_delta_pct': 15,
        'entrypoint': 'ts-travel-service'
    },
    'PRESERVE_TICKET': {
        'services': [
            'ts-ui-dashboard',
            'ts-preserve-service',
            'ts-security-service',
            'ts-contacts-service',
            'ts-seat-service',
            'ts-order-service',
            'ts-user-service',
            'ts-station-service',
            'ts-assurance-service'
        ],
        'description': 'Customer reserves and books train tickets (Heaviest core transaction)',
        'keywords': ['preserve', 'book', 'reserve', 'dat ve', 'mua ve', 'giu cho', 'booking', 'ticket', 've tau'],
        'resource_profile': 'cpu-heavy',
        'expected_delta_pct': 25,
        'entrypoint': 'ts-preserve-service'
    },
    'PRESERVE_OTHER_TICKET': {
        'services': [
            'ts-ui-dashboard',
            'ts-preserve-other-service',
            'ts-travel2-service',
            'ts-security-service',
            'ts-contacts-service',
            'ts-seat-service',
            'ts-order-other-service',
            'ts-user-service',
            'ts-assurance-service'
        ],
        'description': 'Customer reserves secondary/cross-regional train tickets',
        'keywords': ['preserve other', 've lien van', 've dac biet', 'chuyen tuyen', 'other ticket'],
        'resource_profile': 'cpu-heavy',
        'expected_delta_pct': 20,
        'entrypoint': 'ts-preserve-other-service'
    },
    'PAY_ORDER': {
        'services': [
            'ts-ui-dashboard',
            'ts-inside-payment-service',
            'ts-order-service',
            'ts-payment-service'
        ],
        'description': 'Customer pays for booked train tickets via internal payment gateway',
        'keywords': ['pay', 'payment', 'thanh toan', 'tien ve', 'tra tien', 'banking', 'wallet', 'chuyen khoan'],
        'resource_profile': 'cpu-socket',
        'expected_delta_pct': 20,
        'entrypoint': 'ts-inside-payment-service'
    },
    'COLLECT_TICKET': {
        'services': [
            'ts-ui-dashboard',
            'ts-execute-service',
            'ts-order-service'
        ],
        'description': 'Customer prints and collects physical tickets at the train station',
        'keywords': ['collect', 'print', 'nhan ve', 'lay ve', 'in ve', 'check in', 'execute'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 10,
        'entrypoint': 'ts-execute-service'
    },
    'CANCEL_ORDER': {
        'services': [
            'ts-ui-dashboard',
            'ts-cancel-service',
            'ts-order-service',
            'ts-inside-payment-service'
        ],
        'description': 'Customer cancels booked tickets with automated refund',
        'keywords': ['cancel', 'refund', 'huy ve', 'tra ve', 'hoan tien', 'huy don'],
        'resource_profile': 'cpu-heavy',
        'expected_delta_pct': 15,
        'entrypoint': 'ts-cancel-service'
    },
    'FOOD_DELIVERY_ON_TRAIN': {
        'services': [
            'ts-ui-dashboard',
            'ts-food-service',
            'ts-food-map-service',
            'ts-travel-service',
            'ts-station-service'
        ],
        'description': 'NEW FEATURE: Order meals delivered directly to train seats during transit',
        'keywords': ['food', 'meal', 'dat com', 'do an', 'suat an', 'mon an', 'delivery', 'an uong'],
        'resource_profile': 'cpu-memory',
        'expected_delta_pct': 25,
        'entrypoint': 'ts-food-service'
    },
    'CONSIGN_BAGGAGE': {
        'services': [
            'ts-ui-dashboard',
            'ts-consign-service',
            'ts-consign-price-service'
        ],
        'description': 'Customer consigns heavy baggage and packages via freight cars',
        'keywords': ['consign', 'baggage', 'cargo', 'ky gui', 'hanh ly', 'hang hoa', 'gui do'],
        'resource_profile': 'cpu',
        'expected_delta_pct': 15,
        'entrypoint': 'ts-consign-service'
    }
}


def remove_accents(input_str: str) -> str:
    nfkd_form = unicodedata.normalize('NFD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).replace('đ', 'd').replace('Đ', 'D')


def classify_request(text: str) -> str:
    """Classify natural language query to Train Ticket request type."""
    text_clean = remove_accents(text.lower())
    scores = {}
    for rtype, info in CALL_CHAINS.items():
        score = 0
        for kw in info['keywords']:
            kw_clean = remove_accents(kw.lower())
            pattern = r'\b' + re.escape(kw_clean) + r'\b'
            if re.search(pattern, text_clean):
                # Multi-word phrase matches get higher weight
                score += 2 if ' ' in kw_clean else 1
        scores[rtype] = score

    max_score = max(scores.values())
    if max_score == 0:
        return 'SEARCH_TRIP'

    priority_order = [
        'CANCEL_ORDER', 'COLLECT_TICKET', 'PAY_ORDER',
        'FOOD_DELIVERY_ON_TRAIN', 'CONSIGN_BAGGAGE',
        'PRESERVE_OTHER_TICKET', 'PRESERVE_TICKET', 'SEARCH_TRIP'
    ]
    for rtype in priority_order:
        if scores.get(rtype, 0) == max_score:
            return rtype
    return max(scores, key=scores.get)


def get_blast_radius(request_type: str) -> Dict[str, Any]:
    """Return affected services and metadata for a given request type."""
    if request_type not in CALL_CHAINS:
        raise ValueError(f"Unknown request type: {request_type}. Available: {list(CALL_CHAINS.keys())}")

    chain = CALL_CHAINS[request_type]
    return {
        'request_type': request_type,
        'affected_services': chain['services'],
        'n_services': len(chain['services']),
        'description': chain['description'],
        'resource_profile': chain['resource_profile'],
        'expected_delta_pct': chain['expected_delta_pct'],
        'entrypoint': chain['entrypoint']
    }


if __name__ == '__main__':
    test_queries = [
        "Toi muon dat ve tau tu Ha Noi den Sai Gon",
        "Thanh toan tien ve tau qua cong inside payment",
        "Tra cuu hanh trinh va gio chay tau",
        "Huy ve tau va hoan lai tien",
        "Dat them suat an mon an nong giao tan ghe ngoi",
        "Ky gui hanh ly va hang hoa theo chuyen tau",
        "In va lay ve vat ly tai cay ATM ga tau"
    ]
    print("=" * 70)
    print("  TESTING TRAIN TICKET REQUEST ROUTER")
    print("=" * 70)
    for q in test_queries:
        rtype = classify_request(q)
        blast = get_blast_radius(rtype)
        print(f"\nQuery: '{q}'")
        print(f"  -> Type: {rtype}")
        print(f"  -> Entrypoint: {blast['entrypoint']}")
        print(f"  -> Blast Radius ({blast['n_services']} services): {blast['affected_services']}")
