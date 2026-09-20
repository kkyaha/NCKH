# -*- coding: utf-8 -*-
"""Taxonomy SockShop, phan 'services' cua tung archetype MINE TU LOG THAT
thay vi viet tay (xem src/scm/callchain_from_logs.py va kiem chung tren
90 run RE2-SS trong experiments/callchain_from_logs_validation.py, Jaccard
trung binh 0.60 so voi SOCKSHOP_CALL_CHAINS).

KHONG thay the SOCKSHOP_CALL_CHAINS lam mac dinh (request_router.py,
ParserAgent) -- do chi la MOT lua chon them, dang trong TAXONOMIES duoi ten
rieng 'SockShop_LogMined'. Ly do khong doi mac dinh: do chinh xac mine duoc
(Jaccard 0.40-0.67) khong dong deu, co truong hop (ADD_TO_CART) mat han
service 'carts' that su dinh nghia archetype do, doi lay 'payment'/'user'
loi vao tu cua so thoi gian chong lan -- doi mac dinh san pham can them danh
gia rieng, khong phai muc tieu cua module nay.

Gioi han cau truc (khong the khac di, du ky thuat mining tot den dau):
CHI mine duoc 'services' cho archetype DA TON TAI trong luu luong that (co
dong log quan sat duoc) -- 4/10 archetype cua SOCKSHOP_CALL_CHAINS
(REGISTER, GET_CATALOGUE, ADD_TO_CART, PLACE_ORDER, qua ENDPOINT_TO_
REQUEST_TYPE ben duoi). VIEW_CART/LOGIN (khong co endpoint rieng trong log
RE2-SS) va 4 tinh nang GIA DINH chua trien khai (APPLY_PROMO_CODE,
RECOMMEND_PRODUCTS, TRACK_PACKAGE, WRITE_PRODUCT_REVIEW) KHONG THE mine --
theo dinh nghia, mot tinh nang chua ton tai thi khong co dong log nao ghi
lai no. Nhung archetype nay GIU NGUYEN gia tri viet tay tu SOCKSHOP_CALL_
CHAINS, khong bi ghi de.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from callchain_from_logs import mine_call_chains_from_multiple, merge_with_archetype_meta  # noqa: E402
from request_router import SOCKSHOP_CALL_CHAINS  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DEFAULT_LOGS_GLOB = os.path.join(BASE_DIR, 'data', 'raw', 'RE2-SS', '*', '*', 'logs.csv')

# Anh xa endpoint (method+path quan sat duoc trong dong access-log) -> ten
# archetype cua SOCKSHOP_CALL_CHAINS -- CHI 4 archetype nay co the doi chieu,
# vi callchain_from_logs.py khong biet gi ve ten archetype, chi biet endpoint.
ENDPOINT_TO_REQUEST_TYPE = {
    'POST /register':  'REGISTER',
    'GET /catalogue':  'GET_CATALOGUE',
    'POST /cart':      'ADD_TO_CART',
    'POST /orders':    'PLACE_ORDER',
}

_META_FIELDS = ('description', 'keywords', 'resource_profile', 'expected_delta_pct')


def build_sockshop_call_chains_from_logs(logs_glob: str = None,
                                          base_chains: dict = None) -> dict:
    """Ban sao cua base_chains (mac dinh SOCKSHOP_CALL_CHAINS) voi truong
    'services' cua CAC ARCHETYPE MINE DUOC thay bang gia tri quan sat tu
    log that -- moi truong khac (keywords/description/resource_profile/
    expected_delta_pct) giu nguyen viet tay, dung schema CALL_CHAINS chuan
    de dua thang vao ParserAgent(call_chains=...).
    """
    if base_chains is None:
        base_chains = SOCKSHOP_CALL_CHAINS
    if logs_glob is None:
        logs_glob = DEFAULT_LOGS_GLOB

    logs_paths = sorted(glob.glob(logs_glob))
    mined = mine_call_chains_from_multiple(logs_paths) if logs_paths else {}

    archetype_meta = {
        ep: {k: base_chains[rt][k] for k in _META_FIELDS}
        for ep, rt in ENDPOINT_TO_REQUEST_TYPE.items()
        if ep in mined and rt in base_chains
    }
    merged = merge_with_archetype_meta(mined, archetype_meta)

    result = {k: dict(v) for k, v in base_chains.items()}
    for ep, rt in ENDPOINT_TO_REQUEST_TYPE.items():
        if ep in merged:
            result[rt]['services'] = merged[ep]['services']
    return result


SOCKSHOP_LOGMINED_CALL_CHAINS = build_sockshop_call_chains_from_logs()

__all__ = ['SOCKSHOP_LOGMINED_CALL_CHAINS', 'build_sockshop_call_chains_from_logs',
           'ENDPOINT_TO_REQUEST_TYPE']

if __name__ == '__main__':
    for name in ENDPOINT_TO_REQUEST_TYPE.values():
        hand  = SOCKSHOP_CALL_CHAINS[name]['services']
        mined = SOCKSHOP_LOGMINED_CALL_CHAINS[name]['services']
        flag  = '' if set(hand) == set(mined) else '  <-- khac ban tay'
        print(f"  {name:<15} tay={hand}  mine={mined}{flag}")
    print(f"\n{len(ENDPOINT_TO_REQUEST_TYPE)}/{len(SOCKSHOP_CALL_CHAINS)} archetype "
          f"duoc mine tu log, con lai giu nguyen viet tay.")
