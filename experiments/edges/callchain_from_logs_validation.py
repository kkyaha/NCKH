# -*- coding: utf-8 -*-
"""
KIEM CHUNG: callchain_from_logs.py mine dung services toi muc nao?
====================================================================
So sanh services mine TU LOG THO (src/scm/callchain_from_logs.py, khong
gia dinh ten dich vu/route nao) voi SOCKSHOP_CALL_CHAINS -- bang hieu
chinh VIET TAY hien co trong src/scm/request_router.py, dung lam so sanh
DOC LAP vi no duoc viet tu tai lieu kien truc SockShop, khong tu chinh
log nay.

Chay tren TOAN BO 90 (scenario x run) cua RE2-SS de gom du quan sat cho ca
endpoint hiem, khong chi 1 run don.

Dung: python experiments/callchain_from_logs_validation.py
"""

import glob
import os
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))

from callchain_from_logs import mine_call_chains_from_multiple  # noqa: E402
from request_router import SOCKSHOP_CALL_CHAINS  # noqa: E402

# Anh xa endpoint (method+path quan sat duoc trong log) -> request_type cua
# bang tay hien co -- CHI dung de SO SANH trong script nay, khong dua vao
# module callchain_from_logs.py (module do khong biet gi ve anh xa nay).
ENDPOINT_TO_REQUEST_TYPE = {
    'POST /register':  'REGISTER',
    'GET /catalogue':  'GET_CATALOGUE',
    'POST /cart':      'ADD_TO_CART',
    'POST /orders':    'PLACE_ORDER',
}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    logs_paths = sorted(glob.glob(os.path.join(
        BASE_DIR, 'data', 'raw', 'RE2-SS', '*', '*', 'logs.csv')))
    print(f"[1] Tim thay {len(logs_paths)} file logs.csv (RE2-SS, toan bo scenario x run).")

    t0 = time.time()
    mined = mine_call_chains_from_multiple(logs_paths)
    print(f"[2] Da mine {len(mined)} endpoint trong {time.time() - t0:.1f}s.\n")

    print("=" * 78)
    print("KET QUA MINE (tu log tho, khong biet truoc kien truc SockShop)")
    print("=" * 78)
    for ep, info in sorted(mined.items()):
        freqs = ", ".join(f"{s}={f:.2f}" for s, f in info['service_frequency'].items())
        print(f"  {ep:<20} n_obs={info['n_observations']:<6} services={info['services']}")
        print(f"    {'':<20} freq: {freqs}")

    print("\n" + "=" * 78)
    print("SO SANH VOI SOCKSHOP_CALL_CHAINS (bang tay, request_router.py)")
    print("=" * 78)
    rows = []
    for ep, rtype in ENDPOINT_TO_REQUEST_TYPE.items():
        if ep not in mined:
            print(f"  [BO QUA] {ep}: khong mine duoc (khong xuat hien trong log).")
            continue
        mined_set = set(mined[ep]['services'])
        hand_set = set(SOCKSHOP_CALL_CHAINS[rtype]['services'])
        j = jaccard(mined_set, hand_set)
        rows.append((ep, rtype, mined_set, hand_set, j))
        print(f"\n  {ep} <-> {rtype}")
        print(f"    Mine tu log : {sorted(mined_set)}")
        print(f"    Bang tay    : {sorted(hand_set)}")
        print(f"    Jaccard     : {j:.2f}  "
              f"(giao={sorted(mined_set & hand_set)}, "
              f"chi-mine={sorted(mined_set - hand_set)}, "
              f"chi-tay={sorted(hand_set - mined_set)})")

    if rows:
        avg_j = sum(r[4] for r in rows) / len(rows)
        print(f"\n[3] Jaccard trung binh tren {len(rows)} request type doi chieu duoc: {avg_j:.3f}")


if __name__ == '__main__':
    main()
