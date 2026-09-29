# -*- coding: utf-8 -*-
"""
DO CHUOI GOI THUC TE CUA TINH NANG (chuc nang + backend nao bi goi, moi lan dung goi may lan)
=============================================================================================
Tao mot khach that (dia chi, the, 1 mon trong gio, 1 don cu), goi route cua tinh nang N lan, va dem so request den tung
service qua /metrics (chenh lech bo dem `request_duration_seconds_count`, bo route `metrics`). Dung de:
  * kiem tra tinh nang HOAT DONG (ma tra ve, than tra ve) truoc khi ton ~2 gio do ramp;
  * do k_s thuc (so lan moi service bi goi cho moi lan dung) va SO SANH voi chain cua taxonomy (nguon tuyen duong cua cong cu).

    python experiments/probe_feature_chain.py --features cartsum,quickadd,express --n 20

Chay khi he thong ranh (khong co tai khac), neu khong so dem bi lan.
"""

import argparse
import json
import re
import subprocess
import sys
import os
import time
from collections import Counter

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
import load_sweep_collect as L  # noqa: E402  (FEATURES, taxonomy)
import feasibility_predictor as FP  # noqa: E402

SVCS = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
PORT = {'front-end': 8079}


_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')      # cung quy uoc voi collector.py


def counts():
    """Bo dem request moi service, TACH THEO METHOD.

    Truoc day ham nay cong don moi nhan lai thanh mot so. Nhan `method` VAN CO
    trong `/metrics` (collector.py da doc no de dung routes.csv) -- giu lai thi
    tach duoc loi goi DOC khoi loi goi GHI ma khong ton them mot lan goi nao.
    Ly do can: chi phi mot loi goi ghi do duoc dat hon mot loi goi doc 1.83x o
    `carts` va 4.0x o `carts-db` (13.9x disk I/O), nen boi so k dem theo SO LUOT
    goi danh gia thap chi phi that cua tinh nang thien ve ghi.
    """
    script = ('for t in ' + ' '.join(f'{s}:{PORT.get(s, 80)}' for s in SVCS) +
              '; do echo "@@$t"; curl -s http://$t/metrics | grep "^request_duration_seconds_count"; done')
    out = subprocess.run(['docker', 'run', '--rm', '--network', 'sockshop_default', 'curlimages/curl:8.10.1', 'sh', '-c', script],
                         capture_output=True, text=True).stdout
    res, cur = {}, None
    for l in out.splitlines():
        if l.startswith('@@'):
            cur = l[2:].split(':')[0]
            res[cur] = {'total': 0.0, 'by_method': {}}
            continue
        m = re.match(r'^request_duration_seconds_count\{(.*)\}\s+(\S+)', l)
        if not m or 'metrics' in m.group(1):
            continue
        lb, val = dict(_LABEL.findall(m.group(1))), float(m.group(2))
        res[cur]['total'] += val
        meth = lb.get('method', '?').upper()
        res[cur]['by_method'][meth] = res[cur]['by_method'].get(meth, 0.0) + val
    return res


LAST_USER = {}
PROBE_ITEMS = []
_REG_N = [0]
SEARCH_WORDS = ['sock', 'blue', 'sport', 'magic', 'red', 'geek', 'black', 'holy', 'green', 'formal']


def make_user(c):
    u = f'probe{time.time_ns() % 10**10}'
    LAST_USER['u'] = u
    c.post('/register', json={'username': u, 'password': 'pw123456', 'email': u + '@x.io'})
    c.get('/login', auth=(u, 'pw123456'))
    c.post('/addresses', json={'street': 'S', 'number': '1', 'country': 'US', 'city': 'N', 'postcode': '1'})
    c.post('/cards', json={'longNum': '5544154011345918', 'expires': '08/23', 'ccv': '123'})
    cat = c.get('/catalogue?size=50').json()
    cheap = [i['id'] for i in cat if i['price'] <= 50]
    c.post('/cart', json={'id': cheap[0], 'quantity': 1})
    r_ord = c.post('/orders', json={})               # don cu (don xoa gio)
    try:
        LAST_USER['order'] = (r_ord.json() or {}).get('id')   # `reorder` can mot don cu de mua lai
    except Exception:
        LAST_USER['order'] = None
    c.post('/cart', json={'id': cheap[0], 'quantity': 1})
    PROBE_ITEMS[:] = [i['id'] for i in cat]
    return cheap, [i['id'] for i in cat]


def body_for(feat, cheap, items):
    if feat == 'promo':
        return {'code': 'SAVE20'}
    if feat == 'review':
        return {'productId': items[0], 'stars': 5, 'text': 'ok'}
    if feat == 'quickadd':
        return {'id': items[1]}
    if feat == 'express':
        return {'id': cheap[1]}
    # ---- vong tien cuu 2
    if feat == 'login':
        return {'username': LAST_USER['u'], 'password': 'pw123456'}
    if feat == 'register':
        _REG_N[0] += 1
        u = f'pr{time.time_ns() % 10**10}x{_REG_N[0]}'
        return {'username': u, 'password': 'pw123456', 'email': u + '@x.io'}
    if feat == 'wishlist':
        return {'id': items[_REG_N[0] % len(items)]}
    if feat == 'reorder':
        return {'orderId': LAST_USER.get('order')}
    return None


def url_for(feat, url):
    if feat == 'catsearch':
        _REG_N[0] += 1
        return f'{url}?q={SEARCH_WORDS[_REG_N[0] % len(SEARCH_WORDS)]}'
    if feat == 'related':
        _REG_N[0] += 1
        return f'{url}?id={PROBE_ITEMS[_REG_N[0] % len(PROBE_ITEMS)]}'
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--n-latency', type=int, default=200,
                    help='so loi goi RIENG de do do tre luc he RANH (D_feat). Tach khoi --n '
                         'de phep do k giu nguyen bit-for-bit so voi cac lan chay truoc; '
                         'n=20 khong du uoc luong phan vi cao nen mac dinh lon hon.')
    ap.add_argument('--user-every', type=int, default=20,
                    help='xoay sang tai khoan moi sau bao nhieu loi goi trong pha do do tre '
                         '(tranh gio hang/don hang cong don lam do tre trôi)')
    ap.add_argument('--save', default='', help='ghi ket qua k do duoc ra JSON (dung cho P3)')
    a = ap.parse_args()
    c = httpx.Client(base_url='http://127.0.0.1', timeout=30)
    cheap, items = make_user(c)
    print(f'\n=== CHUOI GOI THUC TE ({a.n} lan/tinh nang) ===')
    out = {}
    for feat in a.features.split(','):
        meta = L.FEATURES[feat]
        method, url = meta['req']
        tax = FP.SOCKSHOP_CALL_CHAINS[meta['archetype']]['services']
        codes, sample = Counter(), None
        if feat in ('login', 'register', 'wishlist', 'account', 'preview', 'orderhist', 'related',
                    'orderfull', 'reorder'):
            cheap, items = make_user(c)                 # khach moi cho moi tinh nang (register doi danh tinh phien)
        counts()                                        # lam nong ket noi
        before = counts()
        for _ in range(a.n):
            kw = {}
            b = body_for(feat, cheap, items)
            if b is not None:
                kw['json'] = b
            r = c.request(method, url_for(feat, url), **kw)
            codes[r.status_code] += 1
            sample = sample or r.text[:200]
        after = counts()
        per = {s: round((after[s]['total'] - before[s]['total']) / a.n, 2) for s in SVCS}
        hit = [s for s in SVCS if per[s] >= 0.5]

        # --- boi so goi TACH THEO METHOD (doc vs ghi), cung mot lan probe
        per_meth = {}
        for s in SVCS:
            d = {m: round((after[s]['by_method'].get(m, 0.0) - before[s]['by_method'].get(m, 0.0)) / a.n, 2)
                 for m in set(before[s]['by_method']) | set(after[s]['by_method'])}
            d = {m: v for m, v in d.items() if v}
            if d:
                per_meth[s] = d

        # --- D_feat: do tre cua CHINH tinh nang khi he RANH, khong dung du lieu tai nao
        # PHAI XOAY TAI KHOAN: body_for() co TAC DUNG PHU TICH LUY (quickadd them hang vao
        # gio, express dat don). Goi lien tuc tren MOT tai khoan thi gio phinh dan va phep
        # do tu bop meo chinh no -- do tre tang vi lich su, khong phai vi tinh nang. Harness
        # do tai da ne bang pool tai khoan (docs/DATA_FRAMEWORK.md muc 7); o day xoay tai
        # khoan moi `--user-every` loi goi de D_feat do dung thu ma ramp se gap.
        lat = []
        for i in range(a.n_latency):
            if i and i % a.user_every == 0:
                cheap, items = make_user(c)          # ngoai vung bam gio
            b2 = body_for(feat, cheap, items)
            kw2 = {'json': b2} if b2 is not None else {}
            t0 = time.perf_counter()
            c.request(method, url_for(feat, url), **kw2)
            lat.append(time.perf_counter() - t0)
        lat.sort()
        q = lambda p: lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))]
        d_feat = {'n': len(lat), 'mean': round(sum(lat) / len(lat), 5),
                  'p50': round(q(.50), 5), 'p90': round(q(.90), 5),
                  'p95': round(q(.95), 5), 'p99': round(q(.99), 5), 'max': round(lat[-1], 5),
                  'samples_ms': [round(x * 1000, 3) for x in lat]}
        print(f'\n[{feat}] {method} {url} -> ma tra ve {dict(codes)}')
        print(f'  mau than tra ve: {sample}')
        print(f'  so lan moi service bi goi cho MOI lan dung: { {s: v for s, v in per.items() if v} }')
        print(f'  chain THUC TE (>=0.5 lan/lan dung): {hit}')
        print(f'  chain taxonomy ({meta["archetype"]}):    {tax}')
        print(f'  => {"KHOP" if set(hit) == set(tax) else "KHAC"}'
              + ('' if set(hit) == set(tax) else f'  (thua: {sorted(set(hit) - set(tax))}, thieu: {sorted(set(tax) - set(hit))})'))
        w = {s: sum(v for m, v in d.items() if m != 'GET') for s, d in per_meth.items()}
        print(f'  boi so theo METHOD: {per_meth}')
        print(f'    -> luot GHI (khong phai GET) moi lan dung: { {s: v for s, v in w.items() if v} or "khong co" }')
        print(f'  D_feat khi he RANH ({d_feat["n"]} loi goi): p50={d_feat["p50"]*1000:.1f}ms  '
              f'p95={d_feat["p95"]*1000:.1f}ms  p99={d_feat["p99"]*1000:.1f}ms  max={d_feat["max"]*1000:.1f}ms')
        out[feat] = {'measured_per_use': per, 'chain_measured': hit, 'chain_taxonomy': tax,
                     'archetype': meta['archetype'], 'n_probes': a.n,
                     'per_use_by_method': per_meth, 'latency_idle': d_feat}

    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)) or '.', exist_ok=True)
        json.dump({'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'n_probes': a.n, 'features': out},
                  open(a.save, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print(f'\n[OK] k do duoc -> {a.save}')


if __name__ == '__main__':
    main()
