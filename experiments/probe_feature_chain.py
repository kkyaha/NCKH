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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_sweep_collect as L  # noqa: E402  (FEATURES, taxonomy)
import feasibility_predictor as FP  # noqa: E402

SVCS = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
PORT = {'front-end': 8079}


def counts():
    script = ('for t in ' + ' '.join(f'{s}:{PORT.get(s, 80)}' for s in SVCS) +
              '; do echo "@@$t"; curl -s http://$t/metrics | grep "^request_duration_seconds_count"; done')
    out = subprocess.run(['docker', 'run', '--rm', '--network', 'sockshop_default', 'curlimages/curl:8.10.1', 'sh', '-c', script],
                         capture_output=True, text=True).stdout
    res, cur = {}, None
    for l in out.splitlines():
        if l.startswith('@@'):
            cur = l[2:].split(':')[0]
            res[cur] = 0.0
            continue
        m = re.match(r'^request_duration_seconds_count\{(.*)\}\s+(\S+)', l)
        if m and 'metrics' not in m.group(1):
            res[cur] += float(m.group(2))
    return res


def make_user(c):
    u = f'probe{time.time_ns() % 10**10}'
    c.post('/register', json={'username': u, 'password': 'pw123456', 'email': u + '@x.io'})
    c.get('/login', auth=(u, 'pw123456'))
    c.post('/addresses', json={'street': 'S', 'number': '1', 'country': 'US', 'city': 'N', 'postcode': '1'})
    c.post('/cards', json={'longNum': '5544154011345918', 'expires': '08/23', 'ccv': '123'})
    cat = c.get('/catalogue?size=50').json()
    cheap = [i['id'] for i in cat if i['price'] <= 50]
    c.post('/cart', json={'id': cheap[0], 'quantity': 1})
    c.post('/orders', json={})                       # don cu (don xoa gio)
    c.post('/cart', json={'id': cheap[0], 'quantity': 1})
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
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--n', type=int, default=20)
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
        counts()                                        # lam nong ket noi
        before = counts()
        for _ in range(a.n):
            kw = {}
            b = body_for(feat, cheap, items)
            if b is not None:
                kw['json'] = b
            r = c.request(method, url, **kw)
            codes[r.status_code] += 1
            sample = sample or r.text[:200]
        after = counts()
        per = {s: round((after[s] - before[s]) / a.n, 2) for s in SVCS}
        hit = [s for s in SVCS if per[s] >= 0.5]
        print(f'\n[{feat}] {method} {url} -> ma tra ve {dict(codes)}')
        print(f'  mau than tra ve: {sample}')
        print(f'  so lan moi service bi goi cho MOI lan dung: { {s: v for s, v in per.items() if v} }')
        print(f'  chain THUC TE (>=0.5 lan/lan dung): {hit}')
        print(f'  chain taxonomy ({meta["archetype"]}):    {tax}')
        print(f'  => {"KHOP" if set(hit) == set(tax) else "KHAC"}'
              + ('' if set(hit) == set(tax) else f'  (thua: {sorted(set(hit) - set(tax))}, thieu: {sorted(set(tax) - set(hit))})'))
        out[feat] = {'measured_per_use': per, 'chain_measured': hit, 'chain_taxonomy': tax,
                     'archetype': meta['archetype'], 'n_probes': a.n}

    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)) or '.', exist_ok=True)
        json.dump({'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'n_probes': a.n, 'features': out},
                  open(a.save, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print(f'\n[OK] k do duoc -> {a.save}')


if __name__ == '__main__':
    main()
