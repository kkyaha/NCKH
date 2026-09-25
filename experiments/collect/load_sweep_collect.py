# -*- coding: utf-8 -*-
"""
QUET TAI CO KIEM SOAT SOCK SHOP + THU THAP TELEMETRY
====================================================
Sinh bo du lieu ma khong bo cong khai nao co: ten service THAT + bien do tai
THAT (xem docs/HE_THONG.md muc 3.1 va 5). RE2-SS giu tai phang (CV workload
~0.18, R^2 workload->CPU ~0.015) nen do chinh xac du bao khong do duoc.

Kien truc:
  * he thong: deploy/sockshop/docker-compose.yml (14 container, KHONG co user-sim)
  * do luong: container `ss-collector` (deploy/sockshop/collector/collector.py)
    chay CUNG mang docker, lay CPU tu bo dem cgroup + workload/latency tu
    /metrics THAT cua tung service. (Ban nhap cu dung `docker stats` + dem dong
    log lam workload: sai voi service Java/Go khong log tung request.)
  * tai: load generator MO (open-loop, Poisson) chay tren HOST, khong trong VM
    -> CPU cua loadgen khong nam trong CPU cua container duoc do. Muc tai =
    so request/giay MUC TIEU tai gateway (front-end).
  * thu tu muc tai duoc XAO NGAU NHIEN (seed co dinh, ghi vao meta) de muc tai
    khong bi lan voi troi theo thoi gian (JIT, cache Mongo, ram tang dan).

Dau ra (dung quy uoc cot cua repo, load_multi_service_data() doc duoc):
    <out>/level_<rps>/run<k>/simple_metrics.csv , inject_time.txt
    <out>/collector_full.csv   (toan bo, ke ca warmup/cooldown, de kiem toan)
    <out>/sweep_meta.json      (moi truong, phan cung, digest anh, lich, thong ke loadgen)

Dung:
    python experiments/load_sweep_collect.py --check
    python experiments/load_sweep_collect.py --levels 25,50,100,200,400 --hold 60 --warmup 20 \\
        --out-dir data/raw/SS-CALIBRATION          # tim diem bao hoa truoc
    python experiments/load_sweep_collect.py --levels 10,25,50,100,150,200 --hold 240 --repeats 2
    # ground truth tinh nang moi: baseline + 4 tinh nang, cung muc tai (so sanh truoc/sau)
    python experiments/load_sweep_collect.py --levels 10,25,50,100,150,250 --feature-scales 0.5,1,2 \\
        --features base,promo,recs,track,review
    # du lieu FORECASTING: chuoi tai lien tuc (mua vu/xu huong/doi che do/dot bien/troi co cau/rollout)
    python experiments/load_sweep_collect.py --traces --cooldown 45 --out-dir data/raw/SS-TRACES
    python experiments/forecast_data_audit.py --dir data/raw/SS-TRACES
    # KHA THI (khung chinh): tran CPU + tang tai theo bac -> diem gay; xem docs/DATA_FRAMEWORK.md
    python experiments/load_sweep_collect.py --ramp --limits RE2 --features base,promo,recs,track,review \\
        --feature-scales 1,2 --interval 3 --cooldown 60 --out-dir data/raw/SS-LIMITS
    python experiments/load_sweep_collect.py --report --out-dir data/raw/SS-LOADSWEEP
"""

import argparse
import asyncio
import base64
import http.cookiejar
import json
import os
import platform
import random
import re
import subprocess
import sys
import time

import hashlib
import math

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from request_router import SOCKSHOP_CALL_CHAINS  # noqa: E402  (nguon su that duy nhat cua taxonomy)
import forecast_traces as FT  # noqa: E402
DEPLOY_DIR = os.path.join(BASE_DIR, 'deploy', 'sockshop')
COMPOSE = os.path.join(DEPLOY_DIR, 'docker-compose.yml')
COLLECTOR_PY = os.path.join(DEPLOY_DIR, 'collector', 'collector.py')
DEFAULT_OUT = os.path.join(BASE_DIR, 'data', 'raw', 'SS-LOADSWEEP')

PROJECT = 'sockshop'
NETWORK = f'{PROJECT}_default'
COLLECTOR = 'ss-collector'
import os as _os  # noqa: E402  (chi de doc bien moi truong ngay tai diem khai bao, gan voi TARGET)
# Mac dinh qua host (127.0.0.1 -> Docker Desktop NAT). Da THU nghiem chay chinh loadgen nay
# BEN TRONG 1 container gan thang vao mang docker (bo NAT) de loai tru NAT la nguyen nhan dao
# dong diem gay; ket qua: NAT KHONG phai nguyen nhan (host tai 80 req/s: p99=67ms, dat SLO --
# container CUNG tai do: p99=29s, VI PHAM nang -- tu container tu gay ra loi gia con nang hon
# NAT rat nhieu). Da BO huong container hoa harness. SS_TARGET giu lai nhu 1 override chung
# (vd. tro toi 1 host khac), khong con dung de goi thang qua mang docker.
TARGET = _os.environ.get('SS_TARGET', 'http://127.0.0.1')
SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
VM_SATURATED = 0.85     # vm_cpu_util vuot nguong nay => mau bi nghi nhiem/bao hoa host


def run(args, timeout=120):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


# ------------------------------------------------------------------ moi truong
def check_env(verbose=True) -> bool:
    r = run(['docker', 'info', '--format', '{{.ServerVersion}}|{{.NCPU}}|{{.MemTotal}}|{{.Architecture}}'])
    if r.returncode != 0:
        print('  [X] docker daemon khong chay:', r.stderr.strip()[:200])
        return False
    ver, ncpu, mem, arch = r.stdout.strip().split('|')
    if verbose:
        print(f'  docker {ver} | {ncpu} vCPU | {int(mem) / 2**30:.1f} GiB | {arch}')
    r = run(['docker', 'ps', '--filter', f'label=com.docker.compose.project={PROJECT}',
             '--format', '{{.Label "com.docker.compose.service"}}'])
    up = sorted(r.stdout.split())
    if verbose:
        print(f'  {len(up)} container {PROJECT} dang chay')
    if len(up) < 14:
        print('  [X] thieu container -- chay: docker compose -f deploy/sockshop/docker-compose.yml up -d')
        return False
    try:
        import httpx
        r = httpx.get(TARGET + '/catalogue/size', timeout=10)
        if verbose:
            print(f'  GET /catalogue/size -> {r.status_code}')
        return r.status_code == 200
    except Exception as e:
        print('  [X] khong goi duoc gateway:', e)
        return False


def stack_fingerprint() -> dict:
    r = run(['docker', 'ps', '--filter', f'label=com.docker.compose.project={PROJECT}',
             '--format', '{{.Label "com.docker.compose.service"}}|{{.Image}}'])
    imgs = dict(l.split('|') for l in r.stdout.split())
    dig = run(['docker', 'image', 'inspect', '--format', '{{.RepoTags}} {{.Id}}'] + sorted(set(imgs.values()))).stdout
    return {'images': imgs, 'image_ids': [l.strip() for l in dig.splitlines()]}


def start_collector(out_dir, interval, fname) -> float:
    """Khoi dong collector, tra ve do lech dong ho (container - host) tinh bang giay."""
    run(['docker', 'rm', '-f', COLLECTOR])
    r = run(['docker', 'run', '-d', '--name', COLLECTOR, '--network', NETWORK,
             '--pid', 'host', '--cap-add', 'SYS_PTRACE',      # doc /proc/<pid>/net/sockstat -> _socket
             '-v', '/var/run/docker.sock:/var/run/docker.sock',
             '-v', f'{out_dir}:/out', '-v', f'{COLLECTOR_PY}:/collector.py:ro',
             'python:3.12-slim', 'python', '-u', '/collector.py',
             '--project', PROJECT, '--interval', str(interval), '--out', f'/out/{fname}'])
    if r.returncode != 0:
        raise RuntimeError('khong khoi dong duoc collector: ' + r.stderr)
    time.sleep(3)
    h0 = time.time()
    c = run(['docker', 'exec', COLLECTOR, 'python', '-c', 'import time;print(time.time())'])
    h1 = time.time()
    return float(c.stdout.strip()) - (h0 + h1) / 2


# Cot GROUND-TRUTH (them vao de lam dap an); moi cot con lai trong simple_metrics.csv la telemetry CHUAN
# (schema RE2-SS/RCAEval). Tien to gt_ de may kiem tra tach hai lop ma khong doan.
GT_COLS = {'load_level', 'feature', 'feature_pct', 'limits_cfg', 'split', 'family', 'series_id', 't_in_series',
           'warmup_flag', 'planned_rps', 'planned_feat_pct', 'planned_mix_browse', 'planned_mix_cart',
           'planned_mix_order', 'planned_mix_register', 'step_idx', 'target_rps', 't_in_step', 'step_warm',
           'step_violated', 'ramp_id'}


def to_gt(df):
    return df.rename(columns={c: f'gt_{c}' for c in df.columns if c in GT_COLS})


def from_gt(df):
    return df.rename(columns={c: c[3:] for c in df.columns if c.startswith('gt_')})


def write_aux(run_dir, out_dir, fname, lo, hi):
    """Cat 3 tep chuan RE2 cho khoang [lo, hi] (dong ho collector): raw -> metrics.csv, logs -> logs.csv,
    routes -> routes.csv. Thieu tep (collector cu) thi bo qua."""
    for prefix, name, tcol, scale in (('raw', 'metrics.csv', 'time', 1.0), ('logs', 'logs.csv', 'timestamp', 1e9),
                                      ('routes', 'routes.csv', 'time', 1.0)):
        src = os.path.join(out_dir, fname.replace('collector_', prefix + '_', 1))
        if not os.path.exists(src):
            continue
        d = pd.read_csv(src, on_bad_lines='skip')
        if tcol not in d or d.empty:
            continue
        t = pd.to_numeric(d[tcol], errors='coerce') / scale
        d = d[(t >= lo) & (t <= hi)]
        if name == 'metrics.csv':
            d = d.copy()
            d['time'] = d['time'].round().astype(int)
        d.to_csv(os.path.join(run_dir, name), index=False)


def _orders_mongo(js):
    r = run(['docker', 'exec', f'{PROJECT}-orders-db-1', 'mongo', '--quiet', '--eval', js])
    return r.stdout.strip() if r.returncode == 0 else None


def ensure_orders_index():
    """`orders` tim don theo customerId (promo/recs/track/GET /orders) nhung collection chi co
    index _id -> quet toan bo, chi phi TANG THEO SO DON (phien `order` them ~3-4 don/s o 100 req/s)
    va troi trong luc do. Them index nhu mot lap trinh vien trien khai tinh nang 'lich su don'
    se lam; ap dung cho CA baseline lan tinh nang de hai ben cung dieu kien."""
    _orders_mongo("db.getSiblingDB('data').customerOrder.createIndex({customerId: 1})")
    return _orders_mongo("db.getSiblingDB('data').customerOrder.getIndexes().length") == '2'


def orders_count():
    v = _orders_mongo("db.getSiblingDB('data').customerOrder.count()")
    return int(v) if v and v.isdigit() else None


def stop_collector():
    run(['docker', 'rm', '-f', COLLECTOR])


# ------------------------------------------------------------------ load generator
# (ten, trong so, so request toi front-end) -- dung de doi req/s muc tieu -> session/s.
SESSIONS = [('browse', 0.55, 4), ('cart', 0.25, 5), ('order', 0.15, 5), ('register', 0.05, 2)]
MEAN_REQ_PER_SESSION = sum(w * n for _, w, n in SESSIONS)
POOL_SIZE = 300         # >> so phien cart/order dong thoi (~10-50)
FEAT_POOL_SIZE = 100    # tai khoan rieng cho request tinh nang (chi doc, dung chung duoc)

# 4 tinh nang moi da cai vao front-end (deploy/sockshop/front-end/features/index.js).
# anchor = muc tang tai gateway ky vong (%) theo data/benchmark/parser_benchmark_prompts.json.
# Request tinh nang la request THEM VAO (Poisson song song), toc do = muc_tai * anchor%.
FEATURES = {
    'promo':  {'req': ('POST', '/promo'),           'anchor': 20, 'archetype': 'APPLY_PROMO_CODE'},
    'recs':   {'req': ('GET', '/recommendations'),  'anchor': 30, 'archetype': 'RECOMMEND_PRODUCTS'},
    'track':  {'req': ('GET', '/track'),            'anchor': 15, 'archetype': 'TRACK_PACKAGE'},
    'review': {'req': ('POST', '/reviews'),         'anchor': 10, 'archetype': 'WRITE_PRODUCT_REVIEW'},
    # tinh nang DOC LAP: giao dien HTTP do ben yeu cau quy dinh, backend nao duoc goi la quyet dinh cua nguoi cai
    'cartsum':  {'req': ('GET', '/cart/summary'),        'anchor': 10, 'archetype': 'VIEW_CART'},
    'quickadd': {'req': ('POST', '/cart/quick'),         'anchor': 15, 'archetype': 'ADD_TO_CART'},
    'express':  {'req': ('POST', '/checkout/express'),   'anchor': 25, 'archetype': 'PLACE_ORDER'},
    'browse':   {'req': ('GET', '/catalogue/browse?tags=sport'), 'anchor': 10, 'archetype': 'GET_CATALOGUE'},
}


class LoadGen:
    """Open-loop: phien den theo Poisson, KHONG cho phien truoc xong (khong ne tranh
    bao hoa nhu closed-loop). Cookie duoc quan ly thu cong vi moi phien la 1 nguoi dung."""

    def __init__(self, seed):
        import httpx
        self.httpx = httpx
        self.rng = random.Random(seed)
        self.items, self.cheap = [], []
        self.stats = None
        # nonce theo thoi gian: seed co dinh se sinh lai DUNG username cua lan chay truoc
        # (Mongo giu nguyen giua cac lan) -> `register` loi duplicate-key 500.
        self.nonce = f'{int(time.time()):x}'
        self.pool = asyncio.Queue()
        self.feat_pool = []
        self.n_reg = 0
        self.record_lat = False      # che do ramp: luu (t_xong, do_tre, la_tinh_nang, la_loi) tung request

    @staticmethod
    def _new_stats():
        return {'sent': 0, 'ok': 0, 'errors': 0, 'sessions': 0, 'inflight': 0, 'dropped': 0,
                'starved': 0, 'max_lag_ms': 0.0, 'err_kinds': {},
                'f_calls': 0, 'f_sent': 0, 'f_ok': 0, 'f_err': 0, 'e5xx': 0, 'lat': []}

    async def setup(self):
        r = await self.client.get('/catalogue?size=50')
        cat = r.json()
        self.items = [i['id'] for i in cat]
        # payment tu choi don > 100 USD (406) nen luong dat hang chi dung mon re
        self.cheap = [i['id'] for i in cat if i['price'] <= 50]
        # Pool tai khoan RIENG (co dia chi + the). Neu cac phien dong thoi dung chung 1 tai
        # khoan thi gio hang cong don -> vuot 100 USD -> 406, ty le nay TANG THEO TAI
        # (~9% o 400 req/s) = bien gay nhieu lan voi bien doc lap. Moi phien muon 1 tai khoan.
        sem = asyncio.Semaphore(20)

        async def mk(i):
            async with sem:
                u, jar = f'lg{self.nonce}p{i}', {}
                await self._req(jar, 'POST', '/register',
                                json={'username': u, 'password': 'pw123456', 'email': f'{u}@x.io'})
                await self._req(jar, 'GET', '/login', auth=(u, 'pw123456'))
                await self._req(jar, 'POST', '/addresses', json={
                    'street': 'Sesame', 'number': '1', 'country': 'US', 'city': 'NYC', 'postcode': '10001'})
                await self._req(jar, 'POST', '/cards', json={
                    'longNum': '5544154011345918', 'expires': '08/23', 'ccv': '123'})
                self.pool.put_nowait((u, 'pw123456'))
        await asyncio.gather(*[mk(i) for i in range(POOL_SIZE)])

    async def setup_feat(self):
        """Pool RIENG cho request tinh nang: da dang nhap (jar giu cookie), co dia chi/the,
        1 mon re trong gio va 1 don da dat -> ca 4 tinh nang chay day du va chi DOC, nen
        khong dung vao phien nen va khong can muon/tra tai khoan."""
        sem = asyncio.Semaphore(20)

        async def mk(i):
            async with sem:
                u, jar = f'lg{self.nonce}f{i}', {}
                await self._req(jar, 'POST', '/register',
                                json={'username': u, 'password': 'pw123456', 'email': f'{u}@x.io'})
                await self._req(jar, 'GET', '/login', auth=(u, 'pw123456'))
                await self._req(jar, 'POST', '/addresses', json={
                    'street': 'Sesame', 'number': '1', 'country': 'US', 'city': 'NYC', 'postcode': '10001'})
                await self._req(jar, 'POST', '/cards', json={
                    'longNum': '5544154011345918', 'expires': '08/23', 'ccv': '123'})
                item = self.cheap[i % len(self.cheap)]
                await self._req(jar, 'POST', '/cart', json={'id': item, 'quantity': 1})
                await self._req(jar, 'POST', '/orders', json={})     # co lich su don cho recs/track/promo
                await self._req(jar, 'POST', '/cart', json={'id': item, 'quantity': 1})   # don xoa gio -> them lai
                self.feat_pool.append(jar)
        await asyncio.gather(*[mk(i) for i in range(FEAT_POOL_SIZE)])

    async def feature_call(self, feat):
        st = self.stats
        st['inflight'] += 1
        try:
            method, url = FEATURES[feat]['req']
            kw = {}
            if feat == 'promo':
                kw['json'] = {'code': 'SAVE20'}
            elif feat == 'review':
                kw['json'] = {'productId': self.rng.choice(self.items),
                              'stars': self.rng.randint(1, 5), 'text': 'good socks'}
            elif feat == 'quickadd':
                kw['json'] = {'id': self.rng.choice(self.items)}
            elif feat == 'express':
                kw['json'] = {'id': self.rng.choice(self.cheap)}    # payment tu choi don > 100 USD
            await self._req(self.rng.choice(self.feat_pool), method, url, _f=True, **kw)
        finally:
            st['inflight'] -= 1

    async def _req(self, jar, method, url, **kw):
        st = self.stats
        f = kw.pop('_f', False)         # request cua tinh nang: dem rieng
        headers = kw.pop('headers', {})
        if jar:
            headers['Cookie'] = '; '.join(f'{k}={v}' for k, v in jar.items())
        st['sent'] += 1
        st['f_sent'] += f
        seg = '/' + url.lstrip('/').split('?')[0].split('/')[0]
        t_req = time.perf_counter()
        try:
            r = await self.client.request(method, url, headers=headers, **kw)
        except Exception as e:
            st['errors'] += 1
            st['e5xx'] += 1                                    # timeout/dut ket noi = loi voi nguoi dung
            k = f'{method} {seg} {type(e).__name__}'
            st['err_kinds'][k] = st['err_kinds'].get(k, 0) + 1
            st['f_err'] += f
            if self.record_lat:
                now = time.perf_counter()
                st['lat'].append((now, now - t_req, f, True))
            return None
        if self.record_lat:
            now = time.perf_counter()
            st['lat'].append((now, now - t_req, f, r.status_code >= 500))
        for c in r.headers.get_list('set-cookie'):
            m = re.match(r'([^=;\s]+)=([^;]*)', c)
            if m:
                jar[m.group(1)] = m.group(2)
        if r.status_code >= 400:
            st['errors'] += 1
            st['e5xx'] += r.status_code >= 500
            k = f'{method} {seg} {r.status_code}'
            st['err_kinds'][k] = st['err_kinds'].get(k, 0) + 1
            st['f_err'] += f
        else:
            st['ok'] += 1
            st['f_ok'] += f
        return r

    async def session(self, kind):
        st = self.stats
        st['inflight'] += 1
        jar, acct = {}, None
        try:
            pick = self.rng.choice
            if kind in ('cart', 'order'):
                try:
                    acct = self.pool.get_nowait()
                except asyncio.QueueEmpty:
                    st['starved'] += 1      # het tai khoan: ha xuong browse, co ghi nhan
                    kind = 'browse'
            if kind == 'browse':
                await self._req(jar, 'GET', '/')
                await self._req(jar, 'GET', '/catalogue?size=6')
                await self._req(jar, 'GET', f'/catalogue/{pick(self.items)}')
                await self._req(jar, 'GET', '/tags')
            elif kind == 'cart':
                await self._req(jar, 'GET', '/catalogue?size=6')
                await self._req(jar, 'GET', '/login', auth=acct)
                await self._req(jar, 'DELETE', '/cart')
                await self._req(jar, 'POST', '/cart', json={'id': pick(self.items), 'quantity': 1})
                await self._req(jar, 'GET', '/cart')
            elif kind == 'order':
                await self._req(jar, 'GET', '/catalogue?size=6')
                await self._req(jar, 'GET', '/login', auth=acct)
                await self._req(jar, 'DELETE', '/cart')
                await self._req(jar, 'POST', '/cart', json={'id': pick(self.cheap), 'quantity': 1})
                await self._req(jar, 'POST', '/orders', json={})
            else:  # register: ten duy nhat de khong va cham
                self.n_reg += 1
                u = f'lg{self.nonce}r{self.n_reg}'
                await self._req(jar, 'POST', '/register',
                                json={'username': u, 'password': 'pw123456', 'email': f'{u}@x.io'})
                await self._req(jar, 'GET', '/login', auth=(u, 'pw123456'))
        finally:
            st['inflight'] -= 1
            if acct:
                self.pool.put_nowait(acct)

    async def _arrivals(self, rate, seconds, spawn, tasks, counter, max_inflight):
        """Vong phat Poisson: goi spawn() moi lan co 1 'nguoi den'. Dung chung cho phien nen
        va request tinh nang -> hai dong tai doc lap, cung dong ho."""
        st = self.stats
        t0 = time.perf_counter()
        t_next = t0
        while True:
            t_next += self.rng.expovariate(rate)
            if t_next - t0 >= seconds:
                break
            delay = t_next - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            st['max_lag_ms'] = max(st['max_lag_ms'], (time.perf_counter() - t_next) * 1000)
            if st['inflight'] >= max_inflight:
                st['dropped'] += 1
                continue
            st[counter] += 1
            tasks.append(asyncio.create_task(spawn()))
            if len(tasks) > 5000:
                tasks[:] = [t for t in tasks if not t.done()]

    async def _arrivals_tv(self, rate_fn, duration, spawn, tasks, counter, max_inflight):
        """Poisson KHONG dong nhat: toc do rate_fn(t) (t = giay ke tu luc bat dau). Chinh xac khi
        rate_fn hang so tung doan 1 giay: gap > 1s thi bo qua den moc 1s ke tiep roi tinh lai
        (tinh khong nho cua phan phoi mu)."""
        st = self.stats
        t0 = time.perf_counter()
        t = 0.0
        while t < duration:
            lam = rate_fn(t)
            dt = self.rng.expovariate(lam) if lam > 1e-9 else 1e9
            if dt > 1.0:
                t = math.floor(t) + 1.0
                await asyncio.sleep(max(0.0, t0 + t - time.perf_counter()))
                continue
            t += dt
            if t >= duration:
                break
            await asyncio.sleep(max(0.0, t0 + t - time.perf_counter()))
            st['max_lag_ms'] = max(st['max_lag_ms'], (time.perf_counter() - t0 - t) * 1000)
            if st['inflight'] >= max_inflight:
                st['dropped'] += 1
                continue
            st[counter] += 1
            tasks.append(asyncio.create_task(spawn(t)))
            if len(tasks) > 5000:
                tasks[:] = [x for x in tasks if not x.done()]

    async def run_trace(self, tr, feature=None, anchor=0.0, need_feat=True, max_inflight=3000):
        """Phat mot chuoi tai bien thien (forecast_traces.make_trace). anchor: %tang cua tinh nang o
        feat_scale=1 (rollout: toc do tinh nang = rps * anchor * feat_scale / 100)."""
        httpx = self.httpx
        self.client = httpx.AsyncClient(
            base_url=TARGET, timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000))
        self.client.cookies.jar.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
        try:
            self.stats = self._new_stats()
            if not self.items:
                await self.setup()
            if need_feat and not self.feat_pool:
                await self.setup_feat()
            self.stats = self._new_stats()
            T = len(tr['rps'])
            sess_rate = tr['rps'] / (tr['mix'] @ np.array(FT.KIND_REQ, dtype=float))
            feat_rate = tr['rps'] * anchor * tr['feat_scale'] / 100.0
            idx = lambda t: min(int(t), T - 1)
            tasks, t0 = [], time.perf_counter()
            loops = [self._arrivals_tv(lambda t: sess_rate[idx(t)], T,
                                       lambda t: self.session(self.rng.choices(FT.KINDS, tr['mix'][idx(t)])[0]),
                                       tasks, 'sessions', max_inflight)]
            if feature and anchor > 0:
                loops.append(self._arrivals_tv(lambda t: feat_rate[idx(t)], T,
                                               lambda t: self.feature_call(feature),
                                               tasks, 'f_calls', max_inflight))
            await asyncio.gather(*loops)
            await asyncio.gather(*tasks)
            self.stats['elapsed_s'] = time.perf_counter() - t0
            return dict(self.stats)
        finally:
            await self.client.aclose()

    async def run_level(self, rps, seconds, feature=None, pct=0.0, need_feat=False, max_inflight=3000):
        """rps = tai NEN (req/s). feature/pct: them request tinh nang toc do rps*pct/100."""
        httpx = self.httpx
        self.client = httpx.AsyncClient(
            base_url=TARGET, timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000))
        self.client.cookies.jar.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
        try:
            self.stats = self._new_stats()
            if not self.items:
                await self.setup()
            if need_feat and not self.feat_pool:
                await self.setup_feat()
            self.stats = self._new_stats()      # request cua setup khong tinh vao muc tai
            kinds, weights = zip(*[(k, w) for k, w, _ in SESSIONS])
            tasks, t0 = [], time.perf_counter()
            self.stats['t0_perf'] = t0
            loops = [self._arrivals(rps / MEAN_REQ_PER_SESSION, seconds,
                                    lambda: self.session(self.rng.choices(kinds, weights)[0]),
                                    tasks, 'sessions', max_inflight)]
            if feature and pct > 0:
                loops.append(self._arrivals(rps * pct / 100.0, seconds,
                                            lambda: self.feature_call(feature),
                                            tasks, 'f_calls', max_inflight))
            await asyncio.gather(*loops)
            await asyncio.gather(*tasks)
            self.stats['elapsed_s'] = time.perf_counter() - t0
            return dict(self.stats)
        finally:
            await self.client.aclose()


# ------------------------------------------------------------------ dieu phoi
def host_cpu_sampler():
    import psutil
    psutil.cpu_percent(None)
    me = psutil.Process()
    me.cpu_percent(None)
    return lambda: (psutil.cpu_percent(None), me.cpu_percent(None))


def build_schedule(levels, feats, scales, repeats, seed, base_repeats=None):
    """(muc_tai, tinh_nang, cuong_do, lan). base chi 1 cuong do; tinh nang lap theo `scales`."""
    sched = [(lv, ft, sc, k + 1)
             for lv in levels for ft in feats for sc in ([1.0] if ft == 'base' else scales)
             for k in range((base_repeats or repeats) if ft == 'base' else repeats)]
    random.Random(seed).shuffle(sched)
    return sched


def run_name(feat, lv, scale):
    if feat == 'base':
        return f'level_{lv}'
    return f'{feat}_L{lv}_x{scale:g}'


def flush_runs(out_dir, log, offset, interval, fname, tag, done):
    """Cat + ghi cac luot DA XONG ma chua ghi. Goi sau cooldown (cua so cuoi da nam trong CSV)
    -> neu tien trinh bi ngat giua chung, du lieu cac luot truoc van con nguyen ven."""
    todo = [e for e in log if e['role'] == 'measure' and id(e) not in done]
    if not todo:
        return 0
    df = pd.read_csv(os.path.join(out_dir, fname), on_bad_lines='skip')   # dong cuoi co the dang duoc ghi
    for e in todo:
        lo, hi = e['t_measure_start'] + offset, e['t_end'] + offset
        # dong CSV la trung binh cua cua so (time-interval, time] -> chi lay cua so nam tron trong vung do
        part = df[(df['time'] - interval >= lo) & (df['time'] <= hi)].copy()
        part['time'] = part['time'].round().astype(int)
        part['load_level'] = e['level']
        part['feature'] = e['feature']
        part['feature_pct'] = e['feature_pct']
        part['limits_cfg'] = CURRENT_LIMITS['name']
        d = os.path.join(out_dir, run_name(e['feature'], e['level'], e['scale']), f"run{e['run']}")
        if os.path.exists(d):               # khong bao gio de len du lieu cu
            d += f'_{tag}'
        os.makedirs(d, exist_ok=True)
        to_gt(part).to_csv(os.path.join(d, 'simple_metrics.csv'), index=False)
        write_aux(d, out_dir, fname, lo, hi)
        with open(os.path.join(d, 'inject_time.txt'), 'w') as f:
            f.write(str(2 ** 31 - 1))   # khong tiem loi: toan bo la giai doan "binh thuong"
        e['rows'] = len(part)
        done.add(id(e))
    return len(todo)


def load_all(out_dir):
    frames = []
    for name in sorted(os.listdir(out_dir)):
        dp = os.path.join(out_dir, name)
        if not os.path.isdir(dp):
            continue
        for rn in sorted(os.listdir(dp)):
            p = os.path.join(dp, rn, 'simple_metrics.csv')
            if os.path.exists(p):
                d = from_gt(pd.read_csv(p))
                d['run'] = f'{name}/{rn}'
                if 'feature' not in d:
                    d['feature'], d['feature_pct'] = 'base', 0.0
                frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def feature_report(df, out_dir=None):
    """So sanh moi o (tinh nang, muc tai, cuong do) voi baseline CUNG muc tai: workload/CPU tang o
    service nao, chuoi DO DUOC co khop taxonomy khong, va chi phi CPU cho moi req/s them vao."""
    base = df[df['feature'] == 'base']
    rows = []
    for feat, meta in FEATURES.items():
        fd = df[df['feature'] == feat]
        for (lv, pct), f in fd.groupby(['load_level', 'feature_pct']):
            b = base[base['load_level'] == lv]
            if b.empty:
                continue
            f_rate = lv * pct / 100.0
            for sv in SERVICES:
                rows.append({
                    'feature': feat, 'level': lv, 'pct': pct, 'feat_rps': f_rate, 'service': sv,
                    'base_rps': b[f'{sv}_workload'].mean(), 'd_rps': f[f'{sv}_workload'].mean() - b[f'{sv}_workload'].mean(),
                    'base_cpu': b[f'{sv}_cpu'].mean(), 'd_cpu': f[f'{sv}_cpu'].mean() - b[f'{sv}_cpu'].mean(),
                    'n_base': len(b), 'n_feat': len(f)})
    if not rows:
        return
    eff = pd.DataFrame(rows)
    if out_dir:
        eff.to_csv(os.path.join(out_dir, 'feature_effects.csv'), index=False)
        print(f'\n  [OK] {len(eff)} dong hieu ung (tinh nang x muc x cuong do x service) -> feature_effects.csv')
    for feat, meta in FEATURES.items():
        e = eff[eff['feature'] == feat]
        if e.empty:
            continue
        tax = set(SOCKSHOP_CALL_CHAINS[meta['archetype']]['services'])
        cells = e.groupby(['level', 'pct'])
        khop = sum(set(c.loc[c['d_rps'] > 0.5 * c['feat_rps'], 'service']) == tax for _, c in cells)
        print(f"\n=== {feat} ({meta['archetype']}) taxonomy: {' -> '.join(SOCKSHOP_CALL_CHAINS[meta['archetype']]['services'])}"
              f" | chuoi do duoc KHOP o {khop}/{len(cells)} o (chi tin o o co feat_rps du lon) ===")
        hit = e[e['d_rps'] > 0.5 * e['feat_rps']].copy()
        hit['calls_per_feat'] = hit['d_rps'] / hit['feat_rps']     # so lan service bi goi cho moi lan dung tinh nang
        hit['cpu_per_rps'] = hit['d_cpu'] / hit['d_rps']           # %-core them vao cho moi req/s them vao
        g = hit.groupby('service')
        t = pd.DataFrame({'n': g.size(), 'calls_per_use': g['calls_per_feat'].median(),
                          'cpu_pt_per_rps': g['cpu_per_rps'].median()})
        print(t.round(3).to_string())


def report(out_dir):
    from scipy import stats
    df = load_all(out_dir)
    if df is None:
        print('[report] khong co du lieu trong', out_dir)
        return
    print(f'\n=== BAO CAO: {len(df)} mau, {df["run"].nunique()} run ===')
    sat = (df['vm_cpu_util'] > VM_SATURATED).mean()
    print(f'  vm_cpu_util: trung vi={df["vm_cpu_util"].median():.2f}  max={df["vm_cpu_util"].max():.2f}  '
          f'mau > {VM_SATURATED:.0%}: {sat:.1%}')
    df['target_rps'] = df['load_level'] * (1 + df['feature_pct'] / 100.0)
    grp = df.groupby(['feature', 'feature_pct', 'load_level'])
    g = grp['front-end_workload'].agg(['median', 'std'])
    g['achieved/target'] = g['median'] / grp['target_rps'].first()
    g['vm_cpu'] = grp['vm_cpu_util'].median()
    g['fe_lat50_ms'] = grp['front-end_latency-50'].median() * 1000
    g['fe_err/s'] = grp['front-end_error'].median()
    print('  workload thuc do duoc tai front-end (req/s), theo (tinh nang, %tang, muc nen):')
    print(g.round(3).to_string())
    bad = g.index[g['achieved/target'] < 0.9].tolist()
    if bad:
        print(f'  [CANH BAO] {bad}: workload dat < 90% muc tieu => load generator bi nghen, '
              f'KHONG phai gioi han cua he thong. Loai khoi phan tich.')
    base = df[df['feature'] == 'base']
    if base['load_level'].nunique() >= 3:
        print('\n  SIGNAL GATE (chi baseline)  R^2(workload -> CPU)  (RE2-SS cu: trung vi 0.015, 0% > 0.3)')
        rows = []
        for sv in SERVICES:
            m = base[[f'{sv}_workload', f'{sv}_cpu']].dropna()
            if len(m) < 10 or m.iloc[:, 0].std() == 0:
                continue
            lr = stats.linregress(m.iloc[:, 0], m.iloc[:, 1])
            rows.append((sv, len(m), lr.rvalue ** 2, lr.slope, m.iloc[:, 0].std() / m.iloc[:, 0].mean()))
        t = pd.DataFrame(rows, columns=['service', 'n', 'R2', 'cpu%_per_rps', 'workload_CV'])
        print(t.round(4).to_string(index=False))
        if len(t):
            print(f'\n  R^2 trung vi = {t["R2"].median():.3f} | {(t["R2"] > 0.3).mean():.0%} service co R^2 > 0.3 '
                  f'| workload CV trung vi = {t["workload_CV"].median():.2f} (RE2-SS ~0.18)')
    feature_report(df, out_dir)


LIMITS_FILE = os.path.join(DEPLOY_DIR, 'limits.json')
SLO_FILE = os.path.join(DEPLOY_DIR, 'slo.json')
CURRENT_LIMITS = {'name': 'none', 'cores': {}}


def apply_limits(name):
    """Dat tran CPU LIVE bang `docker update --cpus`, roi DOC LAI de xac minh (`docker inspect`).
    KHONG dung `--cpus 0` de bo tran: tren Docker nay 0 = "khong doi" (da kiem chung: lenh van tra ma 0 va tran cu
    van con). "Khong tran" = `--cpus <NCPU cua VM>`, tran khong bao gio cham toi; go han NanoCpus chi bang cach
    tao lai container."""
    with open(LIMITS_FILE, encoding='utf-8') as f:
        cfgs = json.load(f)['configs']
    if name not in cfgs:
        sys.exit(f'--limits {name}: khong co trong {LIMITS_FILE} ({sorted(cfgs)})')
    if '_INVALID' in cfgs[name]:
        sys.exit(f'--limits {name} bi danh dau KHONG HOP LE: {cfgs[name]["_INVALID"]}')
    ncpu = int(run(['docker', 'info', '--format', '{{.NCPU}}']).stdout.strip())
    for svc in sorted(set().union(*[{k for k in c if not k.startswith('_')} for c in cfgs.values()])):
        want = float(cfgs[name].get(svc, ncpu))
        r = run(['docker', 'update', '--cpus', str(want), f'{PROJECT}-{svc}-1'])
        if r.returncode != 0:
            sys.exit(f'docker update {svc} that bai: {r.stderr.strip()}')
        got = float(run(['docker', 'inspect', f'{PROJECT}-{svc}-1', '--format', '{{.HostConfig.NanoCpus}}']).stdout.strip() or 0) / 1e9
        if abs(got - want) > 1e-6:
            sys.exit(f'XAC MINH THAT BAI: {svc} yeu cau {want} core nhung thuc te {got} -- KHONG tiep tuc (du lieu se sai)')
    CURRENT_LIMITS.update(name=name, cores=cfgs[name], ncpu=ncpu)
    print(f'  tran CPU: {name} {cfgs[name] or f"(khong tran = {ncpu} core)"}  [da xac minh bang docker inspect]')


def load_slo(a):
    with open(SLO_FILE, encoding='utf-8') as f:
        slo = json.load(f)
    if a.slo_p99 is not None:
        slo['p99_s'] = a.slo_p99
    if a.slo_err is not None:
        slo['err_rate'] = a.slo_err
    return {'p99_s': float(slo['p99_s']), 'err_rate': float(slo['err_rate'])}


def eval_step(stats, dwell, warm_frac, slo, lg_cpu):
    """Danh gia SLO cho MOT bac tai, chi tren request BAT DAU sau `warm_frac` dau bac (bo chuyen tiep)."""
    lat = stats.pop('lat')
    cut = stats['t0_perf'] + warm_frac * dwell
    sel = [x for x in lat if x[0] - x[1] >= cut]
    out = {'n_eval': len(sel)}
    gen_limited = bool(stats['dropped'] > 0 and lg_cpu >= 85)
    if len(sel) < 20:
        out.update(p50=None, p99=None, err_rate=1.0, feat_p99=None, violated=True,
                   why=['no_service'], generator_limited=gen_limited)
        return out
    d = np.array([x[1] for x in sel])
    fd = np.array([x[1] for x in sel if x[2]])
    out.update(p50=float(np.percentile(d, 50)), p99=float(np.percentile(d, 99)),
               err_rate=float(np.mean([x[3] for x in sel])),
               feat_p99=float(np.percentile(fd, 99)) if len(fd) >= 10 else None)
    why = []
    if out['p99'] > slo['p99_s']:
        why.append('p99')
    if out['err_rate'] > slo['err_rate']:
        why.append('err')
    if stats['dropped'] and not gen_limited:
        why.append('dropped')          # he khong theo kip (inflight day) trong khi loadgen con du CPU
    out.update(violated=bool(why), why=why, generator_limited=gen_limited)
    return out


def flush_ramps(out_dir, log, offset, interval, fname, tag, done, dwell, warm_frac):
    todo = [e for e in log if e['role'] == 'ramp' and id(e) not in done]
    if not todo:
        return 0
    df = pd.read_csv(os.path.join(out_dir, fname), on_bad_lines='skip')
    for e in todo:
        mid = df['time'] - offset - interval / 2.0
        part = df[(mid >= e['t_start']) & (mid <= e['t_end'])].copy()
        mid = (part['time'] - offset - interval / 2.0).to_numpy()
        starts = np.array([st['t_start'] for st in e['steps']])
        idx = np.clip(np.searchsorted(starts, mid, side='right') - 1, 0, len(starts) - 1)
        part['ramp_id'] = e['id']
        part['feature'] = e['feature']
        part['feature_pct'] = e['feature_pct']
        part['limits_cfg'] = CURRENT_LIMITS['name']
        part['step_idx'] = idx
        part['target_rps'] = [e['steps'][i]['target_rps'] for i in idx]
        part['t_in_step'] = (mid - starts[idx]).round(2)
        part['step_warm'] = (part['t_in_step'] < warm_frac * dwell).astype(int)
        part['step_violated'] = [int(e['steps'][i]['violated']) for i in idx]
        part['time'] = part['time'].round().astype(int)
        d = os.path.join(out_dir, f"ramp_{e['id']}", f"run{e['run']}")
        if os.path.exists(d):
            d += f'_{tag}'
        os.makedirs(d, exist_ok=True)
        to_gt(part).to_csv(os.path.join(d, 'simple_metrics.csv'), index=False)
        write_aux(d, out_dir, fname, e['t_start'] + offset, e['t_end'] + offset + interval)
        with open(os.path.join(d, 'inject_time.txt'), 'w') as f:
            f.write(str(2 ** 31 - 1))
        with open(os.path.join(d, 'steps.json'), 'w', encoding='utf-8') as f:
            json.dump({'id': e['id'], 'feature': e['feature'], 'feature_pct': e['feature_pct'],
                       'limits': CURRENT_LIMITS, 'breakpoint': e['breakpoint'], 'steps': e['steps']},
                      f, indent=1, default=str)
        e['rows'] = len(part)
        done.add(id(e))
    return len(todo)


def breakpoint_from(steps):
    """Diem gay BEN VUNG voi nhieu thoang qua: la bac DAU cua chuoi vi pham lien tiep CUOI CUNG (ket thuc o bac
    cuoi). Mot bac vi pham roi bac sau lai dat SLO (GC, nhieu) KHONG phai diem gay -- duoc ghi rieng
    o `transient`. hi=None neu bac cuoi van dat SLO (chua gay trong dai tai)."""
    k = len(steps)
    while k > 0 and steps[k - 1]['violated']:
        k -= 1
    first = steps[k] if k < len(steps) else None
    return {'lo': steps[k - 1]['target_rps'] if k > 0 else None,
            'hi': first['target_rps'] if first else None,
            'why': first['why'] if first else None,
            'transient': [x['target_rps'] for x in steps[:k] if x['violated']]}


def run_ramp(a, out_dir):
    """Tang tai theo BAC den khi vi pham SLO -> diem gay R* cua moi (tinh nang, cuong do). Nhan kha thi
    cho mot tai dinh L la `L < R*_hi` (khoang [lo, hi] co do phan giai = buoc tai)."""
    slo = load_slo(a)
    feats = a.features.split(',')
    scales = [float(x) for x in a.feature_scales.split(',')]
    plan = [(ft, sc, k + 1) for ft in feats for sc in ([1.0] if ft == 'base' else scales)
            for k in range(a.repeats)]
    random.Random(a.seed).shuffle(plan)
    if a.ramp_levels:
        steps_rps = sorted({int(x) for x in a.ramp_levels.split(',') if x.strip()})
        if len(steps_rps) < 2:
            sys.exit('--ramp-levels can it nhat 2 bac')
        step_desc = 'viet tay: ' + ','.join(map(str, steps_rps))
    else:
        steps_rps = list(range(a.ramp_start, a.ramp_stop + 1, a.ramp_step))
        step_desc = f'buoc {a.ramp_step}'
    interval, warm_frac = a.interval, 0.4
    tag = time.strftime('%Y%m%d_%H%M%S')
    fname = f'collector_{tag}.csv'
    manifest_path = os.path.join(out_dir, f'ramp_manifest_{tag}.json')
    manifest = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'preregistered': True, 'slo': slo,
                'limits': CURRENT_LIMITS, 'steps_rps': steps_rps, 'warm_frac': warm_frac,
                'dwell_s': a.ramp_dwell, 'plan': [{'feature': f, 'scale': sc, 'run': k} for f, sc, k in plan],
                'args': vars(a),
                'code_sha256': {'load_sweep_collect.py': _sha(os.path.abspath(__file__))}, 'ramps': []}
    with open(manifest_path, 'w', encoding='utf-8') as f:      # PRE-REGISTRATION: truoc khi do
        json.dump(manifest, f, indent=1, default=str)
    est = len(plan) * (len(steps_rps) * (a.ramp_dwell + 5) * 0.6 + a.cooldown) / 60 + 3
    print(f'\n{len(plan)} ramp, buoc tai {steps_rps[0]}..{steps_rps[-1]} req/s ({step_desc}, giu {a.ramp_dwell}s), '
          f'SLO p99<={slo["p99_s"]}s loi<={slo["err_rate"]:.1%}, tran={CURRENT_LIMITS["name"]}; '
          f'uoc tinh ~{est:.0f} phut')
    print('  thu tu:', [f'{f}x{sc:g}#{k}' for f, sc, k in plan])
    offset = start_collector(out_dir, interval, fname)
    print(f'[collector] chay, lech dong ho = {offset * 1000:+.0f} ms')
    sample_host = host_cpu_sampler()
    lg = LoadGen(a.seed)
    lg.record_lat = True
    log, done = [], set()
    try:
        print('=== khoi dong nong 100 req/s x 90s (bi loai; tao pool tai khoan) ===')
        asyncio.run(lg.run_level(100, 90, need_feat=True))
        for ft, sc, k in plan:
            time.sleep(a.cooldown)
            flush_ramps(out_dir, log, offset, interval, fname, tag, done, a.ramp_dwell, warm_frac)
            pct = FEATURES[ft]['anchor'] * sc if ft != 'base' else 0.0
            e = {'role': 'ramp', 'id': 'base' if ft == 'base' else f'{ft}_x{sc:g}', 'run': k, 'feature': ft,
                 'scale': sc, 'feature_pct': pct, 't_start': time.time(), 'steps': []}
            print(f"=== ramp {e['id']} #{k}: tinh nang +{pct:g}% ===")
            consec = 0
            for i, rps in enumerate(steps_rps):
                sample_host()
                t_s = time.time()
                st = asyncio.run(lg.run_level(rps, a.ramp_dwell, feature=None if ft == 'base' else ft, pct=pct))
                h_cpu, lg_cpu = sample_host()
                ev = eval_step(st, a.ramp_dwell, warm_frac, slo, lg_cpu)
                ev.update(step=i, target_rps=rps, feat_rps=round(rps * pct / 100.0, 2), t_start=t_s,
                          t_end=time.time(), ok_per_s=round(st['ok'] / st['elapsed_s'], 1),
                          dropped=st['dropped'], lag_max_ms=round(st['max_lag_ms']),
                          host_cpu=h_cpu, loadgen_cpu=lg_cpu)
                e['steps'].append(ev)
                p99 = f"{ev['p99'] * 1000:.0f}ms" if ev['p99'] is not None else 'n/a'
                print(f"   {rps:4d} req/s (+{ev['feat_rps']:.0f} tinh nang): ok {ev['ok_per_s']}/s p99 {p99} "
                      f"loi {ev['err_rate']:.1%} loadgen_cpu {lg_cpu:.0f}% -> "
                      f"{'VI PHAM ' + ','.join(ev['why']) if ev['violated'] else 'dat SLO'}")
                if ev['generator_limited']:
                    print('   [CANH BAO] LOADGEN BAO HOA -> dung ramp; buoc nay khong do he thong')
                    break
                consec = consec + 1 if ev['violated'] else 0
                if consec > a.ramp_extra:      # vi pham LIEN TIEP -> da gay that, khong phai nhieu thoang qua
                    break
            e['t_end'] = time.time()
            e['breakpoint'] = breakpoint_from(e['steps'])
            log.append(e)
            b = e['breakpoint']
            print(f"   => diem gay: dat SLO den {b['lo']}, vi pham tu {b['hi']} req/s"
                  + (f" (nhieu thoang qua o {b['transient']})" if b['transient'] else ''))
    finally:
        time.sleep(interval + 1)
        stop_collector()
        flush_ramps(out_dir, log, offset, interval, fname, tag, done, a.ramp_dwell, warm_frac)
        manifest['ramps'] = [{k: v for k, v in e.items()} for e in log]
        manifest['stack'] = stack_fingerprint()
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=1, default=str)
    print(f'\n[OK] {len(done)} ramp -> {out_dir}')
    for e in log:
        b = e['breakpoint']
        print(f"  {e['id']:12s} #{e['run']}  dat SLO den {b['lo']}  vi pham tu {b['hi']} ({b['why']})")


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def flush_series(out_dir, log, offset, interval, fname, tag, done):
    """Ghi tung chuoi da xong: cot chuan repo + nhan chuoi + bien ngoai sinh `planned_*` (quy dao tai
    da biet truoc). KHONG loai warmup (dong luc hoc cua he thong la mot phan cua bai toan) -- chi gan co."""
    todo = [e for e in log if e['role'] == 'series' and id(e) not in done]
    if not todo:
        return 0
    df = pd.read_csv(os.path.join(out_dir, fname), on_bad_lines='skip')
    for e in todo:
        tr = e['_trace']
        T = len(tr['rps'])
        t_rel = df['time'] - offset - e['t_start']
        part = df[(t_rel - interval >= -1e-6) & (t_rel <= T + 1e-6)].copy()
        rel = (part['time'] - offset - e['t_start']).to_numpy()
        lo_i = np.clip(np.floor(rel - interval).astype(int), 0, T - 1)
        hi_i = np.clip(np.ceil(rel).astype(int), 1, T)
        cs = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in
              (('rps', tr['rps']), ('fscale', tr['feat_scale']))}
        n = np.maximum(hi_i - lo_i, 1)
        anchor = FEATURES[e['feature']]['anchor'] if e['feature'] else 0.0
        part['series_id'] = e['id']
        part['family'] = e['family']
        part['split'] = e['split']
        part['limits_cfg'] = CURRENT_LIMITS['name']
        part['feature'] = e['feature'] or 'base'
        part['t_in_series'] = rel.round(2)
        part['warmup_flag'] = (rel < 60).astype(int)
        part['planned_rps'] = (cs['rps'][hi_i] - cs['rps'][lo_i]) / n
        part['planned_feat_pct'] = anchor * (cs['fscale'][hi_i] - cs['fscale'][lo_i]) / n
        for j, k in enumerate(FT.KINDS):
            part[f'planned_mix_{k}'] = tr['mix'][np.clip((lo_i + hi_i) // 2, 0, T - 1), j]
        part['time'] = part['time'].round().astype(int)
        d = os.path.join(out_dir, f"trace_{e['id']}", 'run1')
        if os.path.exists(d):
            d += f'_{tag}'
        os.makedirs(d, exist_ok=True)
        to_gt(part).to_csv(os.path.join(d, 'simple_metrics.csv'), index=False)
        write_aux(d, out_dir, fname, e['t_start'] + offset, e['t_end'] + offset + interval)
        with open(os.path.join(d, 'inject_time.txt'), 'w') as f:
            f.write(str(2 ** 31 - 1))
        pd.DataFrame({'t': np.arange(T), 'rps': tr['rps'], 'feat_pct': tr['feat_scale'] * anchor,
                      **{f'mix_{k}': tr['mix'][:, j] for j, k in enumerate(FT.KINDS)}}
                     ).to_csv(os.path.join(d, 'planned_1s.csv'), index=False)
        e['rows'] = len(part)
        done.add(id(e))
    return len(todo)


def run_traces(a, out_dir):
    """Che do FORECASTING: chuoi tai lien tuc bien thien, ke hoach + phan chia train/val/test da
    pre-register (ghi vao manifest TRUOC khi do bat ky diem nao)."""
    if a.trace_plan:
        with open(a.trace_plan, encoding='utf-8') as f:
            plan = json.load(f)
    else:
        plan = FT.default_plan(a.trace_duration)
    order = list(range(len(plan)))
    random.Random(a.seed).shuffle(order)          # thu tu chay xao: split khong lan voi troi theo thoi gian
    interval = a.trace_interval
    tag = time.strftime('%Y%m%d_%H%M%S')
    fname = f'collector_{tag}.csv'
    manifest_path = os.path.join(out_dir, f'trace_manifest_{tag}.json')
    manifest = {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'preregistered': True, 'interval_s': interval,
        'args': vars(a), 'plan': plan, 'run_order': [plan[i]['id'] for i in order],
        'code_sha256': {'forecast_traces.py': _sha(os.path.join(BASE_DIR, 'experiments', 'forecast_traces.py')),
                        'load_sweep_collect.py': _sha(os.path.abspath(__file__))},
        'splits': {sp: [p['id'] for p in plan if p['split'] == sp] for sp in sorted({p['split'] for p in plan})},
        'series': []}
    with open(manifest_path, 'w', encoding='utf-8') as f:      # PRE-REGISTRATION: truoc khi do
        json.dump(manifest, f, indent=1, default=str)
    total = sum(p['duration'] + a.cooldown for p in plan) + 150
    print(f'\n{len(plan)} chuoi, ~{total / 60:.0f} phut, lay mau {interval}s. Manifest (pre-register): {manifest_path}')
    print('  thu tu chay:', manifest['run_order'])

    offset = start_collector(out_dir, interval, fname)
    print(f'[collector] chay, lech dong ho container-host = {offset * 1000:+.0f} ms')
    sample_host = host_cpu_sampler()
    lg = LoadGen(a.seed)
    log, done = [], set()
    try:
        print('=== khoi dong nong 100 req/s x 90s (bi loai; tao pool tai khoan) ===')
        asyncio.run(lg.run_level(100, 90, need_feat=True))
        for i in order:
            pl = plan[i]
            time.sleep(a.cooldown)
            flush_series(out_dir, log, offset, interval, fname, tag, done)
            tr = FT.make_trace(pl['family'], pl['duration'], pl['peak'], pl['seed'])
            anchor = FEATURES[pl['feature']]['anchor'] if pl['feature'] else 0.0
            print(f"=== chuoi {pl['id']} [{pl['family']}/{pl['split']}] {pl['duration']}s, "
                  f"tai {tr['rps'].min():.0f}-{tr['rps'].max():.0f} req/s ===")
            sample_host()
            t0 = time.time()
            stats = asyncio.run(lg.run_trace(tr, feature=pl['feature'], anchor=anchor))
            h_cpu, lg_cpu = sample_host()
            print(f"   gui {stats['sent']} req -> ok {stats['ok']} loi {stats['errors']} bi_bo {stats['dropped']} "
                  f"lag_max {stats['max_lag_ms']:.0f}ms | host_cpu {h_cpu:.0f}% loadgen_cpu {lg_cpu:.0f}%")
            if stats['dropped'] or stats['max_lag_ms'] > 500:
                print('   [CANH BAO] LOADGEN BAO HOA trong chuoi nay -> danh dau chuoi KHONG dang tin')
            log.append({'role': 'series', 'id': pl['id'], 'family': pl['family'], 'split': pl['split'],
                        'feature': pl['feature'], 't_start': t0, 't_end': time.time(), '_trace': tr,
                        'loadgen': stats, 'host_cpu_pct': h_cpu, 'loadgen_proc_cpu_pct': lg_cpu})
    finally:
        time.sleep(interval + 1)
        stop_collector()
        flush_series(out_dir, log, offset, interval, fname, tag, done)
        manifest['series'] = [{k: v for k, v in e.items() if k != '_trace'} for e in log]
        manifest['stack'] = stack_fingerprint()
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=1, default=str)
    print(f'\n[OK] {len(done)} chuoi -> {out_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--report', action='store_true', help='chi phan tich du lieu da co trong --out-dir')
    ap.add_argument('--levels', default='10,25,50,100,150,200', help='req/s tai NEN muc tieu tai gateway')
    ap.add_argument('--features', default='base',
                    help='base (khong tinh nang) va/hoac ' + ','.join(FEATURES) + ' -- nhan cheo voi --levels')
    ap.add_argument('--feature-scales', default='1',
                    help='cuong do tinh nang = anchor x scale (vd 0.5,1,2 = nua/dung/gap doi muc tang ky vong); '
                         'nhan cheo voi --levels de phu du vung nho->lon')
    ap.add_argument('--hold', type=int, default=300, help='giay MOI muc (gom ca warmup)')
    ap.add_argument('--warmup', type=int, default=60, help='giay dau moi muc bi loai khoi du lieu do')
    ap.add_argument('--cooldown', type=int, default=30, help='giay nghi giua 2 muc')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--base-repeats', type=int, default=None,
                    help='so lan lap RIENG cho baseline (mac dinh = --repeats). Moi hieu ung tinh nang la '
                         'hieu so voi baseline nen nhieu cua baseline lan vao moi o cung muc -> nen > repeats')
    ap.add_argument('--interval', type=int, default=5, help='cua so lay mau (giay)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--holdout', default='',
                    help='cac muc tai DANH RIENG de kiem chung tien cu (khong duoc fit); chi ghi vao meta')
    ap.add_argument('--limits', default='',
                    help='ten cau hinh tran CPU trong deploy/sockshop/limits.json (none = bo tran); de trong = khong doi')
    ap.add_argument('--ramp', action='store_true',
                    help='CHE DO KHA THI: tang tai theo bac den khi vi pham SLO -> diem gay (nhan kha thi)')
    ap.add_argument('--ramp-start', type=int, default=40)
    ap.add_argument('--ramp-stop', type=int, default=260)
    ap.add_argument('--ramp-step', type=int, default=20)
    ap.add_argument('--ramp-levels', default='',
                    help='cac bac tai req/s VIET TAY, phay-ngan -- DE len --ramp-start/stop/step. Dung cho luoi '
                         'HINH HOC (vd 40,50,65,80,100,125,155,195,240): buoc cap so cong cho do phan giai tuong doi '
                         '10%% o diem gay 200 nhung 50%% o diem gay 40, khien sai so tuong doi khong so duoc giua cac o.')
    ap.add_argument('--ramp-dwell', type=int, default=45, help='giay moi bac (40%% dau bi loai khi cham SLO)')
    ap.add_argument('--ramp-extra', type=int, default=1, help='so bac vi pham THEM sau bac vi pham dau tien')
    ap.add_argument('--slo-p99', type=float, default=None, help='ghi de p99 (giay) trong slo.json')
    ap.add_argument('--slo-err', type=float, default=None, help='ghi de ty le loi trong slo.json')
    ap.add_argument('--traces', action='store_true',
                    help='che do FORECASTING: chuoi tai lien tuc bien thien (xem forecast_traces.py)')
    ap.add_argument('--trace-duration', type=int, default=720, help='giay moi chuoi')
    ap.add_argument('--trace-interval', type=int, default=2, help='cua so lay mau (giay) cho che do chuoi')
    ap.add_argument('--trace-plan', default='', help='file JSON ke hoach chuoi (mac dinh: forecast_traces.default_plan)')
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out_dir)

    if a.report:
        return report(out_dir)
    feats = a.features.split(',')
    bad = [f for f in feats if f != 'base' and f not in FEATURES]
    if bad:
        sys.exit(f'--features khong hop le: {bad} (chon trong base,{",".join(FEATURES)})')
    print('=== kiem tra moi truong ===')
    if not check_env() or a.check:
        return
    if a.warmup >= a.hold and not a.traces:
        sys.exit('--warmup phai nho hon --hold')
    print('  index customerId tren orders-db:', 'OK' if ensure_orders_index() else 'KHONG TAO DUOC (chi phi se troi)')
    if a.limits:
        apply_limits(a.limits)
    if a.ramp:
        os.makedirs(out_dir, exist_ok=True)
        return run_ramp(a, out_dir)
    if a.traces:
        os.makedirs(out_dir, exist_ok=True)
        return run_traces(a, out_dir)

    levels = [int(x) for x in a.levels.split(',')]
    os.makedirs(out_dir, exist_ok=True)
    scales = [float(x) for x in a.feature_scales.split(',')]
    sched = build_schedule(levels, feats, scales, a.repeats, a.seed, a.base_repeats)
    total_min = (len(sched) * (a.hold + a.cooldown) + 150) / 60
    print(f'\n{len(sched)} luot, thu tu (xao, seed={a.seed}): {[f"{ft}@{lv}x{sc:g}#{k}" for lv, ft, sc, k in sched]}')
    print(f'uoc tinh ~{total_min:.0f} phut. Dung moi tac vu nang khac tren may trong luc do.\n')
    if a.holdout:
        print(f'[tien cu] muc tai gio lai (KHONG fit): {a.holdout}\n')

    tag = time.strftime('%Y%m%d_%H%M%S')
    fname = f'collector_{tag}.csv'
    offset = start_collector(out_dir, a.interval, fname)
    print(f'[collector] chay, lech dong ho container-host = {offset * 1000:+.0f} ms')
    sample_host = host_cpu_sampler()
    lg = LoadGen(a.seed)
    need_feat = any(f != 'base' for f in feats)
    log, done = [], set()
    fingerprint = stack_fingerprint()
    docker_info = run(['docker', 'info', '--format', '{{.ServerVersion}}|{{.NCPU}}|{{.MemTotal}}']).stdout.strip()

    def save_meta():
        with open(os.path.join(out_dir, f'sweep_meta_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'collector_file': fname,
                'args': vars(a), 'clock_offset_s': offset, 'mean_req_per_session': MEAN_REQ_PER_SESSION,
                'session_mix': SESSIONS, 'features': FEATURES, 'schedule': log, 'limits': dict(CURRENT_LIMITS),
                'host': {'platform': platform.platform(), 'machine': platform.machine(),
                         'processor': platform.processor(), 'python': sys.version.split()[0]},
                'stack': fingerprint, 'docker': docker_info,
            }, f, indent=1, default=str)
    try:
        # khoi dong nong: JIT/Mongo cache o muc trung vi, KHONG dua vao du lieu do.
        # (Cung la luc tao pool tai khoan tinh nang.)
        wl = sorted(levels)[len(levels) // 2]
        print(f'=== khoi dong nong {wl} req/s x 90s (bi loai) ===')
        t = time.time()
        asyncio.run(lg.run_level(wl, 90, need_feat=need_feat))
        log.append({'role': 'prewarm', 'level': wl, 't_start': t, 't_end': time.time()})

        for lv, ft, sc, k in sched:
            time.sleep(a.cooldown)
            if flush_runs(out_dir, log, offset, a.interval, fname, tag, done):
                save_meta()
            pct = FEATURES[ft]['anchor'] * sc if ft != 'base' else 0.0
            label = 'nen' if ft == 'base' else f'{ft} +{pct:g}%'
            print(f'=== muc {lv} req/s [{label}], lan {k}, giu {a.hold}s (warmup {a.warmup}s) ===')
            n_ord0 = orders_count()
            sample_host()
            t0 = time.time()
            stats = asyncio.run(lg.run_level(lv, a.hold, feature=None if ft == 'base' else ft, pct=pct))
            h_cpu, lg_cpu = sample_host()
            t1 = time.time()
            n_ord1 = orders_count()
            target = lv * (1 + pct / 100.0)
            ach = stats['ok'] / stats['elapsed_s']
            print(f"   gui {stats['sent']} req -> ok {stats['ok']} ({ach:.1f}/s, muc tieu {target:.0f}) loi {stats['errors']} "
                  f"bi_bo {stats['dropped']} het_tk {stats['starved']} lag_max {stats['max_lag_ms']:.0f}ms | "
                  f"host_cpu {h_cpu:.0f}% loadgen_cpu {lg_cpu:.0f}%")
            if ft != 'base':
                print(f"   tinh nang: {stats['f_calls']} lan goi, ok {stats['f_ok']}, loi {stats['f_err']} "
                      f"({stats['f_ok'] / stats['elapsed_s']:.1f}/s, muc tieu {lv * pct / 100:.1f}/s)")
            if stats['err_kinds']:
                top = sorted(stats['err_kinds'].items(), key=lambda kv: -kv[1])[:4]
                print('   loi:', ', '.join(f'{kk}={v}' for kk, v in top))
            if ach < 0.9 * target or stats['dropped']:
                print(f'   [CANH BAO] LOADGEN BAO HOA: dat {ach:.0f}/s < 90% muc tieu {target:.0f} -> muc nay KHONG do he thong')
            log.append({'role': 'measure', 'level': lv, 'feature': ft, 'feature_pct': pct, 'scale': sc, 'run': k,
                        't_start': t0, 't_measure_start': t0 + a.warmup, 't_end': t1,
                        'orders_in_db': [n_ord0, n_ord1],
                        'loadgen': stats, 'host_cpu_pct': h_cpu, 'loadgen_proc_cpu_pct': lg_cpu})
    finally:
        time.sleep(a.interval + 1)
        stop_collector()

    n = flush_runs(out_dir, log, offset, a.interval, fname, tag, done)
    save_meta()
    print(f'\n[OK] {len(done)} run -> {out_dir}')
    report(out_dir)


if __name__ == '__main__':
    main()
