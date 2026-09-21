# -*- coding: utf-8 -*-
"""
KIEM TRA HOP DONG DU LIEU (docs/DATA_FRAMEWORK.md muc 4 va 6)
==============================================================
Cuong che khung du lieu thay vi chi la van ban. Chay tren tung goc du lieu:

    python experiments/data_contract_check.py --dir data/raw/SS-TRAIN  --role train
    python experiments/data_contract_check.py --dir data/raw/SS-LIMITS --role limits

Kiem tra:
  * cot bat buoc cho 5 node chinh (workload, cpu) va cot dinh danh; khong thieu/NaN qua muc
  * don vi CPU hop ly (% cua 1 core): CPU/(req/s) trong khoang vat ly
  * role=train : KHONG co dong tinh nang (feature != base), KHONG co tran (limits_cfg != none),
                 va moi khoi chua bao hoa (khong co step_violated)
  * role=limits: moi ramp co steps.json + diem gay, manifest pre-register, tran va SLO duoc ghi lai;
                 liet ke cac ramp thuoc tap KHOA (khong duoc dung khi chinh mo hinh)
Thoat ma khac 0 neu co FAIL.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

MAIN = ['front-end', 'catalogue', 'user', 'carts', 'orders']
APP7 = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
STD_SUFFIX = {'cpu', 'mem', 'socket', 'diskio', 'workload', 'error', 'latency-50', 'latency-90', 'latency-95',
              'latency-99', 'latency-mean'}                    # RE2-SS co cpu,mem,socket,diskio,workload,error,latency-50/90
ALLOWED_EXTRA = {'time', 'vm_cpu_util', 'vm_mem_avail_mb'}     # bien moi truong, khong phai ground truth
RE2_LOG_COLS = ['time', 'timestamp', 'container_name', 'message', 'level', 'req_path', 'error']
RAW_MUST = ['container-cpu-usage-seconds-total', 'container-memory-working-set-bytes', 'container-sockets',
            'container-spec-cpu-quota']
ACCESS_RE = r'^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+\S+\s+\d{3}\s+[\d.]+\s*ms'
LOCKED_FEATURES = ('track', 'review')          # tap kiem tra KHOA (docs/DATA_FRAMEWORK.md muc 6)
CPU_PER_RPS = (0.003, 3.0)                     # %-core tren req/s: rong de bat loi don vi 100x, khong bat sai so nho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--role', choices=['train', 'limits'], required=True)
    a = ap.parse_args()
    fails, warns = [], []

    def ok(msg):
        print(f'  [PASS] {msg}')

    def fail(msg):
        fails.append(msg)
        print(f'  [FAIL] {msg}')

    def warn(msg):
        warns.append(msg)
        print(f'  [WARN] {msg}')

    files = sorted(glob.glob(os.path.join(a.dir, '*', '*', 'simple_metrics.csv')))
    if not files:
        sys.exit(f'khong co simple_metrics.csv trong {a.dir}')
    frames = []
    for p in files:
        d = pd.read_csv(p)
        d = d.rename(columns={c: c[3:] for c in d.columns if c.startswith('gt_')})   # doc ca cot ground-truth gt_*
        d['_run'] = os.path.relpath(os.path.dirname(p), a.dir).replace(os.sep, '/')
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    print(f'\n=== HOP DONG DU LIEU {a.dir} | role={a.role} | {len(df)} hang, {len(files)} run ===')

    # ---- cot
    need = [f'{s}_{m}' for s in MAIN for m in ('workload', 'cpu')] + ['time']
    miss = [c for c in need if c not in df]
    (fail if miss else ok)(f'thieu cot bat buoc {miss}' if miss else 'du cot workload/cpu cho 5 node chinh')
    if miss:
        return _end(fails, warns)
    nan = df[need].isna().mean().max()
    (warn if nan > 0.01 else ok)(f'NaN lon nhat cua cot bat buoc = {nan:.2%}')

    # ---- don vi CPU
    bad = []
    for s in MAIN:
        w = df[f'{s}_workload'].replace(0, np.nan)
        r = float((df[f'{s}_cpu'] / w).median())
        if not (CPU_PER_RPS[0] <= r <= CPU_PER_RPS[1]):
            bad.append((s, round(r, 4)))
    (fail if bad else ok)(f'CPU/(req/s) ngoai khoang vat ly cho %-core {CPU_PER_RPS}: {bad}' if bad
                          else 'don vi CPU hop ly (% cua 1 core)')

    feature = df['feature'] if 'feature' in df else pd.Series('base', index=df.index)
    lim = df['limits_cfg'] if 'limits_cfg' in df else pd.Series('none', index=df.index)

    if a.role == 'train':
        nf = int((feature != 'base').sum())
        (fail if nf else ok)(f'{nf} dong co luu luong tinh nang trong tap TRAIN (ro ri)' if nf
                             else 'tap train khong co luu luong tinh nang')
        nl = int((lim != 'none').sum())
        (fail if nl else ok)(f'{nl} dong chay duoi tran CPU trong tap TRAIN' if nl
                             else 'tap train khong dat tran CPU')
        if 'step_violated' in df and df['step_violated'].sum():
            fail('tap train chua bac vi pham SLO (khong con la chua bao hoa)')
        if 'vm_cpu_util' in df:
            sat = float((df['vm_cpu_util'] > 0.85).mean())
            (warn if sat > 0.05 else ok)(f'vm_cpu_util > 85%: {sat:.1%}')
    else:
        runs = sorted({os.path.dirname(f) for f in files})
        no_steps = [r for r in runs if not os.path.exists(os.path.join(r, 'steps.json'))]
        (fail if no_steps else ok)(f'ramp thieu steps.json: {[os.path.basename(os.path.dirname(r)) for r in no_steps]}'
                                   if no_steps else 'moi ramp co steps.json (diem gay + SLO tung bac)')
        man = glob.glob(os.path.join(a.dir, 'ramp_manifest_*.json'))
        (fail if not man else ok)('khong co ramp_manifest (pre-register)' if not man
                                  else f'co {len(man)} manifest pre-register')
        for m in man:
            j = json.load(open(m, encoding='utf-8'))
            if not j.get('slo') or not j.get('limits', {}).get('name'):
                fail(f'{os.path.basename(m)} thieu SLO hoac cau hinh tran')
        unbroken = []
        for r in runs:
            sp = os.path.join(r, 'steps.json')
            if os.path.exists(sp):
                bp = json.load(open(sp, encoding='utf-8')).get('breakpoint', {})
                if bp.get('hi') is None:
                    unbroken.append(os.path.basename(os.path.dirname(r)))
        (warn if unbroken else ok)(f'ramp KHONG co diem gay trong dai tai (tang --ramp-stop hoac siet tran): {unbroken}'
                                   if unbroken else 'moi ramp deu co diem gay')
        locked = sorted({f for f in feature.unique() if f in LOCKED_FEATURES})
        print(f'  [INFO] tap KHOA (khong dung khi chinh mo hinh): {locked or "khong co"}')

    check_standard(a, files, ok, fail, warn)
    _end(fails, warns)


def check_standard(a, files, ok, fail, warn):
    """Lop TELEMETRY CHUAN (giong RE2-SS/RCAEval): tep, cot, log, gioi han CPU; lop ground-truth chi o cot gt_*."""
    import re
    print('\n  --- tuan thu schema chuan (RE2-SS) ---')
    raw_cols = [c for c in pd.read_csv(files[0], nrows=0).columns]
    # 1) moi cot khong-gt phai la telemetry chuan hoac bien moi truong; cot ground-truth phai co tien to gt_
    odd = []
    for c in raw_cols:
        if c in ALLOWED_EXTRA or c.startswith('gt_'):
            continue
        svc, _, suf = c.partition('_')
        if suf not in STD_SUFFIX:
            odd.append(c)
    (fail if odd else ok)(f'cot la (khong chuan, khong gt_): {odd[:8]}' if odd
                          else 'moi cot la telemetry chuan hoac gt_* (lop ground-truth tach roi)')
    std_missing = [f'{s}_{m}' for s in APP7 for m in ('cpu', 'mem', 'workload') if f'{s}_{m}' not in raw_cols]
    (warn if std_missing else ok)(f'thieu cot chuan {std_missing[:6]}' if std_missing else 'du cpu/mem/workload cho 7 service')
    sock = [f'{s}_socket' for s in APP7 if f'{s}_socket' not in raw_cols]
    (warn if sock else ok)(f'thieu {len(sock)} cot socket (repo liet ke Socket la metric loi)' if sock else 'co cot socket')
    # 2) tep dong hanh: metrics.csv (raw cAdvisor), logs.csv (schema RE2), routes.csv
    runs = sorted({os.path.dirname(f) for f in files})
    for name in ('metrics.csv', 'logs.csv', 'routes.csv'):
        miss = [os.path.basename(r) for r in runs if not os.path.exists(os.path.join(r, name))]
        (warn if miss else ok)(f'{len(miss)}/{len(runs)} run thieu {name}' if miss else f'moi run co {name}')
    r0 = runs[0]
    mp, lp = os.path.join(r0, 'metrics.csv'), os.path.join(r0, 'logs.csv')
    if os.path.exists(mp):
        mc = pd.read_csv(mp, nrows=2)
        need = [f'{s}_{n}' for s in MAIN for n in RAW_MUST if f'{s}_{n}' not in mc.columns]
        (fail if need else ok)(f'metrics.csv thieu {need[:5]}' if need
                               else 'metrics.csv co counter cAdvisor + spec-cpu-quota (gioi han CPU) cho 5 node chinh')
        q = mc[[f'{s}_container-spec-cpu-quota' for s in MAIN if f'{s}_container-spec-cpu-quota' in mc]]
        print(f'  [INFO] spec-cpu-quota (RE2-SS: 50000 = 0.5 core): {q.iloc[-1].to_dict()}')
    if os.path.exists(lp):
        lg = pd.read_csv(lp, nrows=2000)
        (fail if list(lg.columns) != RE2_LOG_COLS else ok)(f'logs.csv sai schema: {list(lg.columns)}' if list(lg.columns) != RE2_LOG_COLS
                                                          else 'logs.csv dung schema RE2 (time,timestamp,container_name,message,...)')
        fe = lg[lg['container_name'] == 'front-end']['message'].dropna().astype(str)
        acc = fe.map(lambda m: bool(re.match(ACCESS_RE, m))).mean() if len(fe) else float('nan')
        ansi = fe.str.contains('\x1b', regex=False).mean() if len(fe) else 0
        # RE2-SS giu MOI dong log (access-log chi ~23%): dieu kien la CO access-log doc duoc va KHONG con ma ANSI
        (fail if (len(fe) and (acc == 0 or ansi > 0)) else ok)(
            f'front-end: {acc:.0%} dong la access-log doc duoc, {ansi:.0%} con ma ANSI' if len(fe) else 'khong co log front-end de kiem')


def _end(fails, warns):
    print(f'\n=== {len(fails)} FAIL, {len(warns)} WARN ===')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
