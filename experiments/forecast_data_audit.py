# -*- coding: utf-8 -*-
"""
KIEM DINH DU LIEU THEO CHUAN FORECASTING
========================================
Chay tren mot thu muc du lieu (SS-TRACES: chuoi lien tuc; hoac SS-LOADSWEEP: khoi on dinh) va
tra PASS/WARN/FAIL cho tung tieu chi, kem bang ket qua THAM CHIEU (skill score theo giao thuc cua
repo: 1 - MSE_model/MSE_hang_so) de moi nguoi dung bo du lieu biet ro "thang diem" ban dau.

Tieu chi (moi cai co ly do de KHONG bao cao ket qua dep tren du lieu yeu):
  A  Deu dan      : khoang cach thoi gian == chu ky lay mau, khong khuyet lo (chuoi lien tuc)
  B  Day du       : ty le NaN cua workload/cpu
  C  Co mau       : so chuoi/so hang moi split; co test_ood
  D  Khong tam thuong: persistence o horizon dai KHONG giai xong bai toan (chi voi chuoi lien tuc)
  E  Co tin hieu  : R^2 workload->CPU tren train, VA negative control (hoan vi workload) ~ 0
                    => target nao dat moi xung dang lam muc tieu du bao (memory thuong khong dat)
  F  Phu OOD      : dinh workload cua test_ood vuot han train (mac dinh >= 1.3x)
  G  Khong ro ri  : split tach roi theo chuoi; thoi gian don dieu
  H  Khong nhiem  : vm_cpu_util cao / loadgen bao hoa (doc tu manifest)

Split: chuoi co cot `split` (train/val/test_in/test_ood/test_feature) dung nguyen; khoi on dinh
(khong co cot `split`) chia theo muc tai: <= --ood-above la train/test_in (70/30 theo run), con lai
test_ood.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
GATE_R2, NEG_R2_MAX, OOD_RATIO_MIN, VM_SAT, MIN_ROWS_TEST = 0.30, 0.05, 1.3, 0.85, 100


def load(d):
    frames = []
    for p in sorted(glob.glob(os.path.join(d, '*', '*', 'simple_metrics.csv'))):
        df = pd.read_csv(p)
        rel = os.path.relpath(os.path.dirname(p), d).replace(os.sep, '/')
        if 'series_id' not in df:
            df['series_id'] = rel
        df['_path'] = rel
        frames.append(df)
    if not frames:
        sys.exit(f'khong co simple_metrics.csv trong {d}')
    return pd.concat(frames, ignore_index=True)


def skill(y, yhat, y_train_mean):
    den = np.mean((y - y_train_mean) ** 2)
    return float('nan') if den <= 0 else 1.0 - np.mean((y - yhat) ** 2) / den


def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    return lambda z: a + b * z


class R:
    """Gom ket qua tieu chi."""
    def __init__(self):
        self.rows = []

    def add(self, code, name, status, msg):
        self.rows.append((code, name, status, msg))
        print(f'  [{status:4s}] {code} {name}: {msg}')

    def summary(self):
        c = pd.Series([r[2] for r in self.rows]).value_counts().to_dict()
        print(f'\n=== TONG KET: {c} ===')
        return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--interval', type=float, default=None, help='chu ky lay mau (giay); mac dinh doc tu du lieu')
    ap.add_argument('--ood-above', type=float, default=150, help='khoi on dinh: muc tai > gia tri nay la test_ood')
    ap.add_argument('--keep-warmup', action='store_true', help='giu hang warmup_flag=1 khi danh gia')
    a = ap.parse_args()
    df = load(a.dir)
    res = R()
    is_trace = 'split' in df.columns and df['family'].notna().any() if 'family' in df else False
    print(f'\n=== KIEM DINH {a.dir} | {len(df)} hang, {df["series_id"].nunique()} chuoi, '
          f'loai: {"chuoi lien tuc" if is_trace else "khoi on dinh"} ===')

    # ---- gan split
    if not is_trace:
        rng = np.random.RandomState(42)
        runs = df['series_id'].unique()
        lvl = df.groupby('series_id')['load_level'].first()
        tr_runs = [r for r in runs if lvl[r] <= a.ood_above]
        test_in = set(rng.choice(tr_runs, max(1, int(0.3 * len(tr_runs))), replace=False)) if tr_runs else set()
        df['split'] = df['series_id'].map(lambda r: 'test_ood' if lvl[r] > a.ood_above
                                          else ('test_in' if r in test_in else 'train'))
    ev = df if (a.keep_warmup or 'warmup_flag' not in df) else df[df['warmup_flag'] == 0]

    # ---- A: deu dan
    per = []
    for sid, g in df.groupby('series_id'):
        dt = np.diff(g['time'].to_numpy(dtype=float))
        per.append((sid, np.median(dt) if len(dt) else np.nan, (dt <= 0).sum(), len(g)))
    pt = pd.DataFrame(per, columns=['sid', 'dt', 'nonmono', 'n'])
    interval = a.interval or float(pt['dt'].median())
    if is_trace:
        gap = 0
        for sid, g in df.groupby('series_id'):
            dt = np.diff(g['time'].to_numpy(dtype=float))
            gap += (dt > 2.5 * interval).sum()      # 'time' lam tron giay -> cho phep sai so
        frac = gap / max(len(df) - df['series_id'].nunique(), 1)
        res.add('A', 'deu dan', 'PASS' if frac < 0.01 else 'FAIL',
                f'chu ky trung vi {interval:g}s; khuyet lo {frac:.2%} (nguong 1%)')
    else:
        res.add('A', 'deu dan', 'WARN', f'khoi on dinh (chu ky {interval:g}s trong khoi): KHONG phai chuoi lien tuc, '
                'khong dung cho forecasting theo thoi gian -- chi cho mapping tai->tai nguyen')

    # ---- B: day du
    nan = ev[['front-end_workload', 'front-end_cpu'] + [f'{s}_cpu' for s in SERVICES if f'{s}_cpu' in ev]].isna().mean().max()
    res.add('B', 'day du', 'PASS' if nan < 0.01 else 'WARN', f'NaN lon nhat cua workload/cpu = {nan:.2%}')

    # ---- C: co mau
    cnt = ev.groupby('split').agg(hang=('time', 'size'), chuoi=('series_id', 'nunique'))
    print(cnt.to_string())
    need = {'train', 'test_ood'}
    miss = need - set(cnt.index)
    small = [s for s in ('test_in', 'test_ood') if s in cnt.index and cnt.loc[s, 'hang'] < MIN_ROWS_TEST]
    res.add('C', 'co mau', 'FAIL' if miss else ('WARN' if small else 'PASS'),
            f'thieu split {sorted(miss)}' if miss else (f'split < {MIN_ROWS_TEST} hang: {small}' if small else 'du'))

    tr, ood = ev[ev['split'] == 'train'], ev[ev['split'] == 'test_ood']

    # ---- F: phu OOD
    if len(tr) and len(ood):
        ratio = ood['front-end_workload'].quantile(0.99) / tr['front-end_workload'].quantile(0.99)
        res.add('F', 'phu OOD', 'PASS' if ratio >= OOD_RATIO_MIN else 'FAIL',
                f'P99 workload test_ood / train = {ratio:.2f}x (can >= {OOD_RATIO_MIN}x)')

    # ---- G: ro ri
    sets = {sp: set(g['series_id']) for sp, g in df.groupby('split')}
    inter = [(x, y) for x in sets for y in sets if x < y and sets[x] & sets[y]]
    res.add('G', 'khong ro ri', 'PASS' if not inter and not pt['nonmono'].sum() else 'FAIL',
            f'giao nhau giua split: {inter}' if inter else f'split tach roi theo chuoi; thoi gian don dieu (vi pham {int(pt["nonmono"].sum())})')

    # ---- H: nhiem
    sat = (df['vm_cpu_util'] > VM_SAT).mean() if 'vm_cpu_util' in df else float('nan')
    bad_series = []
    for m in glob.glob(os.path.join(a.dir, 'trace_manifest_*.json')) + glob.glob(os.path.join(a.dir, 'sweep_meta_*.json')):
        for e in json.load(open(m, encoding='utf-8')).get('series', []) or json.load(open(m, encoding='utf-8')).get('schedule', []):
            lgs = e.get('loadgen') or {}
            if lgs.get('dropped') or lgs.get('max_lag_ms', 0) > 500:
                bad_series.append(e.get('id') or f"{e.get('feature')}@{e.get('level')}")
    res.add('H', 'khong nhiem', 'PASS' if (sat < 0.05 and not bad_series) else 'WARN',
            f'vm_cpu_util>{VM_SAT:.0%}: {sat:.1%}; chuoi/luot bi loadgen bao hoa: {bad_series or "khong"}')

    # ---- E + reference baselines (per service, target = cpu)
    rows, hor = [], {}
    for s in SERVICES:
        w, c = f'{s}_workload', f'{s}_cpu'
        if w not in ev or c not in ev:
            continue
        t = tr[[w, c]].dropna()
        if len(t) < 30 or t[w].std() == 0:
            continue
        f = ols(t[w].to_numpy(), t[c].to_numpy())
        r2 = 1 - np.mean((t[c] - f(t[w])) ** 2) / np.var(t[c])
        # negative control: hoan vi workload trong train, fit lai, danh gia tren train hoan vi
        perm = np.random.RandomState(0).permutation(t[w].to_numpy())
        fp = ols(perm, t[c].to_numpy())
        r2n = 1 - np.mean((t[c] - fp(perm)) ** 2) / np.var(t[c])
        mu = t[c].mean()
        row = {'service': s, 'R2_train': r2, 'R2_negctl': r2n,
               'mem_R2': np.nan}
        for sp in ('test_in', 'test_ood'):
            e = ev[ev['split'] == sp][[w, c]].dropna()
            row[f'skill_linear_{sp}'] = skill(e[c].to_numpy(), f(e[w].to_numpy()), mu) if len(e) > 5 else np.nan
        # persistence 1 buoc (chi co nghia voi chuoi lien tuc) tren toan bo test
        rows.append(row)
    ref = pd.DataFrame(rows)
    if len(ref):
        ok = ref[(ref['R2_train'] >= GATE_R2) & (ref['R2_negctl'] <= NEG_R2_MAX)]
        res.add('E', 'co tin hieu (cpu)', 'PASS' if len(ok) >= len(ref) * 0.7 else ('WARN' if len(ok) else 'FAIL'),
                f'{len(ok)}/{len(ref)} service dat cong R^2>={GATE_R2} va negative control<={NEG_R2_MAX}')
    # memory: bao ro de khong ai chon lam muc tieu du bao
    memr = []
    for s in SERVICES:
        w, m = f'{s}_workload', f'{s}_mem'
        if w in tr and m in tr:
            t = tr[[w, m]].dropna()
            if len(t) > 30 and t[w].std() > 0 and t[m].std() > 0:
                memr.append((s, np.corrcoef(t[w], t[m])[0, 1] ** 2))
    if memr:
        good = [s for s, v in memr if v >= GATE_R2]
        res.add('E', 'co tin hieu (mem)', 'WARN' if not good else 'PASS',
                f'R^2 workload->mem >= {GATE_R2}: {good or "khong service nao"} (mem thuong khong dang lam muc tieu du bao tai)')

    # ---- D: khong tam thuong (persistence theo horizon)
    if is_trace:
        out = []
        for h_s in (10, 30, 60, 120):
            h = max(int(round(h_s / interval)), 1)
            sk = []
            for s in SERVICES:
                c = f'{s}_cpu'
                if c not in ev:
                    continue
                mu = tr[c].dropna().mean()
                num = den = 0.0
                for _, g in ev[ev['split'].isin(['test_in', 'test_ood', 'val'])].groupby('series_id'):
                    y = g[c].to_numpy(dtype=float)
                    if len(y) > h:
                        num += np.nansum((y[h:] - y[:-h]) ** 2)
                        den += np.nansum((y[h:] - mu) ** 2)
                if den > 0:
                    sk.append(1 - num / den)
            out.append((h_s, np.median(sk) if sk else np.nan))
        hor = dict(out)
        print('  skill persistence (median cac service) theo horizon:', {f'{k}s': round(float(v), 3) for k, v in hor.items()})
        longest = hor[120]
        res.add('D', 'khong tam thuong', 'PASS' if longest < 0.9 else 'FAIL',
                f'persistence horizon 120s co skill {longest:.2f} (< 0.9: bai toan khong bi persistence giai xong)')
    else:
        res.add('D', 'khong tam thuong', 'WARN', 'khong ap dung (khoi on dinh: persistence trong khoi luon dung)')

    print('\n  BANG THAM CHIEU (skill = 1 - MSE/MSE_hang_so_train; >0 tot hon hang so):')
    print(ref.round(3).to_string(index=False))
    out_csv = os.path.join(a.dir, 'forecast_audit_reference.csv')
    ref.to_csv(out_csv, index=False)
    res.summary()
    print(f'  bang tham chieu -> {out_csv}')


if __name__ == '__main__':
    main()
