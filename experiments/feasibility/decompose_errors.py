# -*- coding: utf-8 -*-
"""
PHAN TACH NGUON SAI SO CUA P1 (chi tap PHAT TRIEN: base/promo/recs)
====================================================================
Voi moi buoc tai TRUOC diem gay cua ramp tinh nang, tach sai so muc su dung du doan cua P1 thanh:
  * sai o TAI   : beta * (W_P1 - W_do_duoc)      -- k_s = 1 va co cau nen co dung khong
  * sai o CHI PHI: (alpha + beta*W_do_duoc) - CPU_do_duoc  -- cho TRUOC tai that, CPU moi lan goi co giong trung binh nen khong
Cong hai phan = sai so tong (theo dinh nghia). Dong thoi do TRUC TIEP:
  * k_hat_s     = (W_do_duoc - rho_s*L) / (Delta*L)   -- so lan service s bi goi moi lan dung tinh nang (gia dinh: 1)
  * cost_factor = CPU_do_duoc / (alpha + beta*W_do_duoc) -- ti le chi phi thuc so voi co che hoc tu nen
Dung he so co che TU BAN DONG BANG (khong fit lai). KHONG doc track/review.

    python experiments/decompose_errors.py --frozen data/processed/frozen/predictions_frozen_RE2.json --ramp-dir data/raw/SS-LIMITS
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_frozen as EF  # noqa: E402

SCORED = ['front-end', 'catalogue', 'user', 'carts', 'orders']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--ramp-dir', required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    fz = json.load(open(a.frozen, encoding='utf-8'))
    mech, cores = fz['mechanism'], fz['params']['cores']
    preds = {(p['predictor'], p['feature'], round(float(p['scale']), 3)): p for p in fz['predictions']}
    runs = EF.load_ramps(a.ramp_dir, EF.DEV)          # chi tap phat trien: khong the doc track/review o day
    rows = []
    for r in runs:
        lo = r['bp'].get('lo')
        if lo is None:
            continue
        feat, sc = r['feature'], r['scale']
        pct = float(next((s['feat_rps'] / s['target_rps'] * 100 for s in r['steps'] if s['target_rps']), 0)) if feat != 'base' else 0.0
        delta = pct / 100.0
        chain = set(preds[('P1', feat, sc)]['chain']) if feat != 'base' else set()
        d = r['df']
        d = d[(d['step_warm'] == 0) & (d['target_rps'] <= lo)]
        for L, g in d.groupby('target_rps'):
            for s in SCORED:
                m = mech[s]
                cap = 100.0 * cores[s]
                w_meas, c_meas = float(g[f'{s}_workload'].mean()), float(g[f'{s}_cpu'].mean())
                w_p1 = L * (m['rho'] + (delta if s in chain else 0.0))
                cpu_p1 = m['alpha'] + m['beta'] * w_p1
                cpu_given_w = m['alpha'] + m['beta'] * w_meas
                rows.append({
                    'feature': f'{feat}x{sc:g}', 'fbase': feat, 'node': s,
                    'role': 'gateway' if s == 'front-end' else ('chain' if s in chain else 'non-chain'), 'L': int(L),
                    'k_hat': (w_meas - m['rho'] * L) / (delta * L) if delta else np.nan,
                    'cost_factor': c_meas / cpu_given_w if cpu_given_w > 0 else np.nan,
                    'err_total': (cpu_p1 - c_meas) / cap * 100,
                    'err_workload': m['beta'] * (w_p1 - w_meas) / cap * 100,
                    'err_cost': (cpu_given_w - c_meas) / cap * 100})
    d = pd.DataFrame(rows)
    f = d[d.fbase != 'base']
    print(f'\n=== PHAN TACH SAI SO P1 | {len(runs)} ramp (dev) | sai so co dau, diem % cua tran (duong = du doan CAO hon do duoc) ===\n')
    print('[A] Do TRUC TIEP tren dau vao (gia dinh cua P1: k_hat = 1 cho node trong chain, 0 ngoai chain; cost_factor = 1):')
    t = f.groupby(['fbase', 'node']).agg(vai_tro=('role', 'first'), k_hat=('k_hat', 'median'),
                                          cost_factor=('cost_factor', 'median'), n=('L', 'size')).round(2)
    print(t.to_string())
    print('\n[B] Sai so co dau P1 = sai o TAI + sai o CHI PHI (trung binh, diem %):')
    b = f.groupby(['fbase', 'node'])[['err_workload', 'err_cost', 'err_total']].mean().round(2)
    print(b.to_string())
    print('\n[C] Gop theo vai tro (trung binh do lon |sai so|, diem %) -- phan nao chiem uu the:')
    f2 = f.assign(a_w=f.err_workload.abs(), a_c=f.err_cost.abs(), a_t=f.err_total.abs())
    print(f2.groupby(['fbase', 'role'])[['a_w', 'a_c', 'a_t']].mean().round(2).rename(
        columns={'a_w': '|sai o tai|', 'a_c': '|sai o chi phi|', 'a_t': '|tong|'}).to_string())
    ctl = d[d.fbase == 'base']
    if len(ctl):
        print('\n[D] Doi chung baseline (khong tinh nang): cost_factor trung vi theo node =',
              ctl.groupby('node')['cost_factor'].median().round(2).to_dict())
    if a.out:
        d.to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
