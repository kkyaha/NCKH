# -*- coding: utf-8 -*-
"""
CHAN DOAN: vi sao mot tinh nang vo SLO DUOI nguong CPU da hieu chinh (u* ~ 0.878)
=================================================================================
Cau hoi: bo du doan (P0..P3) phan quyet bang `u_s = CPU_s / C_s` so voi mot nguong duy
nhat u*. Neu SLO vo khi u_max CON THAP hon u*, bo du doan se bao diem gay MUON hon thuc
te -- tuc "kha thi gia", huong nguy hiem. Vong tien cuu 2 co dung hien tuong nay o
`login`/`register` (docs/DATA_FRAMEWORK.md muc 5l).

Gia thuyet kiem tra o day: muc su dung TAI DIEM VO (u_break) phu thuoc vao NHU CAU PHUC VU
cua chinh tinh nang -- do tre p99 do duoc khi he RANH (D_feat, tu probe_feature_chain.py,
KHONG dung mot hat du lieu tai nao). Tinh nang nao da ngon phan lon ngan sach SLO khi he
con ranh thi chi con it du dia, nen p99 cham tran o muc CPU thap hon.

    python experiments/feasibility/diagnose_service_demand_gate.py \
        --ramp-dir data/raw/SS-PROSP2 --k-file data/processed/frozen/k_measured_v3.json

LUU Y VE TINH CHINH DANH: day la CHAN DOAN HOI CUU tren du lieu da xem. No KHONG duoc
dung de chinh lai bo du doan roi bao cao nhu du doan truoc. Muon kiem chung tien cuu cong
nhu cau phuc vu, phai dong bang nguong bang `freeze_service_demand_gate.py` TRUOC khi do.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SCORED = ['front-end', 'catalogue', 'user', 'carts', 'orders']


def u_at_step(run_dir, rps, ceilings):
    """Muc su dung trung binh tung node o MOT bac tai (bo phan warmup cua bac do)."""
    d = pd.read_csv(os.path.join(run_dir, 'simple_metrics.csv'))
    d = d.rename(columns={c: c[3:] for c in d.columns if c.startswith('gt_')})
    seg = d[(d['target_rps'] == rps) & (d['step_warm'] == 0)]
    if not len(seg):
        return None
    return {s: seg[f'{s}_cpu'].mean() / (100.0 * ceilings.get(s, 1.0))
            for s in SCORED if f'{s}_cpu' in seg}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ramp-dir', default='data/raw/SS-PROSP2')
    ap.add_argument('--k-file', default='data/processed/frozen/k_measured_v3.json')
    ap.add_argument('--limits', default='RE2')
    ap.add_argument('--slo-p99-ms', type=float, default=250.0)
    a = ap.parse_args(argv)

    ceilings = json.load(open(os.path.join(BASE, 'deploy', 'sockshop', 'limits.json'),
                              encoding='utf-8'))['configs'][a.limits]
    km = json.load(open(os.path.join(BASE, a.k_file), encoding='utf-8'))['features']

    rows = []
    for sp in sorted(glob.glob(os.path.join(BASE, a.ramp_dir, 'ramp_*', 'run*', 'steps.json'))):
        j = json.load(open(sp, encoding='utf-8'))
        feat, lo = j.get('feature', 'base'), j['breakpoint']['lo']
        if lo is None:
            continue
        u = u_at_step(os.path.dirname(sp), lo, ceilings)
        if not u:
            continue
        # x1 va x2 dung chung ban cai/k nen KHONG phai quan sat doc lap: chi lay x1 (va base)
        anchor = km.get(feat, {}).get('archetype')
        pct = float(j.get('feature_pct', 0) or 0)
        rows.append({'feature': feat, 'pct': pct, 'lo': lo, 'hi': j['breakpoint']['hi'],
                     'u_break': max(u.values()), 'node': max(u, key=u.get),
                     'archetype': anchor})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f'khong co ramp nao duoi {a.ramp_dir}')

    # mot dong moi tinh nang: trung vi u_break qua cac lan lap, o cuong do x1 (pct = anchor)
    base_u = df[df.feature == 'base']['u_break']
    per = []
    for feat, g in df[df.feature != 'base'].groupby('feature'):
        k = km.get(feat)
        if not k:
            continue
        g1 = g[g.pct == min(g.pct)]              # x1
        d_idle = k['latency_idle']
        per.append({'feature': feat, 'archetype': k['archetype'],
                    'n_ramp': len(g1), 'u_break': round(float(np.median(g1['u_break'])), 3),
                    'D_p99_idle_ms': round(d_idle['p99'] * 1000, 1),
                    'D_p50_idle_ms': round(d_idle['p50'] * 1000, 1),
                    'budget_used_idle': round(d_idle['p99'] * 1000 / a.slo_p99_ms, 3),
                    'sum_k': round(sum(k['measured_per_use'].values()), 2)})
    P = pd.DataFrame(per).sort_values('u_break', ascending=False)

    print(f'\n=== MUC SU DUNG TAI DIEM VO vs NHU CAU PHUC VU LUC HE RANH ({a.ramp_dir}) ===')
    print(f'  baseline (khong tinh nang): u_break trung vi = {np.median(base_u):.3f} '
          f'tren {len(base_u)} ramp  [{base_u.min():.3f}, {base_u.max():.3f}]')
    print(f'  SLO p99 = {a.slo_p99_ms:.0f} ms; `budget_used_idle` = D_p99_idle / SLO\n')
    print(P.to_string(index=False))

    if len(P) >= 4:
        try:
            from scipy import stats
            for col in ('D_p99_idle_ms', 'budget_used_idle', 'sum_k'):
                rho, p = stats.spearmanr(P[col], P['u_break'])
                print(f'\n  Spearman u_break vs {col}: rho = {rho:+.3f}, p = {p:.4f} (n = {len(P)})')
        except ImportError:
            print('\n  [bo qua kiem dinh: khong co scipy]')

    print('\n  Doc bang the nao: u_break thap hon nguong da hieu chinh (u* ~ 0.878) nghia la SLO vo'
          '\n  TRUOC khi CPU cham tran -- bo du doan hien tai se bao diem gay MUON hon thuc te.')
    print('\nLUU Y: chan doan HOI CUU tren du lieu da xem; khong duoc dung de chinh bo du doan roi'
          '\nbao cao nhu du doan truoc. Xem freeze_service_demand_gate.py de kiem chung tien cuu.')
    return P


if __name__ == '__main__':
    main()
