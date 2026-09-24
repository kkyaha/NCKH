# -*- coding: utf-8 -*-
"""
HIEU CHINH THAM SO P2 (chi phi moi lan goi c_s va chi phi gateway x) -- CHI tap PHAT TRIEN (promo, recs)
=========================================================================================================
Co che nen (rho, alpha, beta), tran, u* lay NGUYEN tu ban dong bang (khong fit lai). Tham so P2 duoc fit tu cac buoc
tai TRUOC diem gay cua ramp promo/recs. KHONG doc track/review (load_ramps chi lay tap DEV).

  --lofo   kiem chung bo-mot-tinh-nang: fit tren tinh nang con lai, du doan tinh nang bi bo (P1 vs P2)
  --save   ghi tham so fit tren TOAN BO tap phat trien vao file JSON co van tay (dung cho freeze_predictions --p2-params)

    python experiments/fit_feature_costs.py --frozen data/processed/frozen/predictions_frozen_RE2.json --ramp-dir data/raw/SS-LIMITS --lofo
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
sys.path.insert(0, os.path.join(BASE, 'src', 'scm'))
import feasibility_predictor as FP  # noqa: E402
import evaluate_frozen as EF  # noqa: E402


def build_rows(runs, mech, cores, preds):
    rows = []
    for r in runs:
        if r['feature'] == 'base' or r['bp'].get('lo') is None:
            continue
        feat, sc = r['feature'], r['scale']
        pct = next((s['feat_rps'] / s['target_rps'] * 100 for s in r['steps'] if s['target_rps']), 0.0)
        chain = list(preds[('P1', feat, sc)]['chain'])
        d = r['df']
        d = d[(d['step_warm'] == 0) & (d['target_rps'] <= r['bp']['lo'])]
        for L, g in d.groupby('target_rps'):
            for s in FP.SCORED:
                rows.append({'feature': feat, 'scale': sc, 'run': r['run'], 'node': s, 'L': int(L), 'delta': pct / 100.0,
                             'chain': set(chain), 'n_calls': len(chain) - 1, 'w_meas': float(g[f'{s}_workload'].mean()),
                             'c_meas': float(g[f'{s}_cpu'].mean()), 'cap': 100.0 * cores[s]})
    return pd.DataFrame(rows)


def score(P, rows, runs, mode, feature):
    """Sai so muc su dung theo vai tro (diem %) va diem gay du doan cho tinh nang `feature`."""
    out = {}
    sub = rows[rows.feature == feature]
    errs = {'gateway': [], 'chain': [], 'non-chain': []}
    for _, r in sub.iterrows():
        u = P.utilization(r['L'], mode=mode, feature=feature, scale=r['scale'])[r['node']]
        role = 'gateway' if r['node'] == 'front-end' else ('chain' if r['node'] in r['chain'] else 'non-chain')
        errs[role].append(abs(u - r['c_meas'] / r['cap']) * 100)
    out['err'] = {k: round(float(np.mean(v)), 2) for k, v in errs.items() if v}
    out['err_all'] = round(float(np.mean(sum(errs.values(), []))), 2)
    bps = {}
    for sc in sorted(sub.scale.unique()):
        bps[sc] = round(P.breakpoint(mode=mode, feature=feature, scale=sc)[0], 1)
    out['bp'] = bps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--ramp-dir', required=True)
    ap.add_argument('--lofo', action='store_true')
    ap.add_argument('--save', default='')
    a = ap.parse_args()
    fz = json.load(open(a.frozen, encoding='utf-8'))
    mech, cores, u_star = fz['mechanism'], fz['params']['cores'], fz['params']['u_star']
    preds = {(p['predictor'], p['feature'], round(float(p['scale']), 3)): p for p in fz['predictions']}
    runs = EF.load_ramps(a.ramp_dir, EF.DEV)
    rows = build_rows(runs, mech, cores, preds)
    feats = sorted(rows.feature.unique())
    print(f'\n=== HIEU CHINH P2 | tap PHAT TRIEN: {feats} | {len(runs)} ramp ===\n')

    full = FP.fit_feature_cost(rows, mech)
    print('Fit tren TOAN BO tap phat trien:')
    print('  c_s (chi phi MOI LAN GOI so voi trung binh nen):', {k: round(v, 3) for k, v in full['c'].items()})
    print(f"  x (gateway: 1 request tinh nang = 1 + x*n_calls lan request nen): {full['x']:.4f}")
    print('  boi so chi phi gateway theo tung tinh nang (phai gan nhau neu x theo n_calls dung):',
          {k: round(v, 3) for k, v in full['gateway_multiple_by_feature'].items()},
          '| n_calls:', {f: int(rows[rows.feature == f].n_calls.iloc[0]) for f in feats})

    if a.lofo:
        print('\n--- KIEM CHUNG BO-MOT-TINH-NANG (fit tren cac tinh nang con lai; du doan tinh nang bi bo) ---')
        for held in feats:
            fit = FP.fit_feature_cost(rows[rows.feature != held], mech)
            P = FP.FeasibilityPredictor(mech, cores, u_star, feature_cost=fit)
            meas = [(r['bp']['lo'], r['bp']['hi']) for r in runs if r['feature'] == held]
            print(f"\n  bo {held}: tham so fit tren cac tinh nang con lai: c={ {k: round(v, 2) for k, v in fit['c'].items()} }, x={fit['x']:.3f}")
            for mode in ('P1', 'P2'):
                sc = score(P, rows, runs, mode, held)
                print(f"    {mode}: sai so muc su dung TB {sc['err_all']:5.2f} diem % (theo vai tro {sc['err']}); diem gay du doan {sc['bp']}")
            by_scale = {}
            for r in runs:
                if r['feature'] == held:
                    by_scale.setdefault(r['scale'], []).append((r['bp']['lo'], r['bp']['hi']))
            print('    do duoc [lo, hi):', {s: (float(np.median([x[0] for x in v])), float(np.median([x[1] for x in v]))) for s, v in by_scale.items()})

    if a.save:
        files = sorted(glob.glob(os.path.join(a.ramp_dir, 'ramp_*', 'run*', 'steps.json')))
        dev_files = [f for f in files if json.load(open(f, encoding='utf-8')).get('feature') in ('promo', 'recs')]
        doc = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'stage': 'DEVELOPMENT (fit tren promo/recs; khong phai du doan truoc)',
               'frozen_base': {'file': os.path.basename(a.frozen), 'sha256': hashlib.sha256(open(a.frozen, 'rb').read()).hexdigest()},
               'fit_features': feats, 'fit_ramp_files': len(dev_files),
               'fit_data_sha256': FP.sha256_files([os.path.join(os.path.dirname(f), 'simple_metrics.csv') for f in dev_files]),
               'params': {'c': full['c'], 'x': full['x']}, 'gateway_multiple_by_feature': full['gateway_multiple_by_feature']}
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        json.dump(doc, open(a.save, 'w', encoding='utf-8'), indent=1, sort_keys=True)
        print(f'\n[OK] tham so P2 (phat trien) -> {a.save}')


if __name__ == '__main__':
    main()
