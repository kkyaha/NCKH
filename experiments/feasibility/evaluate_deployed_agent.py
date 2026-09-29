# -*- coding: utf-8 -*-
"""
DANH GIA TIEN CUU BO PHAN QUYET DA TICH HOP (FeasibilityAgent), khong chi bo du doan offline
============================================================================================
Khac `evaluate_frozen.py` (cham diem tung con so du doan), script nay cham diem ĐUONG RA THAT
ma he thong tra cho nguoi dung: `FeasibilityVerdict` (FEASIBLE / MARGINAL / INFEASIBLE) cong
cac co canh bao (`latency_risk`, `extrapolating`).

Vi sao la TIEN CUU: ca `FeasibilityAgent.assess()` va nguong `LATENCY_RISK_N_CALLS = 4` duoc
commit 2026-09-22, TRUOC khi 8 tinh nang cua vong tien cuu 2 (2026-09-25) ton tai. Khong co
tham so nao duoc chinh sau khi thay du lieu moi. Vi the:
  * do chinh xac cua phan quyet tren luoi tai, VA
  * do nhay cua co `latency_risk` o dung nhung o ma phan quyet CPU that bai
deu la ket qua tien cuu.

    python experiments/feasibility/evaluate_deployed_agent.py \
        --features login,register,wishlist,catsearch,account,preview,orderhist,related \
        --ramp-dir data/raw/SS-PROSP2 --k-file data/processed/frozen/k_measured_v3.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src', 'agents'))
sys.path.insert(0, os.path.join(BASE, 'src', 'scm'))


def measured_cells(ramp_dir, features, anchors):
    """Diem gay do duoc [lo, hi] cho tung (tinh nang, cuong do), trung vi qua cac lan lap."""
    cells = {}
    for sp in sorted(glob.glob(os.path.join(BASE, ramp_dir, 'ramp_*', 'run*', 'steps.json'))):
        j = json.load(open(sp, encoding='utf-8'))
        feat = j.get('feature', 'base')
        if feat not in features:
            continue
        pct = float(j.get('feature_pct', 0) or 0)
        sc = round(pct / anchors[feat], 3) if anchors.get(feat) else 1.0
        b = j['breakpoint']
        cells.setdefault((feat, sc), {'lo': [], 'hi': []})
        cells[(feat, sc)]['lo'].append(b['lo'])
        if b['hi'] is not None:
            cells[(feat, sc)]['hi'].append(b['hi'])
    return {c: {'lo': float(np.median(v['lo'])),
                'hi': float(np.median(v['hi'])) if v['hi'] else None,
                'n': len(v['lo'])} for c, v in cells.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--ramp-dir', default='data/raw/SS-PROSP2')
    ap.add_argument('--k-file', default='data/processed/frozen/k_measured_v3.json')
    ap.add_argument('--x1-only', action='store_true',
                    help='chi cham o x1 (cac cuong do cua cung tinh nang khong doc lap voi nhau)')
    a = ap.parse_args(argv)

    import feasibility_predictor as FP
    from feasibility_agent import NewFeatureFeasibilityAgent, LATENCY_RISK_N_CALLS
    sys.path.insert(0, os.path.join(BASE, 'experiments', 'collect'))
    import load_sweep_collect as L

    feats = [f.strip() for f in a.features.split(',') if f.strip()]
    anchors = {f: L.FEATURES[f]['anchor'] for f in feats}
    km = json.load(open(os.path.join(BASE, a.k_file), encoding='utf-8'))['features']
    cells = measured_cells(a.ramp_dir, feats, anchors)
    agent = NewFeatureFeasibilityAgent()

    rows = []
    for (feat, sc), m in sorted(cells.items()):
        if a.x1_only and sc != 1.0:
            continue
        k = km[feat]['measured_per_use']
        lo, hi = m['lo'], m['hi']
        # Phan quyet tai chinh diem gay DO DUOC (lo): he THUC SU con dat SLO o day, nen phan quyet
        # dung phai la FEASIBLE/MARGINAL. Va tai bac VO (hi) phan quyet dung phai la INFEASIBLE.
        v_lo = agent.assess(feat, scale=sc, L_peak=lo, k=k, bootstrap=False)
        v_hi = agent.assess(feat, scale=sc, L_peak=hi, k=k, bootstrap=False) if hi else None
        n_calls = sum(k.get(s, 1.0) for s in FP.SOCKSHOP_CALL_CHAINS[FP.FEATURE_ARCHETYPE[feat]]['services']
                      if s != FP.GATEWAY)
        r_pred = v_lo.breakpoint_rps
        late = (hi is not None and r_pred > hi)          # huong NGUY HIEM: bao song sot khi da vo
        rows.append({'cell': f'{feat}x{sc:g}', 'R*_pred': r_pred, 'lo': lo, 'hi': hi, 'n_ramp': m['n'],
                     'verdict@lo': v_lo.verdict, 'verdict@hi': (v_hi.verdict if v_hi else '-'),
                     'sai_huong_nguy_hiem': late,
                     'sum_k': round(n_calls, 2), 'latency_risk': v_lo.latency_risk,
                     'extrapolating': v_lo.extrapolating})
    d = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(f'\n=== PHAN QUYET CUA AGENT DA TICH HOP ({len(d)} o, k do bang probe -> P3) ===')
    print(d.to_string(index=False))

    ok_lo = d['verdict@lo'].isin(['FEASIBLE', 'MARGINAL']).sum()
    ok_hi = (d['verdict@hi'] == 'INFEASIBLE').sum()
    n_hi = (d['hi'].notna()).sum()
    print(f'\n  Phan quyet tai bac CON DAT SLO (lo): dung {ok_lo}/{len(d)} (dung = FEASIBLE hoac MARGINAL)')
    print(f'  Phan quyet tai bac DA VO   SLO (hi): dung {ok_hi}/{n_hi} (dung = INFEASIBLE)')

    dang = d[d['sai_huong_nguy_hiem']]
    print(f'\n  O sai HUONG NGUY HIEM (du doan diem gay VUOT bac da vo): {len(dang)}/{len(d)}'
          + (f" -> {', '.join(dang['cell'])}" if len(dang) else ''))
    print(f'\n=== CO `latency_risk` (nguong Sigma k >= {LATENCY_RISK_N_CALLS}, dong bang 2026-09-22) ===')
    tp = int((d['latency_risk'] & d['sai_huong_nguy_hiem']).sum())
    fn = int((~d['latency_risk'] & d['sai_huong_nguy_hiem']).sum())
    fp = int((d['latency_risk'] & ~d['sai_huong_nguy_hiem']).sum())
    print(f'  bat dung o nguy hiem (TP): {tp}   bo sot (FN): {fn}   bao dong o an toan (FP): {fp}')
    if fn and not tp:
        print('  => Co NAY KHONG bat duoc cac o ma phan quyet CPU that bai: nguong dua tren TONG so luot goi,'
              '\n     nhung sai so con lai khong tuong quan voi tong so luot goi (xem diagnose_service_demand_gate.py).')
    return d


if __name__ == '__main__':
    main()
