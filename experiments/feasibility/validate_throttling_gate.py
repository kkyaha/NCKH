# -*- coding: utf-8 -*-
"""
KIEM CHUNG CONG THROTTLING tren chinh hai cau hinh da THAT BAI (C1, C2) va tren RE2
===================================================================================
Cong (src/agents/feasibility_agent.py, `QUOTA_MIN_VALIDATED_CORES`): neu han ngach CPU cua node
NGHEN nho hon muc da kiem chung thi bo du doan khong ap dung duoc (che do CFS throttling lam vo
SLO o muc su dung trung binh 20-45%, khong phai ~88%) -> tra UNDECIDED thay vi mot con so.

Script nay kiem hai dieu, tren du lieu da co, khong can do them:
  1. Cong CHAN dung hai cau hinh da that bai (C1: carts 0.15 core; C2: catalogue 0.08 core).
  2. Cong KHONG chan cau hinh RE2 -- tuc no khong lam mat do phu cua cac ket qua hop le.

    python experiments/feasibility/validate_throttling_gate.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src', 'agents'))
sys.path.insert(0, os.path.join(BASE, 'src', 'scm'))


def measured_breakpoints(root):
    """[lo, hi] do duoc cua rieng cac ramp BASELINE (khong tinh nang) -- de so cung mot thu voi
    du doan baseline `assess(None)`; gop ca ramp tinh nang vao se lam lech moc so sanh."""
    los, his = [], []
    for sp in sorted(glob.glob(os.path.join(BASE, 'data', 'raw', root, 'ramp_base*', 'run*', 'steps.json'))):
        b = json.load(open(sp, encoding='utf-8'))['breakpoint']
        if b['lo'] is not None:
            los.append(b['lo'])
        if b['hi'] is not None:
            his.append(b['hi'])
    return (float(np.median(los)) if los else None,
            float(np.median(his)) if his else None, len(los))


def throttled_fraction(root):
    """Ti le CHU KY BI THROTTLE cao nhat qua cac node, o bac tai cuoi cung do duoc."""
    best = {}
    for p in sorted(glob.glob(os.path.join(BASE, 'data', 'raw', root, 'ramp_*', 'run*', 'metrics.csv'))):
        d = pd.read_csv(p)
        for c in d.columns:
            if not c.endswith('container-cpu-cfs-throttled-periods-total'):
                continue
            svc = c.split('_container')[0]
            tot = c.replace('throttled-periods-total', 'periods-total')
            if tot not in d.columns:
                continue
            dp, dt = d[c].iloc[-1] - d[c].iloc[0], d[tot].iloc[-1] - d[tot].iloc[0]
            if dt and dt > 0:
                best[svc] = max(best.get(svc, 0.0), float(dp) / float(dt))
    return best


def main():
    import feasibility_agent as FA

    limits = json.load(open(os.path.join(BASE, 'deploy', 'sockshop', 'limits.json'),
                            encoding='utf-8'))['configs']
    rows = []
    for cfg, root in (('RE2', 'SS-PROSP2'), ('C1', 'SS-LIMITS-C1'), ('C2', 'SS-LIMITS-C2')):
        if not os.path.isdir(os.path.join(BASE, 'data', 'raw', root)):
            print(f'  [bo qua] khong co data/raw/{root}')
            continue
        cores_cfg = {s: float(limits[cfg].get(s, 1.0)) for s in FA.FP.SERVICES}

        def fake(services, *a, **kw):
            return {s: cores_cfg.get(s, 1.0) for s in services}, f'cau hinh {cfg} (tinh)', []
        orig = FA.read_live_cores
        FA.read_live_cores = fake
        try:
            v = FA.NewFeatureFeasibilityAgent().assess(None, L_peak=100.0, bootstrap=False)
        finally:
            FA.read_live_cores = orig

        lo, hi, n = measured_breakpoints(root)
        thr = throttled_fraction(root)
        thr_bott = thr.get(v.bottleneck)
        rows.append({'cau_hinh': cfg, 'du_lieu': root, 'n_ramp': n,
                     'node_nghen': v.bottleneck, 'han_ngach_core': v.bottleneck_cores,
                     'cong_CHAN': v.throttling_risk, 'phan_quyet': v.verdict,
                     'R*_du_doan': v.breakpoint_rps, 'do_duoc_lo': lo, 'hi': hi,
                     'throttle_node_nghen': (round(thr_bott, 3) if thr_bott is not None else None)})
    d = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(f'\n=== CONG THROTTLING (nguong {FA.QUOTA_MIN_VALIDATED_CORES:g} core) ===')
    print(d.to_string(index=False))

    fired = set(d[d['cong_CHAN']]['cau_hinh'])
    print(f"\n  Cong chan: {sorted(fired) or 'khong cau hinh nao'}")
    ok = (fired == {'C1', 'C2'}) if {'C1', 'C2'} <= set(d['cau_hinh']) else None
    if ok is True:
        print('  => DUNG: chan dung hai cau hinh da that bai, KHONG chan RE2 (khong mat do phu).')
    elif ok is False:
        print('  => SAI so voi ky vong: xem lai nguong hoac cau hinh.')
    print('\n  Luu y: nguong nay lay tu CAU HINH han ngach (quyet dinh o Phase 0b), khong fit tren'
          '\n  tinh nang nao. Han che: chi xac nhan 0.5 core la hop le va <=0.15 core la khong;'
          '\n  khoang (0.15, 0.5) CHUA duoc do, nen cong con bao thu trong khoang do.')
    return d


if __name__ == '__main__':
    main()
