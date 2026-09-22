# -*- coding: utf-8 -*-
"""
DONG BANG P3 TIEN CUU (sau khi CO code, TRUOC khi chay ramp/xem dap an)
========================================================================
P3 can k_s DO DUOC, ma k chi do duoc SAU KHI code ton tai -- nen "dong bang truoc" cho P3 phai chia
hai buoc: (1) P0/P1/P2 dong bang TRUOC khi agent viet code (da lam, freeze_predictions.py --feature-set
prosp), (2) P3 dong bang o DAY: sau khi co code + do k bang probing chuc nang (KHONG dung tai/diem gay),
NHUNG TRUOC khi chay ramp. Ca hai deu la du doan that su (khong nhin dap an), chi khac thoi diem.

    python experiments/freeze_p3_prospective.py --feature browse --k-file data/processed/frozen/k_measured_prosp.json \\
        --base-frozen data/processed/frozen/predictions_frozen_RE2_P2_prosp.json --ramp-dir data/raw/SS-LIMITS-PROSP
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'scm'))
import feasibility_predictor as FP  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GRID = list(range(40, 261, 20))


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature', required=True)
    ap.add_argument('--k-file', required=True)
    ap.add_argument('--base-frozen', required=True)
    ap.add_argument('--ramp-dir', required=True, help='chi de KIEM CHUA co du lieu -- khong doc gia tri')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    exist = glob.glob(os.path.join(a.ramp_dir, f'ramp_{a.feature}_*'))
    if exist:
        sys.exit(f'TU CHOI: {a.ramp_dir} da co ramp cua {a.feature} ({exist[:2]}...); P3 khong con la du doan truoc.')

    base = json.load(open(a.base_frozen, encoding='utf-8'))
    kdoc = json.load(open(a.k_file, encoding='utf-8'))
    k = kdoc['features'][a.feature]['measured_per_use']
    P = FP.FeasibilityPredictor(base['mechanism'], base['params']['cores'], base['params']['u_star'],
                                feature_cost=base['p2_params'])

    preds = []
    for scale in (1.0, 2.0):
        r, node = P.breakpoint(mode='P2', feature=a.feature, scale=scale, k=k)
        preds.append({'predictor': 'P3', 'feature': a.feature, 'scale': scale,
                      'chain': kdoc['features'][a.feature]['chain_measured'], 'k_used': k,
                      'breakpoint_rps': round(r, 1), 'bottleneck': node,
                      'grid': {str(L): P.verdict(L, mode='P2', feature=a.feature, scale=scale, k=k) for L in GRID}})

    a.out = a.out or os.path.join(BASE, 'data', 'processed', 'frozen', f'predictions_frozen_RE2_P3_prosp_{a.feature}.json')
    doc = {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
          'stage': f'P3 TIEN CUU: k do bang probing SAU khi agent doc lap viet code, TRUOC khi chay ramp/xem dap an cho {a.feature}',
          'feature': a.feature, 'k_source_file': os.path.basename(a.k_file), 'k_source_sha256': sha_file(a.k_file),
          'base_frozen_file': os.path.basename(a.base_frozen), 'base_frozen_sha256': sha_file(a.base_frozen),
          'code_sha256': {'feasibility_predictor.py': sha_file(FP.__file__), os.path.basename(__file__): sha_file(__file__)},
          'predictions': preds}
    json.dump(doc, open(a.out, 'w', encoding='utf-8'), indent=1, sort_keys=True, ensure_ascii=False)
    digest = sha_file(a.out)
    open(a.out + '.sha256', 'w').write(f'{digest}  {os.path.basename(a.out)}\n')
    print(f'=== P3 TIEN CUU dong bang cho {a.feature} ===')
    for p in preds:
        print(f"  x{p['scale']:g}: R*={p['breakpoint_rps']} req/s, nghen o {p['bottleneck']}")
    print(f'file: {a.out}\nSHA-256: {digest}')


if __name__ == '__main__':
    main()
