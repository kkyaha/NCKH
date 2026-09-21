# -*- coding: utf-8 -*-
"""
DONG BANG DU DOAN (pre-registration) TRUOC KHI DO DAP AN
========================================================
Chay SAU Pha A + Phase 0 va TRUOC Pha B. Ghi du doan P0/P1/P1_ctrl cho moi (tinh nang, cuong do) vao mot file
JSON co van tay du lieu/ma nguon, roi in SHA-256 cua file (ghi hash vao git/luu rieng de chung minh khong sua sau).

    python experiments/freeze_predictions.py --train-dir data/raw/SS-TRAIN --ramp-dir data/raw/SS-LIMITS --limits RE2
    python experiments/freeze_predictions.py --train-dir data/raw/SS-TRAIN --ramp-dir data/raw/SS-LIMITS-C1 --limits C1
    python experiments/freeze_predictions.py --dev --ramp-dir data/raw/SS-LIMITS --limits RE2     # thu code, KHONG phai dong bang

Chot chan:
  * tu choi neu --ramp-dir da co ramp cua tinh nang (da nhin truoc dap an) tru khi --allow-post-hoc (khi do ghi post_hoc=true)
  * hieu chinh u* CHI tu ramp baseline; train CHI tu du lieu khong tinh nang, khong tran (--dev moi cho phep tran)
  * co che duoc kiem tren cac muc tai GIU LAI (> --train-max) truoc khi dung de du doan
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE, 'src', 'scm'))
import feasibility_predictor as FP  # noqa: E402

GRID = list(range(40, 261, 20))
CELLS = [('base', 1.0)] + [(f, s) for f in FP.FEATURE_ARCHETYPE for s in (1.0, 2.0)]


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def mechanism_check(df_all, mech, train_max):
    """Loi tuong doi cua CPU du doan (dung W do duoc, de chi do loi CO CHE) tren muc tai giu lai."""
    lv = FP.level_of(df_all)
    h = df_all[lv > train_max]
    if 'feature' in h:
        h = h[h['feature'] == 'base']
    if 'step_violated' in h:
        h = h[h['step_violated'] == 0]
    rows = []
    for level, g in h.groupby(FP.level_of(h)):
        for s in FP.SCORED:
            meas = float(g[f'{s}_cpu'].mean())
            pred = mech[s]['alpha'] + mech[s]['beta'] * float(g[f'{s}_workload'].mean())
            rows.append({'level': float(level), 'service': s, 'measured': round(meas, 3), 'predicted': round(pred, 3),
                         'rel_err_pct': round(100 * (pred - meas) / meas, 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', default='')
    ap.add_argument('--ramp-dir', required=True)
    ap.add_argument('--limits', default='RE2')
    ap.add_argument('--train-max', type=float, default=150)
    ap.add_argument('--out', default='', help='mac dinh data/processed/frozen/predictions_frozen_<limits>.json (MOI cau hinh tran mot file)')
    ap.add_argument('--dev', action='store_true', help='thu code: train tu chinh ramp baseline (co tran); KHONG phai dong bang')
    ap.add_argument('--allow-post-hoc', action='store_true')
    a = ap.parse_args()

    if not a.out:
        a.out = os.path.join(BASE, 'data', 'processed', 'frozen', f'predictions_frozen_{a.limits}{"_dev" if a.dev else ""}.json')

    # ---- chot chan: chua co dap an tinh nang
    feat_ramps = [d for d in glob.glob(os.path.join(a.ramp_dir, 'ramp_*'))
                  if os.path.isdir(d) and os.path.basename(d) != 'ramp_base']      # chi THU MUC (bo qua ramp_manifest_*.json)
    post_hoc = bool(feat_ramps)
    if post_hoc and not a.allow_post_hoc and not a.dev:
        sys.exit(f'TU CHOI: {a.ramp_dir} da co ramp tinh nang ({[os.path.basename(d) for d in feat_ramps][:3]}...). '
                 f'Dong bang bay gio khong con la du doan truoc. Dung --allow-post-hoc neu chap nhan (se ghi post_hoc=true).')

    with open(os.path.join(BASE, 'deploy', 'sockshop', 'limits.json'), encoding='utf-8') as f:
        cores = json.load(f)['configs'][a.limits]
    ncpu = 12
    cores = {s: cores.get(s, ncpu) for s in FP.SERVICES}            # khong co gioi han => khong tran (12 core)

    # ---- du lieu train (khong tinh nang, khong tran)
    if a.dev:
        df_all, train_root = FP.load_runs(a.ramp_dir), a.ramp_dir
        df_all = df_all[df_all['feature'] == 'base'] if 'feature' in df_all else df_all
        train = FP.select_train(df_all, a.train_max, allow_limits=True)
    else:
        if not a.train_dir:
            sys.exit('--train-dir la bat buoc (tru --dev)')
        df_all, train_root = FP.load_runs(a.train_dir), a.train_dir
        train = FP.select_train(df_all, a.train_max)
    mech = FP.fit_mechanism(train)
    u_star, u_detail = FP.calibrate_u_star(a.ramp_dir, cores)
    P = FP.FeasibilityPredictor(mech, cores, u_star)
    hold = mechanism_check(df_all, mech, a.train_max)

    # ---- du doan
    preds = []
    for feature, scale in CELLS:
        feat = None if feature == 'base' else feature
        variants = [('P0', dict(mode='P0')), ('P1', dict(mode='P1'))]
        if feat:
            variants.append(('P1_ctrl', dict(mode='P1', chain=FP.wrong_chain(feat, 0))))
        for name, kw in variants:
            if name == 'P0' and feat is None:
                continue                                   # baseline: P0 == P1 (khong co tinh nang)
            r, node = P.breakpoint(feature=feat, scale=scale, **kw)
            preds.append({
                'predictor': name if feat else 'base', 'feature': feature, 'scale': scale,
                'chain': (kw.get('chain') or P.spec(feat, scale)[1]) if feat else [],
                'breakpoint_rps': round(r, 1), 'bottleneck': node,
                'grid': {str(L): P.verdict(L, feature=feat, scale=scale, **kw) for L in GRID}})

    train_files = sorted(glob.glob(os.path.join(train_root, '*', '*', 'simple_metrics.csv')))
    ramp_files = sorted(glob.glob(os.path.join(a.ramp_dir, 'ramp_base', 'run*', 'simple_metrics.csv')))
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=BASE).stdout.strip()
    except Exception:
        commit = ''
    doc = {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'dev': a.dev, 'post_hoc': post_hoc,
        'params': {'limits': a.limits, 'cores': cores, 'train_max_rps': a.train_max, 'u_star': round(u_star, 4),
                   'u_margin': FP.U_MARGIN, 'scored': FP.SCORED, 'grid_rps': GRID},
        'u_star_calibration': u_detail,
        'mechanism': {s: {k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()} for s, m in mech.items()},
        'mechanism_check_holdout': hold,
        'fingerprint': {
            'train_files': len(train_files), 'train_sha256': FP.sha256_files(train_files),
            'ramp_base_files': len(ramp_files), 'ramp_base_sha256': FP.sha256_files(ramp_files),
            'code_sha256': {'feasibility_predictor.py': sha_file(FP.__file__), 'freeze_predictions.py': sha_file(__file__),
                            'limits.json': sha_file(os.path.join(BASE, 'deploy', 'sockshop', 'limits.json')),
                            'slo.json': sha_file(os.path.join(BASE, 'deploy', 'sockshop', 'slo.json'))},
            'git_commit': commit},
        'predictions': preds}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    body = json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False).encode('utf-8')
    with open(a.out, 'wb') as f:
        f.write(body)
    digest = hashlib.sha256(body).hexdigest()
    with open(a.out + '.sha256', 'w') as f:
        f.write(f'{digest}  {os.path.basename(a.out)}\n')

    # ---- bao cao
    tag = 'DEV (KHONG phai dong bang)' if a.dev else ('POST-HOC' if post_hoc else 'DONG BANG (pre-registered)')
    print(f'\n=== {tag} | u*={u_star:.3f} tu {len(u_detail)} ramp baseline: {[(d["bottleneck"], d["u"]) for d in u_detail]} ===')
    print(f'train: {len(train)} hang (<= {a.train_max:g} req/s, {len(train_files)} tep)')
    print('\nHe so co che (5 node chinh):')
    print(pd.DataFrame({s: {k: mech[s][k] for k in ('rho', 'alpha', 'beta', 'r2', 'n')} for s in FP.SCORED}).T.round(4).to_string())
    if hold:
        print('\nKiem co che tren muc tai GIU LAI (loi % cua CPU du doan voi W do duoc):')
        print(pd.DataFrame(hold).pivot(index='service', columns='level', values='rel_err_pct').to_string())
    print('\nDiem gay du doan (req/s) va node nghen:')
    t = pd.DataFrame([{'cell': f"{p['feature']}x{p['scale']:g}", 'predictor': p['predictor'], 'R*': p['breakpoint_rps'],
                       'bottleneck': p['bottleneck']} for p in preds])
    print(t.pivot(index='cell', columns='predictor', values='R*').reindex(
        [f"{f}x{s:g}" for f, s in CELLS]).to_string())
    print(f'\nfile: {a.out}\nSHA-256: {digest}')


if __name__ == '__main__':
    main()
