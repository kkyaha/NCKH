# -*- coding: utf-8 -*-
"""
DANH GIA BAN DONG BANG SO VOI DAP AN (Pha B)
============================================
So du doan da dong bang (P0 / P1 / P1_ctrl) voi ramp thuc do duoc, o BA muc:

  1. DIEM GAY   : R* du doan so voi khoang do duoc [lo, hi) (lo = buoc dat SLO cuoi, hi = buoc dau chuoi vi pham cuoi).
                  Phan loai: TRONG khoang / SOM hon (an toan) / MUON hon (nguy hiem: bao kha thi gia).
  2. PHAN QUYET : voi moi muc tai L tren luoi, verdict du doan so voi nhan do duoc (L<=lo: kha thi, L>=hi: khong).
                  bao ro loi nguy hiem "kha thi gia" rieng.
  3. MUC SU DUNG TUNG NODE : u_s du doan so voi CPU/tran do duoc o cac buoc TRUOC diem gay (khong can bao hoa) --
                  day la phep do phan biet P0/P1/P1_ctrl (chain co gia tri khong) ke ca khi front-end luon nghen truoc.

TAP KHOA (docs/DATA_FRAMEWORK.md muc 6): mac dinh --split dev chi mo base/promo/recs. Mo track/review can --open-locked
va MOI LAN MO deu ghi vao data/processed/frozen/locked_access.log (dau vet kiem toan). Ban dong bang dung de danh gia
tren tap khoa bat cu luc nao; khong gay hai cho pre-registration mien la khong chinh mo hinh dua tren ket qua do.

    python experiments/evaluate_frozen.py --frozen data/processed/frozen/predictions_frozen_RE2.json --ramp-dir data/raw/SS-LIMITS --split dev
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
DEV = ('base', 'promo', 'recs')
LOCKED = ('track', 'review')
INDEP = ('cartsum', 'quickadd', 'express')     # tinh nang DOC LAP (cai boi agent rieng)
PROSP = ('browse',)                            # tinh nang KIEM DINH TIEN CUU P3 (agent doc lap khac)
ANCHOR = {'promo': 20, 'recs': 30, 'track': 15, 'review': 10, 'cartsum': 10, 'quickadd': 15, 'express': 25, 'browse': 10}


def load_ramps(ramp_dir, features):
    runs = []
    for sp in sorted(glob.glob(os.path.join(ramp_dir, 'ramp_*', 'run*', 'steps.json'))):
        j = json.load(open(sp, encoding='utf-8'))
        feat = j.get('feature', 'base')
        if feat not in features:
            continue
        d = pd.read_csv(os.path.join(os.path.dirname(sp), 'simple_metrics.csv'))
        d = d.rename(columns={c: c[3:] for c in d.columns if c.startswith('gt_')})
        pct = float(j.get('feature_pct', 0) or 0)
        anchor = ANCHOR.get(feat)
        scale = round(pct / anchor, 3) if anchor else 1.0
        runs.append({'feature': feat, 'scale': scale, 'run': os.path.relpath(os.path.dirname(sp), ramp_dir),
                     'bp': j['breakpoint'], 'steps': j['steps'], 'df': d})
    return runs


def label_at(bp, L):
    """1 = khong kha thi (L >= hi), 0 = kha thi (L <= lo), None = mo ho (lo < L < hi)."""
    lo, hi = bp.get('lo'), bp.get('hi')
    if hi is not None and L >= hi:
        return 1
    if lo is not None and L <= lo:
        return 0
    if hi is None and lo is not None:
        return 0 if L <= lo else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--ramp-dir', required=True)
    ap.add_argument('--split', choices=['dev', 'locked', 'all', 'indep', 'prosp'], default='dev')
    ap.add_argument('--open-locked', action='store_true', help='bat buoc de doc tap KHOA (track, review); ghi nhat ky kiem toan')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    if a.split in ('locked', 'all'):
        if not a.open_locked:
            sys.exit('TU CHOI: --split locked/all doc tap KHOA (track, review). Them --open-locked neu day la lan mo co chu dich.')
        with open(os.path.join(os.path.dirname(os.path.abspath(a.frozen)), 'locked_access.log'), 'a', encoding='utf-8') as f:
            f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} MO TAP KHOA split={a.split} frozen_sha256='
                    f'{hashlib.sha256(open(a.frozen, "rb").read()).hexdigest()[:16]} ramp_dir={a.ramp_dir}\n')
    feats = {'dev': DEV, 'locked': LOCKED + ('base',), 'all': DEV + LOCKED, 'indep': INDEP + ('base',), 'prosp': PROSP + ('base',)}[a.split]

    fz = json.load(open(a.frozen, encoding='utf-8'))
    if fz.get('dev') or fz.get('post_hoc'):
        print(f'[CANH BAO] ban dong bang dev={fz.get("dev")} post_hoc={fz.get("post_hoc")} -- KHONG phai du doan truoc.')
    cores = fz['params']['cores']
    preds = {(p['predictor'], p['feature'], round(float(p['scale']), 3)): p for p in fz['predictions']}
    order = ['P0', 'P1', 'P1_ctrl', 'P2']
    PN = [x for x in order if any(p['predictor'] == x for p in fz['predictions'])]
    runs = load_ramps(a.ramp_dir, feats)
    if not runs:
        sys.exit('khong co ramp nao thuoc tap da chon')
    print(f'\n=== DANH GIA {os.path.basename(a.frozen)} | split={a.split} | {len(runs)} ramp | '
          f'{sorted({(r["feature"], r["scale"]) for r in runs})} ===')

    # ---------- 1. diem gay ----------
    rows = []
    cells = sorted({(r['feature'], r['scale']) for r in runs})
    for feat, sc in cells:
        rs = [r for r in runs if (r['feature'], r['scale']) == (feat, sc)]
        los = [r['bp']['lo'] for r in rs if r['bp'].get('lo') is not None]
        his = [r['bp']['hi'] for r in rs if r['bp'].get('hi') is not None]
        lo, hi = (float(np.median(los)) if los else None), (float(np.median(his)) if his else None)
        for pname in ([('base')] if feat == 'base' else PN):
            p = preds.get((pname, feat, sc if feat != 'base' else 1.0))
            if not p:
                continue
            r = p['breakpoint_rps']
            cls = ('khong xac dinh' if lo is None else ('MUON hon (kha thi gia)' if hi is not None and r >= hi else
                   ('TRONG khoang' if r >= lo else 'som hon (an toan)')))
            rows.append({'cell': f'{feat}x{sc:g}', 'predictor': pname, 'R*_pred': r, 'bottleneck_pred': p['bottleneck'],
                         'do_duoc: lo': lo, 'hi': hi, 'n_ramp': len(rs), 'sai_so_vs_lo_%': round(100 * (r - lo) / lo, 1) if lo else None,
                         'phan_loai': cls})
    bp = pd.DataFrame(rows)
    print('\n[1] DIEM GAY (req/s): du doan vs do duoc (lo = buoc dat SLO cuoi, hi = buoc dau vi pham ben vung)')
    print(bp.to_string(index=False))

    # ---------- 2. phan quyet tren luoi ----------
    conf = []
    for r in runs:
        for pname in (['base'] if r['feature'] == 'base' else PN):
            p = preds.get((pname, r['feature'], r['scale'] if r['feature'] != 'base' else 1.0))
            if not p:
                continue
            for Ls, g in p['grid'].items():
                lab = label_at(r['bp'], int(Ls))
                if lab is None:
                    continue
                v = g['verdict']
                conf.append({'predictor': pname, 'feature': r['feature'], 'label': lab,
                             'strict': int(v == 'INFEASIBLE'), 'cautious': int(v in ('INFEASIBLE', 'MARGINAL'))})
    if conf:
        c = pd.DataFrame(conf)
        out = []
        for pname, g in c.groupby('predictor'):
            neg, pos = g[g.label == 0], g[g.label == 1]
            out.append({'predictor': pname, 'n': len(g), 'acc_strict': round((g.strict == g.label).mean(), 3),
                        'KHA_THI_GIA(strict)': int(((g.strict == 0) & (g.label == 1)).sum()) if len(pos) else 0,
                        'BAO_DONG_GIA(strict)': int(((g.strict == 1) & (g.label == 0)).sum()) if len(neg) else 0,
                        'acc_cautious': round((g.cautious == g.label).mean(), 3),
                        'KHA_THI_GIA(cautious)': int(((g.cautious == 0) & (g.label == 1)).sum())})
        print('\n[2] PHAN QUYET tren luoi tai (nhan: L<=lo kha thi, L>=hi khong; bo qua khoang mo ho). strict = chi INFEASIBLE la "khong"; cautious = ca MARGINAL')
        print(pd.DataFrame(out).to_string(index=False))

    # ---------- 3. muc su dung tung node truoc diem gay ----------
    nrows = []
    for r in runs:
        lo = r['bp'].get('lo')
        if lo is None:
            continue
        d = r['df']
        d = d[(d['step_warm'] == 0) & (d['target_rps'] <= lo)]
        for pname in (['base'] if r['feature'] == 'base' else PN):
            p = preds.get((pname, r['feature'], r['scale'] if r['feature'] != 'base' else 1.0))
            if not p:
                continue
            # vai tro node theo chain THAT (cua taxonomy), giong nhau cho moi bo du doan -> so sanh cung tap node
            true_chain = set(preds[('P1', r['feature'], r['scale'])]['chain']) if r['feature'] != 'base' else set()
            for L, g in d.groupby('target_rps'):
                pu = p['grid'].get(str(int(L)), {}).get('u')
                if not pu:
                    continue
                for s, u_pred in pu.items():
                    u_meas = float(g[f'{s}_cpu'].mean()) / (100.0 * cores[s])
                    nrows.append({'predictor': pname, 'feature': r['feature'], 'scale': r['scale'], 'L': int(L), 'node': s,
                                  'role': ('gateway' if s == 'front-end' else ('chain' if s in true_chain else 'non-chain')), 'u_pred': u_pred, 'u_meas': round(u_meas, 4),
                                  'abs_err_pts': abs(u_pred - u_meas) * 100, 'rel_err_%': 100 * (u_pred - u_meas) / u_meas if u_meas else np.nan})
    if nrows:
        n = pd.DataFrame(nrows)
        n_feat = n[n.feature != 'base']
        if len(n_feat):
            print('\n[3] MUC SU DUNG TUNG NODE, cac buoc TRUOC diem gay (sai so tuyet doi, diem % cua tran; thap hon = tot hon)')
            t = n_feat.groupby(['feature', 'predictor'])['abs_err_pts'].agg(['mean', 'median', 'count']).round(2)
            print(t.to_string())
            print('\n    theo VAI TRO node (chain THAT cua taxonomy; gateway = front-end, sai giong nhau cho moi bo do chi phi dieu phoi):')
            print(n_feat.groupby(['feature', 'role', 'predictor'])['abs_err_pts'].mean().round(2).unstack().to_string())
            print('\n    sai so tuong doi trung binh (%) theo node, tat ca tinh nang:')
            print(n_feat.groupby(['node', 'predictor'])['rel_err_%'].mean().round(1).unstack().to_string())
        nb = n[n.feature == 'base']
        if len(nb):
            print('\n    baseline (sai so tuyet doi trung binh, diem %):', nb.groupby('node')['abs_err_pts'].mean().round(2).to_dict())
        if a.out:
            n.to_csv(a.out, index=False)
            print(f'\n[OK] chi tiet tung node -> {a.out}')
    print('\nLUU Y: ket qua tren la cua ban DONG BANG, khong duoc chinh mo hinh de cai thien roi coi la du doan truoc.')


if __name__ == '__main__':
    main()
