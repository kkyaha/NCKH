# -*- coding: utf-8 -*-
"""
P3 -- P2 + BOI SO GOI (k_s) DO DUOC, THAY VI GIA DINH k_s=1 (CHUAN DOAN HOI CUU / RETROSPECTIVE)
==================================================================================================
Khong phai mot lop mo hinh moi: FeasibilityPredictor.workloads(mode='P2', k=...) DA NHAN tham so k
tu truoc (xem feasibility_predictor.py) -- freeze_predictions.py don gian chua bao gio truyen no
(mac dinh k=1 cho moi node trong chain). P3 = P2 (giu NGUYEN he so chi phi c_s, x da fit tren
promo/recs) + k do duoc bang probing chuc nang nhe (experiments/probe_feature_chain.py --save),
KHONG dung du lieu tai/diem gay -- ve nguyen tac co the do TRUOC khi dua ra phan quyet that.

QUAN TRONG -- day la CHUAN DOAN HOI CUU, KHONG phai du doan dong bang moi: du lieu doc lap
(cartsum/quickadd/express) DA duoc do va DA duoc xem (evaluate_frozen.py --split indep chay
truoc do). Script nay tra loi "neu do k truoc khi phan quyet thi loi co giam khong?", dung de
CHAN DOAN co che, khong dung de tuyen bo do chinh xac tien cuu moi (can mot vong du lieu doc lap
THU TU moi, chua tung xem, de kiem chung P3 tien cuu -- xem docs/DATA_FRAMEWORK.md).

Hai bien the:
  P3   : giu chain TAXONOMY (khong doi tap node bi anh huong), chi sua k cho node DA co trong chain
  P3+chain : chain = tap DO DUOC (vd cartsum them catalogue ma taxonomy khong co), cong voi k do duoc

    python experiments/evaluate_p3.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(BASE, 'src', 'scm'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feasibility_predictor as FP  # noqa: E402
import evaluate_frozen as EF  # noqa: E402

FROZEN_DIR = os.path.join(BASE, 'data', 'processed', 'frozen')


def load_predictor():
    base = json.load(open(os.path.join(FROZEN_DIR, 'predictions_frozen_RE2.json'), encoding='utf-8'))
    p2 = json.load(open(os.path.join(FROZEN_DIR, 'p2_params_dev.json'), encoding='utf-8'))
    kmeas = json.load(open(os.path.join(FROZEN_DIR, 'k_measured.json'), encoding='utf-8'))['features']
    kmeas.update(json.load(open(os.path.join(FROZEN_DIR, 'k_measured_prosp.json'), encoding='utf-8'))['features'])
    P = FP.FeasibilityPredictor(base['mechanism'], base['params']['cores'], base['params']['u_star'],
                                feature_cost=p2['params'])
    return P, kmeas


def eval_set(P, kmeas, ramp_dir, features, label):
    runs = EF.load_ramps(ramp_dir, features)
    runs = [r for r in runs if r['feature'] != 'base' and r['bp'].get('lo') is not None]
    rows, node_rows = [], []
    cells = sorted({(r['feature'], r['scale']) for r in runs})
    for feat, sc in cells:
        rs = [r for r in runs if (r['feature'], r['scale']) == (feat, sc)]
        lo = float(np.median([r['bp']['lo'] for r in rs]))
        his = [r['bp']['hi'] for r in rs if r['bp'].get('hi') is not None]
        hi = float(np.median(his)) if his else None
        tax_chain = FP.SOCKSHOP_CALL_CHAINS[FP.FEATURE_ARCHETYPE[feat]]['services']
        meas = kmeas[feat]
        k = meas['measured_per_use']
        variants = {
            'P1 (k=1)':        dict(mode='P1'),
            'P2 (c,x; k=1)':   dict(mode='P2'),
            'P3 (k do duoc)':  dict(mode='P2', k=k),
            'P3+chain':        dict(mode='P2', k=k, chain=meas['chain_measured']),
        }
        for name, kw in variants.items():
            rp, node = P.breakpoint(feature=feat, scale=sc, **kw)
            cls = 'MUON (kha thi gia)' if (hi is not None and rp >= hi) else ('TRONG khoang' if rp >= lo else 'SOM (an toan)')
            rows.append({'set': label, 'cell': f'{feat}x{sc:g}', 'model': name, 'R*_pred': round(rp, 1),
                        'node_nghen': node, 'do_duoc_lo': lo, 'hi': hi, 'sai_so_%': round(100 * (rp - lo) / lo, 1),
                        'phan_loai': cls})
            for r in rs:
                d = r['df']
                d = d[(d['step_warm'] == 0) & (d['target_rps'] <= lo)]
                for L, g in d.groupby('target_rps'):
                    u = P.utilization(L, feature=feat, scale=sc, **kw)
                    for s in FP.SCORED:
                        u_meas = float(g[f'{s}_cpu'].mean()) / P.cap[s]
                        node_rows.append({'set': label, 'feature': feat, 'model': name, 'node': s,
                                          'role': 'gateway' if s == 'front-end' else ('chain' if s in tax_chain else 'non-chain'),
                                          'abs_err_pts': abs(u[s] - u_meas) * 100})
    return pd.DataFrame(rows), pd.DataFrame(node_rows)


def main():
    P, kmeas = load_predictor()
    print('\n=== He so k DO DUOC (bang probing chuc nang, doc lap voi du lieu tai) ===')
    for f, m in kmeas.items():
        extra = set(m['chain_measured']) - set(m['chain_taxonomy'])
        missing = set(m['chain_taxonomy']) - set(m['chain_measured'])
        print(f"  {f:9s} k={ {s: v for s, v in m['measured_per_use'].items() if v} }"
              + (f"  [chain THEM: {sorted(extra)}]" if extra else '')
              + (f"  [chain THIEU: {sorted(missing)}]" if missing else ''))

    all_bp, all_node = [], []
    for ramp_dir, feats, label in [
            (os.path.join(BASE, 'data', 'raw', 'SS-LIMITS-CLEAN'), EF.INDEP, 'DOC LAP (chinh, du lieu SACH)'),
            (os.path.join(BASE, 'data', 'raw', 'SS-LIMITS-CLEAN'), EF.PROSP, 'TIEN CUU (browse, du lieu SACH)'),
            (os.path.join(BASE, 'data', 'raw', 'SS-LIMITS'), ('promo', 'recs'), 'dev (kiem tra khong hoi quy)'),
            (os.path.join(BASE, 'data', 'raw', 'SS-LIMITS'), ('track', 'review'), 'khoa (k~1 do duoc, ky vong khong doi)')]:
        bp, nd = eval_set(P, kmeas, ramp_dir, feats, label)
        all_bp.append(bp)
        all_node.append(nd)
    bp = pd.concat(all_bp, ignore_index=True)
    node = pd.concat(all_node, ignore_index=True)

    for label in bp['set'].unique():
        print(f'\n=== DIEM GAY -- {label} ===')
        t = bp[bp['set'] == label].pivot(index='cell', columns='model', values='R*_pred')
        lohi = bp[bp['set'] == label].drop_duplicates('cell').set_index('cell')[['do_duoc_lo', 'hi']]
        print(pd.concat([lohi, t], axis=1).to_string())
        err = bp[bp['set'] == label].pivot(index='cell', columns='model', values='sai_so_%')
        print('\n  sai so % so voi lo (duong = muon hon, nguy hiem):')
        print(err.to_string())

    print('\n=== TOM TAT: |sai so %| trung binh theo mo hinh va tap ===')
    print(bp.groupby(['set', 'model'])['sai_so_%'].apply(lambda x: round(x.abs().mean(), 1)).unstack().to_string())

    print('\n=== Sai so muc su dung node (diem % tran, truoc diem gay), theo vai tro ===')
    print(node.groupby(['set', 'role', 'model'])['abs_err_pts'].mean().round(2).unstack().to_string())

    out = os.path.join(FROZEN_DIR, 'p3_evaluation.csv')
    bp.to_csv(out, index=False)
    print(f'\n[OK] -> {out}')
    print('\nLUU Y: day la chuan doan HOI CUU tren du lieu DA XEM (indep + du lieu Pha B cu). '
          'Khong the dung nhu bang chung tien cuu cho P3 -- can mot vong du lieu doc lap MOI, '
          'chua tung do, de kiem chung P3 nhu mot du doan dong bang that su.')


if __name__ == '__main__':
    main()
