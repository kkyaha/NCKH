# -*- coding: utf-8 -*-
"""
BO DU DOAN CO THUC SU DU BAO TOT KHONG? -- kiem tra truoc khi bao cao
=====================================================================
Mot con so "sai so trung binh 12,6%" TU NO khong noi len dieu gi. No chi co nghia khi tra loi duoc
ba cau hoi, va script nay tra loi ca ba tren du lieu vong tien cuu 2 (n = 8 o doc lap):

  [1] Co HON du doan tam thuong khong? So voi hai moc nen KHONG dung do thi goi nao:
        * `const_base`  : du doan diem gay cua baseline cho MOI tinh nang (bo qua tinh nang han)
        * `const_oracle`: hang so TOT NHAT co the (trung vi cua chinh dap an) -- day la moc GIAN LAN,
          dung lam CAN DUOI cho mo hinh bat ky hang so nao; thua no moi dang ke.
  [2] Chenh lech co vung khong? Wilcoxon ghep cap tren cung 8 o (n nho nen bao ca san cua kiem dinh).
  [3] Co vuot duoc SAN NHIEU cua phep do khong? Hai lan lap cua cung mot o thuong lech mot bac luoi;
      luoi hinh hoc ×1,25 nen mot bac ~ 25%. Neu sai so mo hinh xap xi sai so lap lai thi DU LIEU
      KHONG DU PHAN GIAI de noi mo hinh nao tot hon -- phai noi ro thay vi khoe con so.

    python experiments/feasibility/assess_predictive_performance.py

LUU Y: day la DANH GIA cac du doan DA DONG BANG; khong co tham so nao duoc chinh o day.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'experiments', 'collect'))


def measured(ramp_dir, features, anchors):
    """lo/hi do duoc cho tung o x1, GIU ca tung lan lap de uoc luong san nhieu."""
    cells = {}
    for sp in sorted(glob.glob(os.path.join(BASE, ramp_dir, 'ramp_*', 'run*', 'steps.json'))):
        j = json.load(open(sp, encoding='utf-8'))
        feat = j.get('feature', 'base')
        if feat not in features and feat != 'base':
            continue
        pct = float(j.get('feature_pct', 0) or 0)
        sc = round(pct / anchors[feat], 3) if feat != 'base' and anchors.get(feat) else 1.0
        if sc != 1.0:
            continue                      # cuong do x2 khong doc lap voi x1 -> bo
        b = j['breakpoint']
        cells.setdefault(feat, {'lo': [], 'hi': []})
        cells[feat]['lo'].append(b['lo'])
        if b['hi'] is not None:
            cells[feat]['hi'].append(b['hi'])
    return cells


def frozen_preds(features):
    """P1/P2 tu ban dong bang truoc ramp; P3 tu ban dong bang rieng tung tinh nang."""
    out = {}
    for tag in ('prosp2', 'prosp3'):
        p = os.path.join(BASE, 'data', 'processed', 'frozen', f'predictions_frozen_RE2_P2_{tag}.json')
        if not os.path.exists(p):
            continue
        for pr in json.load(open(p, encoding='utf-8'))['predictions']:
            if pr['feature'] in features and round(float(pr['scale']), 3) == 1.0:
                out.setdefault(pr['feature'], {})[pr['predictor']] = pr['breakpoint_rps']
    for f in features:
        p = os.path.join(BASE, 'data', 'processed', 'frozen', f'predictions_frozen_RE2_P3_prosp_{f}.json')
        if os.path.exists(p):
            for pr in json.load(open(p, encoding='utf-8'))['predictions']:
                if pr['feature'] == f and round(float(pr.get('scale', 1.0)), 3) == 1.0:
                    out.setdefault(f, {})['P3'] = pr['breakpoint_rps']
    return out


def wilcoxon(a, b):
    try:
        from scipy import stats
        d = np.asarray(a) - np.asarray(b)
        if not np.any(d):
            return None, None
        return stats.wilcoxon(a, b)
    except ImportError:
        return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='login,register,wishlist,catsearch,account,preview,orderhist,related')
    ap.add_argument('--ramp-dir', default='data/raw/SS-PROSP2')
    a = ap.parse_args(argv)

    import load_sweep_collect as L
    feats = [f.strip() for f in a.features.split(',') if f.strip()]
    anchors = {f: L.FEATURES[f]['anchor'] for f in feats}
    cells = measured(a.ramp_dir, feats, anchors)
    preds = frozen_preds(feats)

    base_lo = float(np.median(cells['base']['lo']))
    meas = {f: float(np.median(cells[f]['lo'])) for f in feats}
    oracle = float(np.median(list(meas.values())))

    rows = []
    for f in feats:
        m = meas[f]
        r = {'cell': f, 'do_duoc_lo': m, 'n_lap': len(cells[f]['lo'])}
        for name, p in (('P1', preds[f].get('P1')), ('P2', preds[f].get('P2')), ('P3', preds[f].get('P3')),
                        ('const_base', base_lo), ('const_oracle', oracle)):
            r[name] = p
            r[f'e_{name}'] = abs(100.0 * (p - m) / m) if p else None
            r[f'abs_{name}'] = abs(p - m) if p else None
        rows.append(r)
    d = pd.DataFrame(rows)
    pd.set_option('display.width', 240)

    print(f'\n=== [0] DU DOAN vs DO DUOC (n = {len(d)} o doc lap, cuong do x1) ===')
    print(f'    baseline do duoc = {base_lo:g} req/s | hang so oracle (trung vi dap an) = {oracle:g} req/s')
    print(d[['cell', 'do_duoc_lo', 'P1', 'P2', 'P3', 'const_base', 'const_oracle']].to_string(index=False))

    print('\n=== [1] SAI SO TUYET DOI TRUNG BINH (%) -- thap hon la tot hon ===')
    models = ['P1', 'P2', 'P3', 'const_base', 'const_oracle']
    summ = pd.DataFrame({
        'sai_so_TB_%': [d[f'e_{m}'].mean() for m in models],
        'trung_vi_%': [d[f'e_{m}'].median() for m in models],
        'xau_nhat_%': [d[f'e_{m}'].max() for m in models],
        'sai_so_TB_req/s': [d[f'abs_{m}'].mean() for m in models],
    }, index=models).round(1)
    print(summ.to_string())

    best_naive = min(summ.loc['const_base', 'sai_so_TB_%'], summ.loc['const_oracle', 'sai_so_TB_%'])
    print(f"\n    Moc nen tot nhat trong hai hang so: {best_naive:.1f}%"
          f" | P3: {summ.loc['P3', 'sai_so_TB_%']:.1f}%"
          f" -> {'P3 THUA moc nen' if summ.loc['P3', 'sai_so_TB_%'] < best_naive else 'P3 KHONG thua moc nen'}")

    print('\n=== [2] KIEM DINH GHEP CAP (Wilcoxon, cung 8 o) ===')
    pairs = [('P1', 'P2'), ('P2', 'P3'), ('P2', 'const_base'), ('P3', 'const_base'),
             ('P2', 'const_oracle'), ('P3', 'const_oracle')]
    for x, y in pairs:
        st, p = wilcoxon(d[f'e_{x}'].values, d[f'e_{y}'].values)
        if p is None:
            print(f'    {x} vs {y}: khong tinh duoc (khong co scipy hoac sai so trung nhau)')
        else:
            better = x if d[f'e_{x}'].mean() < d[f'e_{y}'].mean() else y
            print(f'    {x} vs {y}: W = {st:g}, p = {p:.4f}  ({better} tot hon ve trung binh)')
    print(f'    San cua kiem dinh: voi n = {len(d)}, p hai phia NHO NHAT ma Wilcoxon dat duoc la '
          f'{2 / 2 ** len(d):.4f}.')

    print('\n=== [3] SAN NHIEU CUA PHEP DO (lech giua hai lan lap cung mot o) ===')
    sp = []
    for f in feats + ['base']:
        los = cells[f]['lo']
        if len(los) >= 2:
            m = float(np.median(los))
            sp.append({'cell': f, 'n_lap': len(los), 'lo_cac_lan': los,
                       'bien_do_%': round(100.0 * (max(los) - min(los)) / m, 1)})
    s = pd.DataFrame(sp)
    print(s.to_string(index=False))
    noise = s['bien_do_%'].median()
    print(f"\n    Bien do lap lai trung vi = {noise:.1f}% (luoi hinh hoc x1,25 -> mot bac ~ 25%)")
    p3e = summ.loc['P3', 'sai_so_TB_%']
    print(f"    Sai so P3 = {p3e:.1f}% so voi san nhieu {noise:.1f}%: "
          + ('SAI SO O MUC NHIEU -- du lieu KHONG du phan giai de xep hang cac mo hinh.'
             if p3e <= noise else 'sai so LON HON nhieu do, nen so sanh co y nghia.'))

    print('\n=== [4] HUONG NGUY HIEM (bao song sot khi he DA vo) ===')
    for m in ('P1', 'P2', 'P3'):
        bad = [f for f in feats if cells[f]['hi'] and d.loc[d.cell == f, m].iloc[0] > float(np.median(cells[f]['hi']))]
        print(f'    {m}: {len(bad)}/{len(feats)} o' + (f" -> {', '.join(bad)}" if bad else ''))
    print('\n    Sai so trung binh va huong nguy hiem la HAI thu khac nhau: mot bo du doan co sai so'
          '\n    trung binh dep van co the bao KHA THI o dung muc tai da lam vo SLO.')
    return d


if __name__ == '__main__':
    main()
