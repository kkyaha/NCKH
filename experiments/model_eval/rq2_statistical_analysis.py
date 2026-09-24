# -*- coding: utf-8 -*-
"""
RQ2 statistical analysis, corrected
===================================
Thay cho cac ham kiem dinh rai rac trong evaluation_suite.py va
trainticket_evaluation.py, script nay lam DUNG quy trinh Demsar (2006) cho ca
hai testbed va ca hai bien the SCM:

  1. Friedman omnibus tren toan bo model. Neu KHONG bac bo -> khong duoc phep
     dien giai bat ky so sanh cap nao (truoc day paper bao cao p=0.0137 cho
     Sock Shop trong khi Friedman p=0.156).
  2. Wilcoxon signed-rank theo cap, pooled va tach theo tung metric.
  3. Hieu chinh Holm trong TUNG ho gia thuyet cua tung he thong (truoc day
     khong hieu chinh gi ca, du chay ~12 kiem dinh moi he thong).
  4. Effect size dua tren HANG, bat bien voi moi phep doi thang do don dieu
     cua MAPE: rank-biserial correlation r va Cliff's delta. Can thiet vi MAPE
     rat nhay voi mau so nho (Hyndman & Koehler 2006) va nhieu node o day co
     baseline gan 0.
  5. Dem so lan thang CO XU LY HOA (truoc day idxmin am tham pha hoa theo thu
     tu cot, lam sai lech dem tren Train Ticket noi co 20 cap hoa).

Output: data/processed/scm_results/rq2_corrected_statistics.csv
"""

import os
import sys
import math

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')

SYSTEMS = {
    'SockShop': '05_model_comparison.csv',
    'TrainTicket': 'trainticket_model_comparison.csv',
}
BASELINES = ['LinearReg', 'GradBoost', 'GaussianProcess']
SCM_VARIANTS = ['SCM_Deployed', 'SCM_Auto']
VALUE_COL = 'mape_full_res_pct'
ALPHA = 0.05


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation. Duong = SCM co MAPE THAP hon."""
    d = np.asarray(d, dtype=float)
    d = d[(d != 0) & np.isfinite(d)]
    if d.size == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    r_pos, r_neg = r[d > 0].sum(), r[d < 0].sum()
    return (r_neg - r_pos) / (r_pos + r_neg)


def cliffs_delta(a, b):
    """Duong = SCM tot hon (MAPE thap hon) tren nhieu cap hon."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    if n == 0:
        return np.nan
    return (np.sum(a < b) - np.sum(a > b)) / n


def holm(pvals, labels):
    """Holm-Bonferroni step-down, ap dung trong TUNG ho gia thuyet."""
    order = np.argsort(pvals)
    m = len(pvals)
    out, running = {}, 0.0
    for i, idx in enumerate(order):
        adj = min(1.0, (m - i) * pvals[idx])
        running = max(running, adj)   # dam bao don dieu
        out[labels[idx]] = running
    return out


def analyse(system, path):
    df = pd.read_csv(path)
    piv = df.pivot_table(index=['service', 'metric'], columns='model',
                         values=VALUE_COL).dropna()
    models = [c for c in piv.columns]
    rows = []

    chi2, p_omni = stats.friedmanchisquare(*[piv[c].values for c in models])
    omnibus_ok = p_omni < ALPHA
    print(f"\n{'='*78}\n{system}: n={len(piv)} service-metric pairs, models={models}")
    print(f"  Friedman omnibus: chi2={chi2:.3f}, p={p_omni:.4g} "
          f"-> {'REJECT (post-hoc licensed)' if omnibus_ok else 'NOT rejected (post-hoc NOT licensed)'}")

    # --- dem so lan thang, co xu ly hoa ---
    mins = piv.min(axis=1)
    is_min = piv.eq(mins, axis=0)
    n_tied_rows = int((is_min.sum(axis=1) > 1).sum())
    sole = (is_min & (is_min.sum(axis=1) == 1).values[:, None]).sum()
    print(f"  Sole-winner counts: {sole.to_dict()}  (+{n_tied_rows} tied rows)")

    # --- Wilcoxon theo cap, gom thanh ho gia thuyet rieng cho tung bien the SCM ---
    for scm in SCM_VARIANTS:
        if scm not in piv.columns:
            continue
        raw, labels, recs = [], [], []
        for bl in BASELINES:
            for scope in ['pooled'] + sorted(df['metric'].unique()):
                sub = piv if scope == 'pooled' else \
                    piv[piv.index.get_level_values('metric') == scope]
                if len(sub) < 5:
                    continue
                a, b = sub[scm].values, sub[bl].values
                d = a - b
                n_ident = int(np.sum(np.isclose(d, 0)))
                if np.allclose(d, 0):
                    # moi cap giong het nhau -> Wilcoxon khong xac dinh
                    recs.append(dict(comparison=f'{scm}_vs_{bl}', scope=scope, n=len(sub),
                                     n_identical=n_ident, median_delta=0.0, p=np.nan,
                                     r=np.nan, cliff=0.0, wins=0))
                    continue
                stat, p = stats.wilcoxon(a, b)
                lab = f'{scm}_vs_{bl}|{scope}'
                raw.append(p); labels.append(lab)
                recs.append(dict(comparison=f'{scm}_vs_{bl}', scope=scope, n=len(sub),
                                 n_identical=n_ident, median_delta=float(np.median(d)),
                                 p=float(p), r=rank_biserial(d),
                                 cliff=cliffs_delta(a, b),
                                 wins=int(np.sum(a < b))))
        adj = holm(raw, labels) if raw else {}
        print(f"\n  --- {scm} (Holm family size = {len(raw)}) ---")
        for rec in recs:
            lab = f"{rec['comparison']}|{rec['scope']}"
            p_adj = adj.get(lab, np.nan)
            sig = ('***' if p_adj < 0.001 else '**' if p_adj < 0.01
                   else '*' if p_adj < 0.05 else 'ns') if np.isfinite(p_adj) else '--'
            rows.append(dict(system=system, omnibus_chi2=round(chi2, 3),
                             omnibus_p=round(p_omni, 6),
                             omnibus_rejects=omnibus_ok, **rec,
                             p_holm=p_adj, verdict=sig))
            pstr = f"{rec['p']:.4g}" if np.isfinite(rec['p']) else 'undef'
            astr = f"{p_adj:.4g}" if np.isfinite(p_adj) else '  --'
            rstr = f"{rec['r']:+.3f}" if np.isfinite(rec['r']) else '  n/a'
            print(f"    {rec['comparison']:26s}|{rec['scope']:7s} n={rec['n']:3d} "
                  f"ident={rec['n_identical']:2d} medD={rec['median_delta']:+8.3f} "
                  f"p={pstr:>9s} p_holm={astr:>9s} {sig:3s} r={rstr} "
                  f"wins={rec['wins']}/{rec['n']}")
    return rows


def main():
    all_rows = []
    for system, fname in SYSTEMS.items():
        path = os.path.join(OUT_DIR, fname)
        if not os.path.exists(path):
            print(f"  [skip] {path} khong ton tai")
            continue
        all_rows += analyse(system, path)
    out = pd.DataFrame(all_rows)
    dest = os.path.join(OUT_DIR, 'rq2_corrected_statistics.csv')
    out.to_csv(dest, index=False)
    print(f"\n[OK] Da luu: {dest}  ({len(out)} dong)")


if __name__ == '__main__':
    main()
