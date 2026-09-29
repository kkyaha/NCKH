# -*- coding: utf-8 -*-
"""HIEU CHINH FAMILY-WISE + KHOANG TIN CAY cho bo kiem dinh da mo rong
=======================================================================
Bai da tu dat chuan o Section "Benchmarks and Statistical Methodology": Friedman
omnibus truoc, Wilcoxon ghep cap CHI khi omnibus cho phep, moi p hieu chinh Holm,
va kem effect size theo HANG (rank-biserial ghep cap + Cliff's delta) vi chung bat
bien voi moi phep bien doi don dieu cua thang sai so.

Cac kiem dinh moi phat sinh khi mo rong (so sanh mo hinh tren hai bo du lieu, thien
lech cua chi bao nghen) CHUA tung duoc hieu chinh. Script nay lam ba viec:

  [1] So sanh mo hinh: Friedman -> Wilcoxon ghep cap -> Holm, TREN TUNG BO DU LIEU
      (moi bo la mot HO kiem dinh rieng; khong gop hai bo vao mot ho vi chung tra
      loi hai cau hoi khac nhau).
  [2] Khoang tin cay bootstrap cho sai so diem gay theo tung split. Voi split n<=2
      KHONG bao cao khoang -- bootstrap tren 2 diem khong co y nghia, va bao cao mot
      khoang gia o do la sai lech nang hon la khong bao cao.
  [3] Thien lech cua chi bao nghen (CPU vs throttling): ho 3 kiem dinh, hieu chinh Holm.

    python experiments/statistical_rigor.py
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from scipy.optimize import nnls
from scipy.stats import friedmanchisquare, wilcoxon, mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
import _paths  # noqa: F401  -- dua cac nhom con khac vao sys.path
sys.path.insert(0, os.path.join(BASE, 'src', 'scm'))
import feasibility_predictor as FP  # noqa: E402
from queueing_regressor import QueueingLatencyRegressor  # noqa: E402
from data_processor import load_normal_data  # noqa: E402
import evaluate_frozen as EF  # noqa: E402

pd.set_option('display.width', 230)
SEED, N_FIT = 42, 2000
SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']


# ---------------------------------------------------------------- thong ke
def holm(pvals, labels):
    """Holm-Bonferroni. Tra ve dict label -> (p_tho, p_hieu_chinh, bac_bo_o_0.05)."""
    order = np.argsort(pvals)
    m, out, running = len(pvals), {}, 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)              # bat buoc don dieu
        out[labels[i]] = (pvals[i], running, running < 0.05)
    return out


def rank_biserial(x, y):
    """Rank-biserial ghep cap: (T+ - T-)/(T+ + T-). Duong = x LON hon y."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if not len(d):
        return 0.0
    r = pd.Series(np.abs(d)).rank().to_numpy()
    tp, tn = r[d > 0].sum(), r[d < 0].sum()
    return float((tp - tn) / (tp + tn))


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return float((gt - lt) / (len(x) * len(y)))


def boot_ci(vals, n_boot=10000, alpha=0.05, seed=SEED):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if len(v) < 3:
        return (np.nan, np.nan)                  # n<=2: khong bao cao khoang
    rs = np.random.RandomState(seed)
    m = [np.mean(rs.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


# ---------------------------------------------------------------- [1] so sanh mo hinh
class NNLSReg:
    def fit(self, X, y):
        (self.a_, self.b_), _ = nnls(np.column_stack([np.ones(len(X)), X.ravel()]), y)
        return self

    def predict(self, X):
        return self.a_ + self.b_ * X.ravel()


def models():
    return {'NNLS(SCM)': NNLSReg(), 'LinearReg': LinearRegression(),
            'GradBoost': GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                   learning_rate=0.05, random_state=SEED),
            'RandForest': RandomForestRegressor(n_estimators=200, random_state=SEED),
            'GaussProc': Pipeline([('s', StandardScaler()),
                                   ('g', GaussianProcessRegressor(
                                       kernel=ConstantKernel(1.) * RBF(1.) + WhiteKernel(.1),
                                       n_restarts_optimizer=1, alpha=1e-3, normalize_y=True))]),
            'Queueing': QueueingLatencyRegressor()}


def mape(y, p):
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.sum() else np.nan


def eval_dataset(cells, name):
    rows = []
    for svc, w, y in cells:
        o = np.argsort(w)
        w, y = w[o], y[o]
        n = int(len(w) * 0.67)
        rs = np.random.RandomState(SEED)
        itr = np.arange(n)
        if n > N_FIT:
            itr = rs.choice(itr, N_FIT, replace=False)
        Xtr, ytr, Xte, yte = w[itr].reshape(-1, 1), y[itr], w[n:].reshape(-1, 1), y[n:]
        if len(yte) < 10 or np.std(ytr) == 0:
            continue
        r = {'service': svc}
        for nm, mdl in models().items():
            try:
                mdl.fit(Xtr, ytr)
                r[nm] = mape(yte, np.asarray(mdl.predict(Xte)).ravel())
            except Exception:
                r[nm] = np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index('service')


def report_family(M, name):
    M = M.dropna(axis=1, how='any')
    print(f'\n--- HO KIEM DINH: {name}  (n = {len(M)} service) ---')
    print('  MAPE trung vi: ' + ', '.join(f'{c}={M[c].median():.2f}%' for c in M.columns))
    stat, p_om = friedmanchisquare(*[M[c].values for c in M.columns])
    print(f'  Friedman omnibus: chi2={stat:.3f}, p={p_om:.4f}'
          f'  -> {"CHO PHEP post-hoc" if p_om < 0.05 else "KHONG cho phep post-hoc (dung lai o day)"}')
    if p_om >= 0.05:
        return
    ref = 'NNLS(SCM)'
    others = [c for c in M.columns if c != ref]
    ps = [wilcoxon(M[ref].values, M[c].values).pvalue for c in others]
    adj = holm(ps, others)
    print(f'  Wilcoxon ghep cap {ref} vs tung baseline, hieu chinh Holm (m={len(others)}):')
    print(f"    {'baseline':12s} {'trung vi hieu':>14s} {'p tho':>8s} {'p Holm':>8s} {'bac bo':>7s} "
          f"{'rank-biserial':>14s} {'Cliff d':>8s}")
    n_min_p = 2 ** -(len(M) - 1)
    for c in others:
        p0, pa, rej = adj[c]
        d = np.median(M[ref].values - M[c].values)
        print(f'    {c:12s} {d:+14.2f} {p0:8.4f} {pa:8.4f} {str(rej):>7s} '
              f'{rank_biserial(M[ref].values, M[c].values):+14.3f} '
              f'{cliffs_delta(M[ref].values, M[c].values):+8.3f}')
    print(f'  ⚠ San cua kiem dinh: voi n={len(M)}, p hai phia NHO NHAT ma Wilcoxon dat duoc la '
          f'{n_min_p:.4f}. Moi p bang gia tri nay chi co nghia "tat ca {len(M)} cung huong".')


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', default='SS-LIMITS,SS-LIMITS-CLEAN',
                    help='thu muc ramp duoi data/raw/ dung cho phan [2]; them SS-PROSP2 cho vong tien cuu 2')
    ap.add_argument('--k-file', action='append', default=[], metavar='JSON',
                    help='file k do duoc THEM (vd k_measured_v2.json); lap nhieu lan')
    ap.add_argument('--extra-split', action='append', default=[], metavar='NHAN=f1,f2',
                    help='them mot tap vao phan [2] (vd "tien cuu 2=login,register")')
    ap.add_argument('--extra-x1-only', action='store_true',
                    help='tap --extra-split chi tinh o x1 (MOT o moi tinh nang DOC LAP): cac cuong do cua cung tinh nang '
                         'dung chung k/chain/ban cai nen KHONG phai quan sat doc lap (pseudo-replication)')
    a = ap.parse_args(argv)

    print('=' * 96)
    print('  [1] SO SANH MO HINH -- Friedman -> Wilcoxon -> Holm, moi bo du lieu la MOT ho')
    print('=' * 96)
    SS = FP.select_train(FP.load_runs(os.path.join(BASE, 'data', 'raw', 'SS-TRAIN')), train_max=10 ** 9)
    for metric in ('cpu', 'mem'):
        ss_cells, re_cells = [], []
        for s in SERVICES:
            col = f'{s}_{metric}'
            if col in SS:
                d = SS[[f'{s}_workload', col]].dropna()
                if len(d) > 50:
                    ss_cells.append((s, d[f'{s}_workload'].to_numpy(float), d[col].to_numpy(float)))
            r = load_normal_data(s, metric)
            if r is not None and len(r) > 50:
                re_cells.append((s, r['Workload'].to_numpy(float), r['Target'].to_numpy(float)))
        for cells, nm in ((re_cells, f'RE2-SS (RCAEval) / {metric.upper()}'),
                          (ss_cells, f'SS-TRAIN (tu do) / {metric.upper()}')):
            if cells:
                report_family(eval_dataset(cells, nm), nm)

    print('\n' + '=' * 96)
    print('  [2] KHOANG TIN CAY BOOTSTRAP cho sai so diem gay (10.000 lan, phan vi 2.5-97.5)')
    print('=' * 96)
    fz = json.load(open(os.path.join(BASE, 'data', 'processed', 'frozen',
                                     'predictions_frozen_RE2.json'), encoding='utf-8'))
    p2 = json.load(open(os.path.join(BASE, 'data', 'processed', 'frozen',
                                     'p2_params_dev.json'), encoding='utf-8'))['params']
    KM = json.load(open(os.path.join(BASE, 'data', 'processed', 'frozen',
                                     'k_measured.json'), encoding='utf-8'))['features']
    KM.update(json.load(open(os.path.join(BASE, 'data', 'processed', 'frozen',
                                          'k_measured_prosp.json'), encoding='utf-8'))['features'])
    P = FP.FeasibilityPredictor(fz['mechanism'], fz['params']['cores'], fz['params']['u_star'],
                                feature_cost=p2)
    for kf in a.k_file:                          # vd k_measured_v2.json cua vong tien cuu 2
        KM.update(json.load(open(kf, encoding='utf-8'))['features'])
    cells = {}
    for root in [x.strip() for x in a.roots.split(',') if x.strip()]:
        for sp in sorted(glob.glob(os.path.join(BASE, 'data', 'raw', root, 'ramp_*', 'run*', 'steps.json'))):
            j = json.load(open(sp, encoding='utf-8'))
            feat = j.get('feature', 'base')
            if feat == 'base' or j['breakpoint']['lo'] is None:
                continue
            sc = 2.0 if os.path.basename(os.path.dirname(os.path.dirname(sp))).endswith('x2') else 1.0
            cells.setdefault((feat, sc), []).append(j['breakpoint']['lo'])
    SPLITS = [('dev', ['promo', 'recs']), ('khoa', ['track', 'review']),
              ('doc lap', ['cartsum', 'quickadd', 'express']), ('tien cuu', ['browse'])]
    for spec in a.extra_split:
        if '=' not in spec:
            sys.exit(f'--extra-split sai dang: {spec!r}; can NHAN=f1,f2')
        nm, fl = spec.split('=', 1)
        SPLITS.append((nm.strip() + (' [x1]' if a.extra_x1_only else ''), [x.strip() for x in fl.split(',') if x.strip()]))
    print(f"\n  {'split':10s} {'n':>2s} {'mo hinh':12s} {'|sai so| TB':>12s} {'KTC 95%':>20s}")
    for nm, feats in SPLITS:
        errs = {'P1': [], 'P2': [], 'P3': []}
        for (feat, sc), los in sorted(cells.items()):
            if feat not in feats:
                continue
            if a.extra_x1_only and nm.endswith(' [x1]') and sc != 1.0:
                continue
            meas = float(np.median(los))
            k = KM[feat]['measured_per_use']
            for mdl, kw in (('P1', dict(mode='P1')), ('P2', dict(mode='P2')),
                            ('P3', dict(mode='P2', k=k))):
                r, _ = P.breakpoint(feature=feat, scale=sc, **kw)
                errs[mdl].append(abs(100 * (r - meas) / meas))
        n = len(errs['P1'])
        for mdl in ('P1', 'P2', 'P3'):
            lo_, hi_ = boot_ci(errs[mdl])
            ci = f'[{lo_:.1f}, {hi_:.1f}]' if np.isfinite(lo_) else 'KHONG bao cao (n<=2)'
            print(f'  {nm:10s} {n:2d} {mdl:12s} {np.mean(errs[mdl]):11.1f}% {ci:>20s}')

    print('\n' + '=' * 96)
    print('  [3] THIEN LECH CUA CHI BAO NGHEN -- ho 3 kiem dinh, hieu chinh Holm')
    print('=' * 96)
    grp = {'quat-ra nhieu': ['quickadd_x2', 'quickadd_x1', 'promo_x2', 'express_x1']}
    vals = {'CPU utilization': ([.3190, .4640, .6301, .6699], [.8050, .8095, .8120, .8246, .8401, .8623, .8726, .8754, .8834, .8959, .9194, .9624]),
            'throttled-seconds': ([.1923, .2538, .1524, .3025], [.2126, .1211, .1767, .1976, .2037, .1498, .1913, .2066, .1774, .2428, .3394, .3236]),
            'throttled-periods': ([.0538, .1046, .1519, .1857], [.4111, .3857, .3731, .3971, .4482, .4593, .5476, .4638, .5650, .4729, .6667, .6866])}
    labs = list(vals)
    ps = [mannwhitneyu(*vals[k]).pvalue for k in labs]
    adj = holm(ps, labs)
    print(f"\n  {'chi bao':20s} {'trung vi quat-ra':>17s} {'binh thuong':>12s} {'p tho':>8s} {'p Holm':>8s} {'Cliff d':>9s}")
    for k in labs:
        a, b = vals[k]
        p0, pa, _ = adj[k]
        print(f'  {k:20s} {np.median(a):17.3f} {np.median(b):12.3f} {p0:8.4f} {pa:8.4f} '
              f'{cliffs_delta(a, b):+9.3f}')
    print('\n  Chi bao KHONG thien lech (p Holm lon) la chi bao dung duoc lam NGUONG DUY NHAT'
          '\n  cho moi loai tinh nang -- day la tieu chi chon, khong phai "ket qua am".')


if __name__ == '__main__':
    main()
