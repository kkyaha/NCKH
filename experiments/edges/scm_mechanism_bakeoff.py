# -*- coding: utf-8 -*-
"""
BAKE-OFF CHON CO CHE DU PHONG CHO SCM (Alibaba v2021)
=======================================================
Muc dich: sinh CON SO de quyet dinh dung lop mo hinh nao lam co che Tier-2
(workload -> resource) trong Global DAG, thay vi chon theo cam tinh.

Vi sao chay tren Alibaba chu khong phai RCAEval: signal gate (RQ4, va kiem
lai trong experiments/elasticity_transfer_loso.py) cho thay CA BA he RCAEval
(SockShop 0.015, Train Ticket 0.043, Online Boutique 0.042 R^2 trung vi) deu
KHONG co tin hieu workload->CPU -- moi so sanh mo hinh tren do la vacuous
(negative control khong tach khoi mo hinh that). Alibaba la bo duy nhat co
tin hieu that (R^2 trung vi 0.22-0.30).

CAC NHANH SO SANH
-----------------
  constant          : trung binh tap train (san; skill == 0 theo dinh nghia)
  persistence       : gia tri quan sat cuoi cua tap train
  NNLS_deployed     : LinearRegression(positive=True)  <- DANG TRIEN KHAI (cpu/mem/workload)
  LinearReg         : tuyen tinh khong rang buoc -- do CAI GIA cua rang buoc khong am
  Queueing          : QueueingLatencyRegressor         <- DANG TRIEN KHAI (latency)
  GradBoost         : baseline ML phi tuyen
  ElasticityPrior   : CHUYEN GIAO thuan -- gamma prior (trung vi cua MOI service
                      KHAC) + trung binh nen cua chinh no. Khong fit gi tren
                      du lieu rieng. Day la kich ban "he moi, chua co lich su".
  ElasticityPooled  : co ngot empirical-Bayes: gamma = lam*gamma_own + (1-lam)*gamma_prior

TRUC DANH GIA
-------------
  metric    : cpu (chinh), mem (chung doi chieu -- RQ4 da ket luan khong co skill)
  mode      : indist (ngau nhien 70/30) / ood (train tai THAP -> test tai CAO)
  n_train   : 60 / 150 / 400 / 1200  <- TRUC MOI. Partial pooling chi co tac dung
              khi node THIEU du lieu; thi nghiem LOSO truoc do fit 43k dong/node
              nen lambda ~ 0.999 va pooling khong bao gio kich hoat. Truc nay
              kiem dung che do cold-start ma mot trien khai moi thuc su o trong.
  shuffled  : negative control (hoan vi workload trong tung service)
  Phan tang theo SIGNAL: bao cao rieng nhom service co R^2 > 0.1 -- so sanh mo
              hinh tren service khong co tin hieu la vo nghia (moi nhanh deu ~0).

Output: data/processed/scm_results/scm_mechanism_bakeoff.csv
Dung:   python experiments/scm_mechanism_bakeoff.py [--max-files 2] [--n-services 400]
"""

import os
import sys
import argparse
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))

from queueing_regressor import QueueingLatencyRegressor            # noqa: E402
import alibaba_signal_check as asc                                  # noqa: E402

MIN_POINTS = 45
# Alibaba lay mau 60 giay tren ~12 gio => TOI DA ~720 diem/service, va sau
# split OOD 67/33 chi con ~480 dong train. Cac muc duoi doc thang ra thoi
# gian telemetry ma mot trien khai MOI thuc su co: 30 phut / 1 gio / 2 gio /
# 4 gio -- dung che do cold-start can kiem, thay vi muc 1200 khong ton tai
# trong du lieu (lan chay dau bi bo 100% service vi ly do nay).
N_TRAIN_LEVELS = (30, 60, 120, 240)
SEED = 42
RNG = np.random.RandomState(SEED)


def skill(y, pred, baseline_pred):
    mse_m = float(np.mean((y - pred) ** 2))
    mse_b = float(np.mean((y - baseline_pred) ** 2))
    return np.nan if mse_b <= 1e-15 else 1.0 - mse_m / mse_b


def _split(g, mode):
    if mode == 'ood':
        g = g.sort_values('workload')
        k = int(len(g) * 0.67)
        return g.iloc[:k], g.iloc[k:]
    idx = RNG.permutation(len(g))
    k = int(len(g) * 0.7)
    return g.iloc[idx[:k]], g.iloc[idx[k:]]


def _fitted_models(Xtr, ytr):
    """Cac nhanh CO fit tren du lieu rieng cua service."""
    return {
        'NNLS_deployed': LinearRegression(positive=True),
        'LinearReg':     LinearRegression(),
        'Queueing':      QueueingLatencyRegressor(),
        'GradBoost':     GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                                   learning_rate=0.07, random_state=SEED),
    }


def _elasticity(Xtr, ytr):
    """gamma = b * W_bar / y_bar: do co gian DIEM cua mo hinh tuyen tinh tai
    trung binh -- khong thu nguyen nen pool duoc xuyen service. Tra ve
    (gamma, se_gamma, w_bar, y_bar)."""
    w = Xtr.ravel()
    m = LinearRegression(positive=True).fit(Xtr, ytr)
    b = float(m.coef_[0])
    w_bar, y_bar = float(w.mean()), float(ytr.mean())
    if w_bar == 0 or y_bar == 0:
        return np.nan, np.inf, w_bar, y_bar
    resid = ytr - m.predict(Xtr)
    dof = max(len(w) - 2, 1)
    sxx = float(((w - w.mean()) ** 2).sum())
    se_b = float(np.sqrt((resid ** 2).sum() / dof / sxx)) if sxx > 0 else np.inf
    return b * w_bar / y_bar, se_b * w_bar / y_bar, w_bar, y_bar


def _predict_elasticity(w, gamma, w_bar, y_bar):
    return y_bar + gamma * (y_bar / w_bar) * (w - w_bar)


def build_prior_pool(df, metric):
    """gamma cua MOI service tren toan bo du lieu cua no -- dung lam bon prior.
    Khi cham diem service v, prior cua v = trung vi cua TAT CA service KHAC
    (leave-one-service-out, khong ro ri)."""
    out = {}
    for name, g in df.groupby('msname', observed=True):
        g = g[g.workload > 0]
        if len(g) < MIN_POINTS:
            continue
        X, y = g[['workload']].values, g[metric].values
        if np.std(X) < 1e-12 or np.std(y) < 1e-12:
            continue
        gam, _, _, _ = _elasticity(X, y)
        if np.isfinite(gam):
            out[str(name)] = gam
    return out


def _eval_one(g, svc, metric, mode, n_train, shuffled, prior_pool, prior_sum, prior_cnt):
    tr, te = _split(g, mode)
    if len(tr) < n_train or len(te) < 15:
        return []
    tr = tr.iloc[:n_train] if mode == 'ood' else tr.sample(n_train, random_state=SEED)

    Xtr, Xte = tr[['workload']].values, te[['workload']].values
    ytr, yte = tr[metric].values, te[metric].values
    if np.std(Xtr) < 1e-12 or np.std(ytr) < 1e-12:
        return []

    const_tr = np.full(len(yte), float(np.mean(ytr)))
    base = dict(service=svc, metric=metric, mode=mode, n_train=n_train,
                shuffled=shuffled, n_test=len(te))
    out = [dict(base, model='constant', skill=0.0),
           dict(base, model='persistence',
                skill=skill(yte, np.full(len(yte), float(ytr[-1])), const_tr))]

    for name, m in _fitted_models(Xtr, ytr).items():
        try:
            p = m.fit(Xtr, ytr).predict(Xte)
            out.append(dict(base, model=name, skill=skill(yte, p, const_tr)))
        except Exception:
            continue

    # --- nhanh elasticity: prior leave-one-service-out ---
    own_gam, se_gam, w_bar, y_bar = _elasticity(Xtr, ytr)
    n_other = prior_cnt - (1 if svc in prior_pool else 0)
    if n_other > 0 and w_bar != 0 and y_bar != 0 and np.isfinite(own_gam):
        others = [v for k, v in prior_pool.items() if k != svc]
        gam_prior = float(np.median(others))
        out.append(dict(base, model='ElasticityPrior',
                        skill=skill(yte, _predict_elasticity(Xte.ravel(), gam_prior, w_bar, y_bar), const_tr)))
        tau2 = max(float(np.var(others)) - float(se_gam ** 2) if np.isfinite(se_gam) else 0.0, 0.0)
        se2 = float(se_gam ** 2) if np.isfinite(se_gam) else np.inf
        lam = tau2 / (tau2 + se2) if np.isfinite(se2) and (tau2 + se2) > 0 else 0.0
        gam_pool = lam * own_gam + (1 - lam) * gam_prior
        r = dict(base, model='ElasticityPooled',
                 skill=skill(yte, _predict_elasticity(Xte.ravel(), gam_pool, w_bar, y_bar), const_tr))
        r['lambda'] = lam
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-files', type=int, default=2,
                     help='So file CSV moi loai doc tu Alibaba (moi file ~2.8GB)')
    ap.add_argument('--n-services', type=int, default=400,
                     help='So service lay mau de cham diem (mediant on dinh tu ~200)')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Gioi han so file de chay duoc trong thoi gian hop ly (69GB tren dia).
    # Phai cat RIENG tung loai: _find('*.csv') tra ve ca MSCallGraph (khong
    # dung o day) va sort alphabet dat MSCallGraph len dau, nen cat phang
    # top-N se loai sach ca MSResource lan MSRTQps.
    orig_find = asc._find

    def _capped_find(pattern):
        hits = orig_find(pattern)
        if pattern != '*.csv':
            return hits
        res = [f for f in hits if 'Resource' in os.path.basename(f)][:args.max_files]
        qps = [f for f in hits if any(k in os.path.basename(f)
                                       for k in ('RTQps', 'Qps'))][:args.max_files]
        return sorted(res + qps)

    asc._find = _capped_find

    print("=== loading Alibaba ===")
    print(f"  file dung: {[os.path.basename(f) for f in _capped_find('*.csv')]}")
    rs, wl = asc.load_resource(), asc.load_mcr()
    if rs is None or wl is None:
        print(f"[LOI] load_resource={type(rs)}, load_mcr={type(wl)} -- kiem tra ALIBABA_DIR")
        return
    print(f"  resource: {len(rs):,} dong | workload: {len(wl):,} dong")
    df = wl.merge(rs, on=['timestamp', 'msname'], how='inner')
    print(f"  ghep: {len(df):,} dong | {df.msname.nunique():,} microservice")
    if df.empty:
        print("[LOI] Ghep ra rong -- cua so thoi gian cua 2 loai file co the khong trung nhau; "
              "tang --max-files.")
        return

    # --- signal gate per service (phan tang ket qua theo tin hieu) ---
    sig = {}
    for name, g in df.groupby('msname', observed=True):
        g = g[g.workload > 0]
        if len(g) < MIN_POINTS:
            continue
        w, y = g['workload'].values, g['cpu'].values
        if np.std(w) < 1e-12 or np.std(y) < 1e-12:
            continue
        sig[str(name)] = float(np.corrcoef(w, y)[0, 1] ** 2)
    print(f"  signal gate: R^2 trung vi = {np.median(list(sig.values())):.4f}, "
          f"%R^2>0.1 = {np.mean([v > 0.1 for v in sig.values()]):.1%}, "
          f"%R^2>0.3 = {np.mean([v > 0.3 for v in sig.values()]):.1%}")

    pts = df.groupby('msname', observed=True).size()
    print(f"  diem/service: trung vi={int(pts.median())}, p90={int(pts.quantile(0.9))}, "
          f"max={int(pts.max())} -> muc n_train kha dung: "
          f"{[n for n in N_TRAIN_LEVELS if (pts.max() * 0.67) >= n]}")

    eligible = [s for s in sig if s in set(df.msname.astype(str))]
    if len(eligible) > args.n_services:
        eligible = list(RNG.choice(eligible, args.n_services, replace=False))
    print(f"  cham diem tren {len(eligible)} service\n")

    rows = []
    for shuffled in (False, True):
        work = df[df.msname.astype(str).isin(eligible)].copy()
        if shuffled:
            work['workload'] = (work.groupby('msname', observed=True)['workload']
                                .transform(lambda s: RNG.permutation(s.values)))
        for metric in ('cpu', 'mem'):
            prior_pool = build_prior_pool(work, metric)
            prior_cnt = len(prior_pool)
            for name, g in work.groupby('msname', observed=True):
                svc = str(name)
                g = g[g.workload > 0]
                if len(g) < MIN_POINTS:
                    continue
                for mode in ('indist', 'ood'):
                    for n_train in N_TRAIN_LEVELS:
                        for r in _eval_one(g, svc, metric, mode, n_train, shuffled,
                                            prior_pool, None, prior_cnt):
                            r['r2_signal'] = sig.get(svc, np.nan)
                            rows.append(r)
        print(f"  [xong] shuffled={shuffled}: tich luy {len(rows):,} dong ket qua")

    d = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'scm_mechanism_bakeoff.csv')
    d.to_csv(out_path, index=False)
    report(d)
    print(f"\n[OK] -> {out_path}")


def report(d):
    if d.empty or 'shuffled' not in d.columns:
        print("\n[LOI] Khong cham diem duoc service nao -- thuong do moi service co qua it "
              "diem thoi gian so voi N_TRAIN_LEVELS. Xem dong 'diem/service' o tren va "
              "tang --max-files hoac ha muc n_train.")
        return
    real = d[~d.shuffled]
    sig_hi = real[real.r2_signal > 0.1]

    print("\n" + "=" * 92)
    print("  BANG QUYET DINH -- skill trung vi tren service CO TIN HIEU (R^2>0.1), metric=cpu")
    print("=" * 92)
    for mode in ('indist', 'ood'):
        sub = sig_hi[(sig_hi['mode'] == mode) & (sig_hi.metric == 'cpu')]
        if sub.empty:
            continue
        print(f"\n  mode = {mode}")
        piv = sub.pivot_table(index='model', columns='n_train', values='skill', aggfunc='median')
        print(piv.round(4).to_string())
        print("  -- %service co skill > 0 --")
        pivp = sub.pivot_table(index='model', columns='n_train', values='skill',
                                aggfunc=lambda s: (s > 0).mean())
        print(pivp.round(3).to_string())

    print("\n" + "=" * 92)
    print("  NEGATIVE CONTROL (workload da hoan vi) -- PHAI ~0 hoac am")
    print("=" * 92)
    neg = d[d.shuffled & (d.metric == 'cpu') & (d.r2_signal > 0.1)]
    if not neg.empty:
        print(neg.pivot_table(index='model', columns='n_train', values='skill',
                              aggfunc='median').round(4).to_string())

    print("\n" + "=" * 92)
    print("  DOI CHIEU: metric=mem (RQ4 da ket luan khong co skill) -- ky vong ~0")
    print("=" * 92)
    mem = real[(real.metric == 'mem') & (real.r2_signal > 0.1)]
    if not mem.empty:
        print(mem.pivot_table(index='model', columns='mode', values='skill',
                              aggfunc='median').round(4).to_string())

    lam = real[real.model == 'ElasticityPooled']
    if 'lambda' in lam.columns and lam['lambda'].notna().any():
        print("\n  lambda (trong so giu uoc luong RIENG) trung vi theo n_train:")
        print(lam.groupby('n_train')['lambda'].median().round(4).to_string())


if __name__ == '__main__':
    main()
