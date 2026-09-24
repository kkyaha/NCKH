# -*- coding: utf-8 -*-
"""
QUEUEING FORM CO XUNG DANG CHO LATENCY KHONG? (Alibaba v2021)
===============================================================
experiments/scm_mechanism_bakeoff.py cho thay QueueingLatencyRegressor la
nhanh fit TE NHAT o che do ngoai suy khi du bao CPU (skill 0.236 vs 0.357
cua NNLS, thang 27.2% so cap, p<1e-4). Nhung do la ket qua tren CPU, con
trong san pham co che nay duoc dung cho LATENCY. Script nay chay dung phep
kiem con thieu do.

Vi sao phai test rieng: dang phi tuyen phi(W)=W/(C-W) chi co noi dung khi he
tien gan bao hoa. Neu du lieu khong bao gio cham vung do thi phi gan nhu
tuyen tinh tren dai quan sat, cong tuyen voi W, va NNLS ep mot trong hai he
so ve 0 -- dung co che that bai da ghi trong README "Gioi han da biet" #7
(Train Ticket: 28/28 co che latency fit ra coef = 0).

Cap bien dung dung ly thuyet hang doi: workload PHIA PROVIDER (so loi goi
DEN service) -> response time PHIA PROVIDER (thoi gian service tu phuc vu).
  workload = sum(providerRPC_MCR, HTTP_MCR)
  latency  = sum(RT_i * MCR_i) / sum(MCR_i)   <- thoi gian dap ung TRUNG BINH
                                                 CO TRONG SO theo luu luong,
                                                 khong phai trung binh cong
KHONG dung consumerRPC_RT/consumerMQ_RT: do la do tre service QUAN SAT khi
GOI di, thuoc ve node ha nguon, khong phai do tre cua chinh no.

Ba che do danh gia:
  indist   : ngau nhien 70/30
  ood      : train tai THAP 67% -> test tai CAO 33%
  ood_tail : nhu ood nhung CHI cham diem tren 10% diem tai CAO NHAT -- neu
             dang queueing co gia tri o dau thi phai la o day.

Moc so sanh (reference) o day la Queueing, vi no la co che DANG TRIEN KHAI
cho latency -- cau hoi la co nen thay no khong.

Output: data/processed/scm_results/alibaba_latency_queueing_test.csv
Dung:   python experiments/alibaba_latency_queueing_test.py [--max-files 2] [--n-services 300]
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
from scipy.stats import wilcoxon

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))

from queueing_regressor import QueueingLatencyRegressor      # noqa: E402
import alibaba_signal_check as asc                            # noqa: E402

PROVIDER_PAIRS = [('providerRPC_MCR', 'providerRPC_RT'), ('HTTP_MCR', 'HTTP_RT')]
MIN_POINTS = 45
N_TRAIN = 30
TAIL_FRAC = 0.10
SEED = 42
RNG = np.random.RandomState(SEED)


def load_provider_workload_latency(max_files: int) -> pd.DataFrame:
    """Tra ve DataFrame [timestamp, msname, workload, latency] phia provider."""
    files = [f for f in asc._find('*.csv')
             if any(k in os.path.basename(f) for k in ('RTQps', 'Qps'))][:max_files]
    if not files:
        return pd.DataFrame()
    wanted = {m for pair in PROVIDER_PAIRS for m in pair}
    parts = []
    for f in files:
        for ch in pd.read_csv(f, usecols=asc.MCR_COLS, chunksize=asc.CHUNK,
                              dtype={'msname': 'category', 'metric': 'category',
                                     'value': 'float32', 'timestamp': 'int64'}):
            ch = ch[ch['metric'].isin(wanted)]
            if len(ch):
                parts.append(ch.groupby(['timestamp', 'msname', 'metric'],
                                        observed=True)['value'].mean().reset_index())
    if not parts:
        return pd.DataFrame()

    d = pd.concat(parts, ignore_index=True)
    d = (d.groupby(['timestamp', 'msname', 'metric'], observed=True)['value']
         .mean().reset_index())
    wide = d.pivot_table(index=['timestamp', 'msname'], columns='metric',
                         values='value', observed=True).reset_index()

    num = np.zeros(len(wide))      # tong (RT * MCR)
    den = np.zeros(len(wide))      # tong MCR
    for mcr, rt in PROVIDER_PAIRS:
        if mcr in wide.columns and rt in wide.columns:
            m = wide[mcr].fillna(0).to_numpy(float)
            r = wide[rt].fillna(0).to_numpy(float)
            ok = np.isfinite(m) & np.isfinite(r) & (m > 0)
            num[ok] += (r[ok] * m[ok])
            den[ok] += m[ok]
    wide['workload'] = den
    wide['latency'] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return wide[['timestamp', 'msname', 'workload', 'latency']].dropna()


def skill(y, pred, baseline_pred):
    mse_m = float(np.mean((y - pred) ** 2))
    mse_b = float(np.mean((y - baseline_pred) ** 2))
    return np.nan if mse_b <= 1e-15 else 1.0 - mse_m / mse_b


def _elasticity(Xtr, ytr):
    w = Xtr.ravel()
    m = LinearRegression(positive=True).fit(Xtr, ytr)
    b = float(m.coef_[0])
    w_bar, y_bar = float(w.mean()), float(ytr.mean())
    if w_bar == 0 or y_bar == 0:
        return np.nan, np.inf, w_bar, y_bar
    resid = ytr - m.predict(Xtr)
    sxx = float(((w - w.mean()) ** 2).sum())
    se_b = float(np.sqrt((resid ** 2).sum() / max(len(w) - 2, 1) / sxx)) if sxx > 0 else np.inf
    return b * w_bar / y_bar, se_b * w_bar / y_bar, w_bar, y_bar


def _predict_elasticity(w, gamma, w_bar, y_bar):
    return y_bar + gamma * (y_bar / w_bar) * (w - w_bar)


def _split(g, mode):
    if mode == 'indist':
        idx = RNG.permutation(len(g))
        k = int(len(g) * 0.7)
        return g.iloc[idx[:k]], g.iloc[idx[k:]]
    g = g.sort_values('workload')
    k = int(len(g) * 0.67)
    tr, te = g.iloc[:k], g.iloc[k:]
    if mode == 'ood_tail':                      # chi giu 10% tai cao nhat
        te = te.iloc[-max(int(len(g) * TAIL_FRAC), 10):]
    return tr, te


def _eval_one(g, svc, mode, shuffled, prior_pool):
    tr, te = _split(g, mode)
    if len(tr) < N_TRAIN or len(te) < 8:
        return []
    tr = tr.iloc[:N_TRAIN] if mode != 'indist' else tr.sample(N_TRAIN, random_state=SEED)

    Xtr, Xte = tr[['workload']].values, te[['workload']].values
    ytr, yte = tr['latency'].values, te['latency'].values
    if np.std(Xtr) < 1e-12 or np.std(ytr) < 1e-12:
        return []

    const_tr = np.full(len(yte), float(np.mean(ytr)))
    base = dict(service=svc, mode=mode, shuffled=shuffled, n_test=len(te))
    out = [dict(base, model='constant', skill=0.0),
           dict(base, model='persistence',
                skill=skill(yte, np.full(len(yte), float(ytr[-1])), const_tr))]

    models = {
        'Queueing':      QueueingLatencyRegressor(),     # DANG TRIEN KHAI cho latency
        'NNLS_deployed': LinearRegression(positive=True),
        'LinearReg':     LinearRegression(),
        'GradBoost':     GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                                   learning_rate=0.07, random_state=SEED),
    }
    for name, m in models.items():
        try:
            out.append(dict(base, model=name,
                            skill=skill(yte, m.fit(Xtr, ytr).predict(Xte), const_tr)))
        except Exception:
            continue

    own_gam, se_gam, w_bar, y_bar = _elasticity(Xtr, ytr)
    others = [v for k, v in prior_pool.items() if k != svc]
    if others and np.isfinite(own_gam) and w_bar and y_bar:
        gam_prior = float(np.median(others))
        out.append(dict(base, model='ElasticityPrior',
                        skill=skill(yte, _predict_elasticity(Xte.ravel(), gam_prior, w_bar, y_bar), const_tr)))
        tau2 = max(float(np.var(others)) - (float(se_gam ** 2) if np.isfinite(se_gam) else 0.0), 0.0)
        se2 = float(se_gam ** 2) if np.isfinite(se_gam) else np.inf
        lam = tau2 / (tau2 + se2) if np.isfinite(se2) and (tau2 + se2) > 0 else 0.0
        out.append(dict(base, model='ElasticityPooled',
                        skill=skill(yte, _predict_elasticity(
                            Xte.ravel(), lam * own_gam + (1 - lam) * gam_prior, w_bar, y_bar), const_tr)))
    return out


def build_prior_pool(df):
    out = {}
    for name, g in df.groupby('msname', observed=True):
        g = g[g.workload > 0]
        if len(g) < MIN_POINTS:
            continue
        X, y = g[['workload']].values, g['latency'].values
        if np.std(X) < 1e-12 or np.std(y) < 1e-12:
            continue
        gam, _, _, _ = _elasticity(X, y)
        if np.isfinite(gam):
            out[str(name)] = gam
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-files', type=int, default=2)
    ap.add_argument('--n-services', type=int, default=300)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== loading Alibaba (provider-side workload + response time) ===")
    df = load_provider_workload_latency(args.max_files)
    if df.empty:
        print("[LOI] Khong doc duoc du lieu -- kiem tra ALIBABA_DIR.")
        return
    df = df[(df.workload > 0) & (df.latency > 0)]
    print(f"  {len(df):,} dong | {df.msname.nunique():,} microservice")

    sig = {}
    for name, g in df.groupby('msname', observed=True):
        if len(g) < MIN_POINTS:
            continue
        w, y = g['workload'].values, g['latency'].values
        if np.std(w) < 1e-12 or np.std(y) < 1e-12:
            continue
        sig[str(name)] = float(np.corrcoef(w, y)[0, 1] ** 2)
    if not sig:
        print("[LOI] Khong service nao du diem.")
        return
    vals = np.array(list(sig.values()))
    print(f"  SIGNAL GATE workload->latency: R^2 trung vi={np.median(vals):.4f}, "
          f"%>0.1={np.mean(vals > 0.1):.1%}, %>0.3={np.mean(vals > 0.3):.1%}")
    pts = df.groupby('msname', observed=True).size()
    print(f"  diem/service: trung vi={int(pts.median())}, max={int(pts.max())}")

    eligible = list(sig)
    if len(eligible) > args.n_services:
        eligible = list(RNG.choice(eligible, args.n_services, replace=False))

    rows = []
    for shuffled in (False, True):
        work = df[df.msname.astype(str).isin(eligible)].copy()
        if shuffled:
            work['workload'] = (work.groupby('msname', observed=True)['workload']
                                .transform(lambda s: RNG.permutation(s.values)))
        prior_pool = build_prior_pool(work)
        for name, g in work.groupby('msname', observed=True):
            g = g[g.workload > 0]
            if len(g) < MIN_POINTS:
                continue
            for mode in ('indist', 'ood', 'ood_tail'):
                for r in _eval_one(g, str(name), mode, shuffled, prior_pool):
                    r['r2_signal'] = sig.get(str(name), np.nan)
                    rows.append(r)
        print(f"  [xong] shuffled={shuffled}: {len(rows):,} dong")

    d = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'alibaba_latency_queueing_test.csv')
    d.to_csv(out_path, index=False)
    report(d)
    print(f"\n[OK] -> {out_path}")


def report(d):
    if d.empty:
        print("[LOI] khong co ket qua.")
        return
    real = d[(~d.shuffled) & (d.r2_signal > 0.1)]
    print("\n" + "=" * 88)
    print("  LATENCY -- skill trung vi tren service CO TIN HIEU (R^2>0.1)")
    print("=" * 88)
    piv = real.pivot_table(index='model', columns='mode', values='skill', aggfunc='median')
    cnt = real.pivot_table(index='model', columns='mode', values='skill',
                            aggfunc=lambda s: (s > 0).mean())
    print(piv.round(4).to_string())
    print("\n  -- %service co skill > 0 --")
    print(cnt.round(3).to_string())

    print("\n" + "=" * 88)
    print("  KIEM DINH GHEP CAP vs Queueing (co che DANG TRIEN KHAI cho latency)")
    print("=" * 88)
    for mode in ('indist', 'ood', 'ood_tail'):
        sub = real[real['mode'] == mode]
        p = sub.pivot_table(index='service', columns='model', values='skill').dropna()
        if p.empty or 'Queueing' not in p.columns:
            continue
        print(f"\n  mode={mode}  (n={len(p)} cap; Queueing median={p['Queueing'].median():.4f})")
        rows = []
        for m in [c for c in p.columns if c not in ('Queueing', 'constant')]:
            diff = p[m] - p['Queueing']
            try:
                pv = wilcoxon(p[m], p['Queueing']).pvalue
            except Exception:
                pv = np.nan
            rows.append({'model': m, 'median_skill': p[m].median(),
                         'vs_Queueing': diff.median(), 'win_rate': (diff > 0).mean(),
                         'wilcoxon_p': pv})
        r = pd.DataFrame(rows).sort_values('median_skill', ascending=False)
        r['sig'] = np.where(r.wilcoxon_p < 0.05, '*', '')
        print(r.round(4).to_string(index=False))

    neg = d[d.shuffled & (d.r2_signal > 0.1)]
    if not neg.empty:
        print("\n" + "=" * 88)
        print("  NEGATIVE CONTROL -- PHAI ~0 hoac am")
        print("=" * 88)
        print(neg.pivot_table(index='model', columns='mode', values='skill',
                              aggfunc='median').round(4).to_string())


if __name__ == '__main__':
    main()
