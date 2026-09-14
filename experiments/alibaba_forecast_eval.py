# -*- coding: utf-8 -*-
"""
Danh gia du bao capacity tren Alibaba v2021 -- CO CHE DO THAT BAI
==================================================================
Vi sao khong tai dung giao thuc cu
----------------------------------
Giao thuc cu (RQ1/RQ2 tren RCAEval) KHONG THE sai. Memory dat MAPE 2-4% va
trong nhu thanh cong, trong khi R^2 = 0.003 chung minh khong co quan he nao:
memory gan nhu khong doi, nen MOT HANG SO cung dat 2-4%. Do do MAPE thap khong
phan biet duoc "mo hinh hoc duoc quan he" voi "mo hinh doan gia tri trung
binh". Chay lai dung giao thuc do tren du lieu tot hon chi lam sai lam cu co
con so dep hon.

Sau chot an toan, moi chot deu CO THE cho ket qua am
----------------------------------------------------
  0. Signal gate      : R^2 tren TOAN BO du lieu (khong chia). R^2 ~ 0 -> dung,
                        moi tuyen bo accuracy vo nghia.
  1. Negative control : hoan vi cot workload roi chay lai TOAN BO. Neu mo hinh
                        van "du bao tot" thi metric dang do thu khac.
  2. Baseline hang so : du doan trung binh tap train, bo qua workload. Moi mo
                        hinh PHAI vuot no. (Chot ma ban goc thieu.)
     + persistence    : du doan gia tri quan sat cuoi cua tap train.
  3. Mo hinh          : cung mau train, cung split.
  4. Skill score      : 1 - MSE_model/MSE_constant. Am = te hon hang so.
                        "Khong co tin hieu" TU DONG hien ra, khac han MAPE.
  5. In-dist VS OOD   : bao cao tach rieng. Neu khong hoc duoc quan he TRONG
                        phan phoi thi hoi ngoai suy la vo nghia.
  6. Phan phoi        : bao cao per-service, khong chi trung vi -- 1290 service
                        co R^2 tu 0 den 0.999, trung vi che mat viec chi mot tap
                        con du bao duoc.

Output: data/processed/scm_results/alibaba_forecast_eval.csv
"""

import os
import sys
import glob
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
sys.path.insert(0, os.path.dirname(__file__))
from alibaba_signal_check import load_resource, load_mcr   # noqa: E402

MIN_POINTS = 60
N_FIT = 2000          # tran mau train dung chung cho MOI mo hinh
SEED = 42
RNG = np.random.RandomState(SEED)


def skill(y, pred, baseline_pred):
    """1 - MSE_model/MSE_baseline. >0 tot hon baseline, <=0 la vo dung."""
    mse_m = float(np.mean((y - pred) ** 2))
    mse_b = float(np.mean((y - baseline_pred) ** 2))
    return np.nan if mse_b <= 1e-15 else 1.0 - mse_m / mse_b


def mape(y, pred):
    m = (y != 0) & np.isfinite(y) & np.isfinite(pred)
    return float(np.mean(np.abs((y[m] - pred[m]) / y[m])) * 100) if m.sum() else np.nan


def _models():
    return {
        'LinearReg': LinearRegression(),
        'NNLS_deployed': LinearRegression(positive=True),   # co che trien khai
        'GradBoost': GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                               learning_rate=0.07,
                                               random_state=SEED),
    }


def _split(g, mode):
    """mode='indist': ngau nhien 70/30. mode='ood': train tai THAP -> test tai CAO."""
    if mode == 'ood':
        g = g.sort_values('workload')
        k = int(len(g) * 0.67)
        return g.iloc[:k], g.iloc[k:]
    idx = RNG.permutation(len(g))
    k = int(len(g) * 0.7)
    return g.iloc[idx[:k]], g.iloc[idx[k:]]


def _eval_one(g, metric, mode, shuffled):
    tr, te = _split(g, mode)
    if len(tr) < 30 or len(te) < 15:
        return []
    if len(tr) > N_FIT:
        tr = tr.sample(N_FIT, random_state=SEED)

    Xtr = tr[['workload']].values
    Xte = te[['workload']].values
    ytr, yte = tr[metric].values, te[metric].values
    if np.std(Xtr) < 1e-12 or np.std(ytr) < 1e-12:
        return []

    # --- Chot 2: baseline bo qua workload hoan toan ---
    const_tr = np.full(len(yte), float(np.mean(ytr)))
    persist = np.full(len(yte), float(ytr[-1]))

    out = []
    base = dict(mode=mode, metric=metric, shuffled=shuffled,
                n_train=len(tr), n_test=len(te))
    out.append(dict(base, model='constant', mape=mape(yte, const_tr),
                    skill=0.0))
    out.append(dict(base, model='persistence', mape=mape(yte, persist),
                    skill=skill(yte, persist, const_tr)))
    for name, m in _models().items():
        try:
            p = m.fit(Xtr, ytr).predict(Xte)
        except Exception:
            continue
        out.append(dict(base, model=name, mape=mape(yte, p),
                        skill=skill(yte, p, const_tr)))
    return out


def run():
    print("=== loading ===")
    rs, wl = load_resource(), load_mcr()
    df = wl.merge(rs, on=['timestamp', 'msname'], how='inner')
    print(f"  ghep: {len(df):,} dong | {df.msname.nunique():,} microservice")

    rows = []
    for shuffled in (False, True):          # Chot 1: negative control
        work = df.copy()
        if shuffled:
            # hoan vi workload TRONG TUNG service -> pha quan he, giu phan phoi
            work['workload'] = (work.groupby('msname', observed=True)['workload']
                                .transform(lambda s: RNG.permutation(s.values)))
        for name, g in work.groupby('msname', observed=True):
            g = g[g.workload > 0]
            if len(g) < MIN_POINTS:
                continue
            for metric in ('cpu', 'mem'):
                for mode in ('indist', 'ood'):
                    for r in _eval_one(g, metric, mode, shuffled):
                        r['msname'] = str(name)
                        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT_DIR, 'alibaba_forecast_eval.csv'), index=False)
    return d


def report(d):
    if d.empty:
        print("khong co du lieu")
        return
    d = d.replace([np.inf, -np.inf], np.nan)

    for metric in ('cpu', 'mem'):
        sub = d[d.metric == metric]
        if sub.empty:
            continue
        print(f"\n{'='*78}")
        print(f"  {metric.upper()}  -- skill score (>0 = tot hon baseline hang so)")
        print(f"{'='*78}")
        print(f"  {'model':16s} {'mode':7s} {'shuf':5s} {'n':>5s} "
              f"{'skill_med':>10s} {'%skill>0':>9s} {'MAPE_med':>9s}")
        for mode in ('indist', 'ood'):
            for shuf in (False, True):
                s2 = sub[(sub['mode'] == mode) & (sub.shuffled == shuf)]
                for model in ('constant', 'persistence', 'LinearReg',
                              'NNLS_deployed', 'GradBoost'):
                    s3 = s2[s2.model == model]
                    if s3.empty:
                        continue
                    print(f"  {model:16s} {mode:7s} {str(shuf):5s} {len(s3):5d} "
                          f"{s3.skill.median():10.4f} "
                          f"{100*(s3.skill > 0).mean():8.1f}% "
                          f"{s3.mape.median():9.2f}")

    print(f"\n{'='*78}")
    print("  CACH DOC")
    print(f"{'='*78}")
    print("  * shuf=True (workload da hoan vi) PHAI cho skill ~ 0 hoac am.")
    print("    Neu no van duong dang ke -> metric khong do quan he voi workload.")
    print("  * Mo hinh chi co gia tri khi skill > 0 tren shuf=False va ~0 tren True.")
    print("  * MAPE duoc in de doi chieu: chu y truong hop MAPE thap NHUNG skill ~ 0")
    print("    -- do chinh la artefact da lam RCAEval trong nhu co tin hieu.")


if __name__ == '__main__':
    report(run())
