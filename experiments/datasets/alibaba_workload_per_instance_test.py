# -*- coding: utf-8 -*-
"""
BIEN HIEP BIEN NAO CHO LATENCY: WORKLOAD TONG, WORKLOAD/INSTANCE, HAY SO INSTANCE?
====================================================================================
experiments/alibaba_latency_queueing_test.py do duoc: ~50% service san xuat that
co latency GIAM khi workload tang (CPU chi 9.2%), khien moi co che rang buoc
`positive=True` -- gom ca QueueingLatencyRegressor dang trien khai -- bi ep he so
ve 0 va du bao HANG SO (skill trung vi dung bang 0.0000).

Gia thuyet giai thich: AUTOSCALING. Tai tang -> them replica -> tai TREN MOI
INSTANCE khong tang (co khi giam) -> latency giam. Neu dung, bien hiep bien
dung phai la workload/instance chu khong phai workload tho, va viec sua nay
TONG QUAT (moi he co replica count deu ap dung duoc).

LUU Y PHUONG PHAP (ly do script nay ton tai thay vi chi sua 1 dong):
hai thi nghiem truoc dung HAI dinh nghia workload KHAC NHAU ma khong noi ro --
  * alibaba_signal_check.load_mcr()  -> .sum()   qua cac instance (workload TONG)
  * alibaba_latency_queueing_test    -> .mean()  qua cac instance (~per-instance)
Nen khong the ket luan gi neu khong do CA BA bien tren CUNG mot tap service:
  raw_total  = tong MCR qua moi instance      (thong luong toan service)
  per_inst   = raw_total / so instance        (tai moi replica -- ung vien theo gia thuyet)
  n_inst     = so instance                    (de kiem autoscaling co that khong)

Bao cao: (1) dau he so doc cua tung bien -> latency, (2) tuong quan
workload<->n_inst (autoscaling co xay ra khong), (3) skill cua cac co che khi
doi bien hiep bien -- Queueing/NNLS co thoat trang thai suy bien khong.

Output: data/processed/scm_results/alibaba_workload_per_instance_test.csv
Dung:   python experiments/alibaba_workload_per_instance_test.py [--max-files 2]
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
from scipy.stats import wilcoxon

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))

from queueing_regressor import QueueingLatencyRegressor       # noqa: E402
import alibaba_signal_check as asc                             # noqa: E402

PROVIDER_PAIRS = [('providerRPC_MCR', 'providerRPC_RT'), ('HTTP_MCR', 'HTTP_RT')]
MIN_POINTS = 45
N_TRAIN = 30
SEED = 42
RNG = np.random.RandomState(SEED)
PREDICTORS = ('raw_total', 'per_inst', 'n_inst')


def load_with_instance_count(max_files: int) -> pd.DataFrame:
    """[timestamp, msname, raw_total, per_inst, n_inst, latency] phia provider.

    raw_total : TONG MCR qua moi instance (thong luong toan service)
    n_inst    : so msinstanceid phan biet trong (timestamp, msname)
    per_inst  : raw_total / n_inst
    latency   : sum(RT_i * MCR_i)/sum(MCR_i) -- trung binh CO TRONG SO theo luu luong
    """
    files = [f for f in asc._find('*.csv')
             if any(k in os.path.basename(f) for k in ('RTQps', 'Qps'))][:max_files]
    if not files:
        return pd.DataFrame()
    wanted = {m for pair in PROVIDER_PAIRS for m in pair}
    agg_parts, inst_parts = [], []
    cols = ['timestamp', 'msname', 'msinstanceid', 'metric', 'value']
    for f in files:
        for ch in pd.read_csv(f, usecols=cols, chunksize=asc.CHUNK,
                              dtype={'msname': 'category', 'metric': 'category',
                                     'value': 'float32', 'timestamp': 'int64'}):
            ch = ch[ch['metric'].isin(wanted)]
            if not len(ch):
                continue
            # tong gia tri theo metric (cong qua instance)
            agg_parts.append(ch.groupby(['timestamp', 'msname', 'metric'],
                                        observed=True)['value'].sum().reset_index())
            # dem instance phan biet
            inst_parts.append(ch.groupby(['timestamp', 'msname'],
                                          observed=True)['msinstanceid'].nunique().reset_index())
    if not agg_parts:
        return pd.DataFrame()

    d = (pd.concat(agg_parts, ignore_index=True)
         .groupby(['timestamp', 'msname', 'metric'], observed=True)['value'].sum().reset_index())
    inst = (pd.concat(inst_parts, ignore_index=True)
            .groupby(['timestamp', 'msname'], observed=True)['msinstanceid'].max().reset_index()
            .rename(columns={'msinstanceid': 'n_inst'}))

    wide = d.pivot_table(index=['timestamp', 'msname'], columns='metric',
                         values='value', observed=True).reset_index()
    num = np.zeros(len(wide))
    den = np.zeros(len(wide))
    for mcr, rt in PROVIDER_PAIRS:
        if mcr in wide.columns and rt in wide.columns:
            m = wide[mcr].fillna(0).to_numpy(float)
            r = wide[rt].fillna(0).to_numpy(float)
            ok = np.isfinite(m) & np.isfinite(r) & (m > 0)
            num[ok] += r[ok] * m[ok]
            den[ok] += m[ok]
    wide['raw_total'] = den
    wide['latency'] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    out = wide[['timestamp', 'msname', 'raw_total', 'latency']].merge(
        inst, on=['timestamp', 'msname'], how='inner').dropna()
    out = out[(out.raw_total > 0) & (out.latency > 0) & (out.n_inst > 0)]
    out['per_inst'] = out['raw_total'] / out['n_inst']
    return out


def skill(y, pred, base):
    mse_m = float(np.mean((y - pred) ** 2))
    mse_b = float(np.mean((y - base) ** 2))
    return np.nan if mse_b <= 1e-15 else 1.0 - mse_m / mse_b


def slope_sign_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby('msname', observed=True):
        if len(g) < MIN_POINTS:
            continue
        y = g['latency'].to_numpy(float)
        if np.std(y) < 1e-12:
            continue
        rec = {'service': str(name), 'n': len(g)}
        ok = True
        for p in PREDICTORS:
            x = g[p].to_numpy(float)
            if np.std(x) < 1e-12:
                ok = False
                break
            rec[f'slope_{p}'] = float(LinearRegression().fit(x.reshape(-1, 1), y).coef_[0])
            rec[f'r2_{p}'] = float(np.corrcoef(x, y)[0, 1] ** 2)
        if not ok:
            continue
        w, ni = g['raw_total'].to_numpy(float), g['n_inst'].to_numpy(float)
        rec['corr_workload_ninst'] = (float(np.corrcoef(w, ni)[0, 1])
                                       if np.std(ni) > 1e-12 else np.nan)
        rec['n_inst_median'] = float(np.median(ni))
        rows.append(rec)
    return pd.DataFrame(rows)


def eval_models(df: pd.DataFrame, signal_svcs: set) -> pd.DataFrame:
    rows = []
    for shuffled in (False, True):
        work = df.copy()
        if shuffled:
            for p in PREDICTORS:
                work[p] = (work.groupby('msname', observed=True)[p]
                           .transform(lambda s: RNG.permutation(s.values)))
        for name, g in work.groupby('msname', observed=True):
            svc = str(name)
            if svc not in signal_svcs or len(g) < MIN_POINTS:
                continue
            for pred_col in PREDICTORS:
                gg = g.sort_values(pred_col)
                k = int(len(gg) * 0.67)
                tr, te = gg.iloc[:k], gg.iloc[k:]
                if len(tr) < N_TRAIN or len(te) < 8:
                    continue
                tr = tr.iloc[:N_TRAIN]
                Xtr, Xte = tr[[pred_col]].values, te[[pred_col]].values
                ytr, yte = tr['latency'].values, te['latency'].values
                if np.std(Xtr) < 1e-12 or np.std(ytr) < 1e-12:
                    continue
                const = np.full(len(yte), float(np.mean(ytr)))
                base = dict(service=svc, predictor=pred_col, shuffled=shuffled)
                rows.append(dict(base, model='constant', skill=0.0))
                for mname, m in (('Queueing', QueueingLatencyRegressor()),
                                  ('NNLS_deployed', LinearRegression(positive=True)),
                                  ('LinearReg', LinearRegression())):
                    try:
                        rows.append(dict(base, model=mname,
                                         skill=skill(yte, m.fit(Xtr, ytr).predict(Xte), const)))
                    except Exception:
                        continue
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-files', type=int, default=2)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== loading Alibaba (provider-side + instance count) ===")
    df = load_with_instance_count(args.max_files)
    if df.empty:
        print("[LOI] khong doc duoc du lieu -- kiem tra ALIBABA_DIR.")
        return
    print(f"  {len(df):,} dong | {df.msname.nunique():,} microservice")
    print(f"  n_inst: trung vi={df.n_inst.median():.0f}, p90={df.n_inst.quantile(.9):.0f}, "
          f"max={df.n_inst.max():.0f}")

    st = slope_sign_table(df)
    if st.empty:
        print("[LOI] khong service nao du dieu kien.")
        return

    print("\n" + "=" * 86)
    print("  (1) DAU HE SO DOC -> LATENCY, theo tung bien hiep bien")
    print("=" * 86)
    print(f"{'bien':<12}{'%doc AM':>10}{'%doc AM (R2>0.1)':>20}{'R2 trung vi':>14}{'n(R2>0.1)':>12}")
    for p in PREDICTORS:
        hi = st[st[f'r2_{p}'] > 0.1]
        print(f"{p:<12}{(st[f'slope_{p}'] < 0).mean():>9.1%}"
              f"{((hi[f'slope_{p}'] < 0).mean() if len(hi) else np.nan):>19.1%}"
              f"{st[f'r2_{p}'].median():>14.4f}{len(hi):>12}")

    print("\n" + "=" * 86)
    print("  (2) AUTOSCALING CO XAY RA KHONG? tuong quan workload <-> so instance")
    print("=" * 86)
    cw = st['corr_workload_ninst'].dropna()
    print(f"  corr(raw_total, n_inst): trung vi={cw.median():.4f}, "
          f"%duong={np.mean(cw > 0):.1%}, %>0.5={np.mean(cw > 0.5):.1%}")
    print(f"  so service co n_inst THAY DOI theo thoi gian: "
          f"{(st['n_inst_median'] > 0).sum()} / {len(st)} "
          f"(n_inst trung vi toan cuc={st['n_inst_median'].median():.0f})")

    signal_svcs = set(st[st['r2_raw_total'] > 0.1]['service'])
    ev = eval_models(df, signal_svcs)
    st.to_csv(os.path.join(OUT_DIR, 'alibaba_workload_per_instance_test.csv'), index=False)

    if not ev.empty:
        real = ev[~ev.shuffled]
        print("\n" + "=" * 86)
        print("  (3) SKILL THEO BIEN HIEP BIEN (OOD, service co tin hieu tren raw_total)")
        print("=" * 86)
        print(real.pivot_table(index='model', columns='predictor',
                               values='skill', aggfunc='median').round(4).to_string())
        print("\n  -- %service co skill > 0 --")
        print(real.pivot_table(index='model', columns='predictor', values='skill',
                               aggfunc=lambda s: (s > 0).mean()).round(3).to_string())

        print("\n  KIEM DINH: per_inst co cuu duoc Queueing khong? (ghep cap tren service)")
        for mname in ('Queueing', 'NNLS_deployed'):
            p = real[real.model == mname].pivot_table(index='service', columns='predictor',
                                                       values='skill').dropna()
            if p.empty or not {'raw_total', 'per_inst'} <= set(p.columns):
                continue
            diff = p['per_inst'] - p['raw_total']
            try:
                pv = wilcoxon(p['per_inst'], p['raw_total']).pvalue
            except Exception:
                pv = np.nan
            print(f"    {mname:<14} n={len(p):>4}  median(raw)={p['raw_total'].median():.4f}  "
                  f"median(per_inst)={p['per_inst'].median():.4f}  "
                  f"delta={diff.median():+.4f}  thang={np.mean(diff > 0):.1%}  p={pv:.4g}")

        neg = ev[ev.shuffled]
        if not neg.empty:
            print("\n  NEGATIVE CONTROL (phai ~0 hoac am):")
            print(neg.pivot_table(index='model', columns='predictor',
                                  values='skill', aggfunc='median').round(4).to_string())
        ev.to_csv(os.path.join(OUT_DIR, 'alibaba_per_instance_model_eval.csv'), index=False)

    print(f"\n[OK] -> {OUT_DIR}/alibaba_workload_per_instance_test.csv")


if __name__ == '__main__':
    main()
