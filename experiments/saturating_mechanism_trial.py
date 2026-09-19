# -*- coding: utf-8 -*-
"""
THU NGHIEM TIEP THEO (sau nonlinear_mechanism_trial.py): mo hinh BAO HOA CO THAM SO
====================================================================================
Boi canh: nonlinear_mechanism_trial.py da chi ra hgbr_mono (HistGradientBoosting,
monotonic_cst=[1]) KHONG thang linear_pos trong protocol OOD thuc te (train LOW
67% -> test HIGH 33%): MAPE trung binh 13.67% vs 12.90% (quantile summary), va
linear_pos thang nhieu cap (service,metric) nhat (10/21 vs 7/21). Ly do ky thuat:
tree-based model (ke ca ban co rang buoc don dieu) KHONG ngoai suy duoc vuot pham
vi gia tri Workload da thay trong train -- moi leaf ngoai vung train chi tra ve
hang so (gia tri leaf bien), nen duong du bao "phang" ra thay vi tiep tuc tang,
danh gia THAP nguy co o vung tai cao -- dung cai ma he thong nay can du bao.

Cau hoi moi: CPU/Memory hien dung LinearRegression(positive=True) -- ngoai suy
TUYEN TINH VO HAN, trong khi CPU thuc te bi CHAN TREN (gioi han loi/core). Dung
mot duong cong BAO HOA CO THAM SO (sigmoid 3-tham-so, don dieu tang, tiem can
mot tran L duoc uoc luong tu du lieu) co cai thien MAPE/F1 vung tai cao so voi
linear_pos khong, ma van giu duoc tinh don dieu (dieu kien cho Proposition 2)?

Day la cung mot y tuong da dung cho Latency (QueueingLatencyRegressor: W/(C-W),
tiem can VO CUNG tai capacity) nhung ap dung nguoc lai cho CPU/Memory (tiem can
MOT TRAN HUU HAN, khong phai vo cung) -- ca hai deu la "duong cong CO THAM SO
dung dang" thay vi mot bo uoc luong hoc may tong quat (tree/kernel) khong biet
truoc dang ham.

Phuong phap: GIU NGUYEN protocol quantile (train LOW 67% -> test HIGH 33%) va
random (doi chung in-distribution) tu nonlinear_mechanism_trial.py, so 3 co che:
  A) linear_pos  : LinearRegression(positive=True) -- DANG DUNG production.
  B) hgbr_mono   : HistGradientBoostingRegressor(monotonic_cst=[1]) -- da thua o
                   ban truoc, giu lai de doi chung tay ba.
  C) sigmoid_sat : y = L/(1+exp(-k*(x-x0))) + b, don dieu tang, L uoc luong tu
                   max(y_train)*[1.05, 6.0] (khong gia dinh tran co dinh = 100,
                   vi CPU o day khong chuan hoa ve % 0-100 -- xem docstring
                   SigmoidSaturationRegressor). Fallback ve linear_pos neu
                   curve_fit khong hoi tu (thuong xay ra khi du lieu qua nhieu/
                   it bien thien, vd Memory gan hang so).

Output: data/processed/scm_results/saturating_mechanism_trial_<split_mode>.csv
        + <..>_summary.csv
"""

import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from scipy.optimize import curve_fit

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data, SERVICES, METRICS

N_PROJ = 500


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def _sigmoid(x, L, k, x0, b):
    return L / (1.0 + np.exp(-k * (x - x0))) + b


class SigmoidSaturationRegressor(BaseEstimator, RegressorMixin):
    """Duong cong bao hoa 3-tham-so: y = L/(1+exp(-k(x-x0))) + b.

    L (tran tiem can) duoc UOC LUONG, khong co dinh = 100: du lieu CPU o day
    (xem data_processor.METRICS) khong chuan hoa ve thang % 0-100 dong deu qua
    cac service (vd catalogue_cpu max quan sat ~49 nhung p99 chi ~0.23 -- phan
    phoi lech nang, xem docs/UNIFIED_REFERENCE_DOC.md muc "Gioi han da biet" #4
    ve catalogue_cpu). Vi vay tran duoc rang buoc trong [1.05, 6.0] x max(y_train)
    -- du rong de duong cong khong bi ep bao hoa som hon ca du lieu train da
    thay, nhung van huu han (khac linear_pos ngoai suy vo han).

    fit() thu 3 diem khoi tao k khac nhau (0.01, 0.1, 1.0 x 1/std(X)) vi
    curve_fit voi sigmoid rat nhay khoi tao; neu ca 3 deu khong hoi tu, hoac
    residual te hon linear don thuan tren TAP TRAIN, fallback ve
    LinearRegression(positive=True) (dat cung ten thuoc tinh coef_ gia lap de
    tuong thich voi _check_monotone_precondition cua CapacityAgent, xem note
    duoi __init__)."""

    def __init__(self):
        self.is_sigmoid_ = False
        self.params_ = None
        self.fallback_ = None

    def fit(self, X, y):
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel()
        y_max = max(np.max(y), 1e-9)
        x_std = max(np.std(X), 1e-9)

        best = None
        for k0 in (0.01 / x_std, 0.1 / x_std, 1.0 / x_std):
            try:
                p0 = [y_max * 1.5, k0, np.median(X), np.min(y)]
                bounds = (
                    [y_max * 1.05, 1e-6, X.min() - 3 * x_std, 0.0],
                    [y_max * 6.0, 10.0 / x_std, X.max() + 3 * x_std, y_max],
                )
                popt, _ = curve_fit(_sigmoid, X, y, p0=p0, bounds=bounds, maxfev=5000)
                pred = _sigmoid(X, *popt)
                sse = np.sum((pred - y) ** 2)
                if best is None or sse < best[1]:
                    best = (popt, sse)
            except Exception:
                continue

        lin = LinearRegression(positive=True).fit(X.reshape(-1, 1), y)
        lin_sse = np.sum((lin.predict(X.reshape(-1, 1)) - y) ** 2)

        if best is not None and best[1] < lin_sse:
            self.is_sigmoid_ = True
            self.params_ = best[0]
        else:
            self.is_sigmoid_ = False
            self.fallback_ = lin
        # coef_ ao, chi de tuong thich voi cac check doc `.coef_` ben ngoai
        # (vd _check_monotone_precondition trong capacity_agent.py) -- duong
        # cong nay LUON don dieu tang theo thiet ke (L>0, k>0), nen coef_ao=1.0
        self.coef_ = np.array([1.0])
        return self

    def predict(self, X):
        X = np.asarray(X).ravel()
        if self.is_sigmoid_:
            return _sigmoid(X, *self.params_)
        return self.fallback_.predict(X.reshape(-1, 1))


MECHANISMS = {
    'linear_pos': lambda: LinearRegression(positive=True),
    'hgbr_mono': lambda: HistGradientBoostingRegressor(
        monotonic_cst=[1], max_depth=4, max_iter=150, random_state=42
    ),
    'sigmoid_sat': lambda: SigmoidSaturationRegressor(),
}


def fit_and_eval(mech_name, df_train, df_test):
    g = nx.DiGraph()
    g.add_edge('Workload', 'Target')
    model = gcm.InvertibleStructuralCausalModel(g)

    estimator = MECHANISMS[mech_name]()
    model.set_causal_mechanism(
        'Target', AdditiveNoiseModel(SklearnRegressionModel(estimator))
    )
    gcm.auto.assign_causal_mechanisms(model, df_train)
    gcm.fit(model, df_train)

    df_test = df_test.copy()
    df_test['bkt'] = pd.qcut(
        df_test['Workload'], q=min(8, df_test['Workload'].nunique()), duplicates='drop'
    )
    bkts = df_test.groupby('bkt', observed=True)[['Workload', 'Target']].mean()

    y_true, y_pred = [], []
    for _, row in bkts.iterrows():
        wl_val = row['Workload']
        dp = gcm.interventional_samples(
            model, interventions={'Workload': lambda x, w=wl_val: w},
            num_samples_to_draw=N_PROJ
        )
        y_true.append(row['Target'])
        y_pred.append(dp['Target'].mean())

    yt, yp = np.array(y_true), np.array(y_pred)
    thresh = df_train['Target'].mean() + 0.5 * df_train['Target'].std()
    yt_bin = (yt >= thresh).astype(int)
    yp_bin = (yp >= thresh).astype(int)

    is_sigmoid = getattr(estimator, 'is_sigmoid_', None)

    return {
        'mape_pct': _mape(yt, yp),
        'mae': mean_absolute_error(yt, yp),
        'rmse': np.sqrt(mean_squared_error(yt, yp)),
        'r2': r2_score(yt, yp),
        'f1_threshold_cross': f1_score(yt_bin, yp_bin, average='binary', zero_division=1),
        'n_test_buckets': len(bkts),
        'used_sigmoid': is_sigmoid,
    }


def main(system_type='sockshop', split_mode='quantile'):
    df_multi = load_multi_service_data(None, system_type=system_type)
    services = SERVICES if system_type == 'sockshop' else None
    if services is None:
        raise SystemExit(f"He thong '{system_type}' chua duoc script nay ho tro.")

    rows = []
    for metric_name, metric_col, unit, scale in METRICS:
        for svc in services:
            wlc, tgc = f'{svc}_workload', f'{svc}_{metric_col}'
            if wlc not in df_multi.columns or tgc not in df_multi.columns:
                continue
            df = df_multi[[wlc, tgc]].dropna()
            df.columns = ['Workload', 'Target']
            if len(df) < 200:
                print(f"  [SKIP] {svc}/{metric_name}: khong du du lieu ({len(df)} mau)")
                continue
            if len(df) > 2000:
                df = df.sample(2000, random_state=42)

            if split_mode == 'quantile':
                df = df.sort_values('Workload').reset_index(drop=True)
                split = int(len(df) * 0.67)
                df_train, df_test = df.iloc[:split], df.iloc[split:]
            else:
                df_shuf = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                split = int(len(df_shuf) * 0.70)
                df_train, df_test = df_shuf.iloc[:split], df_shuf.iloc[split:]

            print(f"\n[{metric_name} | {svc}] n_train={len(df_train)} n_test={len(df_test)}")
            for mech_name in MECHANISMS:
                try:
                    res = fit_and_eval(mech_name, df_train, df_test)
                except Exception as e:
                    print(f"    {mech_name:<12}: LOI - {e}")
                    continue
                tag = ''
                if mech_name == 'sigmoid_sat':
                    tag = ' [sigmoid]' if res['used_sigmoid'] else ' [fallback=linear]'
                print(f"    {mech_name:<12}: MAPE={res['mape_pct']:6.1f}%  "
                      f"RMSE={res['rmse']:9.4f}  R2={res['r2']:6.3f}  "
                      f"F1_vuot_nguong={res['f1_threshold_cross']:.3f}{tag}")
                rows.append({'service': svc, 'metric': metric_name, 'mechanism': mech_name, **res})

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f'saturating_mechanism_trial_{split_mode}.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu chi tiet: {out_path}")

    print("\n" + "=" * 78)
    print("  TONG HOP (trung binh qua cac cap service x metric)")
    print("=" * 78)
    summary = df_out.groupby('mechanism').agg(
        mape_pct=('mape_pct', 'mean'),
        rmse=('rmse', 'mean'),
        r2=('r2', 'mean'),
        f1_threshold_cross=('f1_threshold_cross', 'mean'),
        n_pairs=('service', 'count'),
    ).round(4)
    print(summary.to_string())

    print("\n  MAPE trung binh theo metric:")
    print(df_out.groupby(['metric', 'mechanism'])['mape_pct'].mean().unstack().round(2).to_string())

    print("\n  So lan thang (MAPE thap nhat) tren tung cap (service,metric):")
    wins = df_out.loc[df_out.groupby(['service', 'metric'])['mape_pct'].idxmin()]
    print(wins['mechanism'].value_counts().to_string())

    summary_path = os.path.join(OUT_DIR, f'saturating_mechanism_trial_{split_mode}_summary.csv')
    summary.to_csv(summary_path)
    print(f"\n[OK] Da luu tong hop: {summary_path}")


if __name__ == '__main__':
    split_mode = sys.argv[1] if len(sys.argv) > 1 else 'quantile'
    main('sockshop', split_mode=split_mode)
