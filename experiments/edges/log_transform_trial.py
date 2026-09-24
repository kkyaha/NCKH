# -*- coding: utf-8 -*-
"""
THU NGHIEM CHAN DOAN: log-transform co sua duoc 4 node lech nang khong?
==========================================================================
Boi canh: hgbr_mono (nonlinear_mechanism_trial.py) va sigmoid_sat
(saturating_mechanism_trial.py) deu THUA linear_pos tren TOAN BO 21 cap --
tuc la van de KHONG phai "sai dang ham hoi quy" (moi ho vong doi ca lop ham
deu that bai giong nhau tren dung nhung node te nhat: catalogue_cpu van
>90% MAPE o CA BA co che). Dieu do goi y nguyen nhan khac: PHAN PHOI LECH
NANG / nhieu nhan (multiplicative noise), khong phai duong cong sai dang.

4 node te nhat hien tai (do tu accuracy_df that, protocol quantile OOD):
  catalogue_cpu   96.0% MAPE (POOR)
  front-end_socket 52.0% MAPE (POOR)
  carts_socket    51.9% MAPE (POOR)
  carts_cpu       26.6% MAPE (POOR)
Ca 4 deu la du lieu KHONG AM, thuong lech phai (mot vai tinh huong tai dot
bien lam gia tri tang vot) -- dac diem kinh dien cua nhieu NHAN (multiplica-
tive noise, epsilon nhan chu khong cong), noi OLS tren thang GOC bi outlier
chi phoi ham so mat (MSE) trong khi MAPE (thang tuong doi) lai la thuoc do
danh gia -- SU LECH PHA giua ham mat mat luc train (squared error, thang
goc) va thuoc do luc danh gia (tuong doi, MAPE) chinh la nghi can hang dau,
KHONG phai ban than LinearRegression khong du manh.

Fix toi thieu: hoi quy tren log1p(target) roi exp lai (LogLinearRegressor).
LinearRegression(positive=True) tren thang log VAN don dieu tang khi ve lai
thang goc (exp la ham tang), nen KHONG lam mat tinh chat can cho Proposition 2.
Day la thay doi it nhat co the -- neu no sua duoc 4 node tren MA KHONG lam
te hon cac node dang tot, gia thuyet "lech phan phoi" duoc xac nhan; neu
khong, quay lai tim nguyen nhan khac (vd thieu covariate).

Phuong phap: GIU NGUYEN protocol quantile (train LOW 67% -> test HIGH 33%)
nhu 2 file truoc, so linear_pos (dang dung) vs log_linear tren CA 21 cap.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data, SERVICES, METRICS

N_PROJ = 500


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


class LogLinearRegressor(BaseEstimator, RegressorMixin):
    """y = expm1(a + b*x), b>=0 (LinearRegression(positive=True) tren log1p(y)).
    Don dieu tang (exp la ham tang), nen giu duoc precondition cua Proposition 2."""

    def fit(self, X, y):
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).ravel()
        self.model_ = LinearRegression(positive=True)
        self.model_.fit(X, np.log1p(np.clip(y, 0, None)))
        self.coef_ = self.model_.coef_  # >=0 tren thang log -> monotone tren thang goc
        return self

    def predict(self, X):
        X = np.asarray(X).reshape(-1, 1)
        return np.expm1(self.model_.predict(X))


MECHANISMS = {
    'linear_pos': lambda: LinearRegression(positive=True),
    'log_linear': lambda: LogLinearRegressor(),
}


def fit_and_eval(mech_name, df_train, df_test):
    g = nx.DiGraph()
    g.add_edge('Workload', 'Target')
    model = gcm.InvertibleStructuralCausalModel(g)
    estimator = MECHANISMS[mech_name]()
    model.set_causal_mechanism('Target', AdditiveNoiseModel(SklearnRegressionModel(estimator)))
    gcm.auto.assign_causal_mechanisms(model, df_train)
    gcm.fit(model, df_train)

    df_test = df_test.copy()
    df_test['bkt'] = pd.qcut(df_test['Workload'], q=min(8, df_test['Workload'].nunique()), duplicates='drop')
    bkts = df_test.groupby('bkt', observed=True)[['Workload', 'Target']].mean()

    y_true, y_pred = [], []
    for _, row in bkts.iterrows():
        wl_val = row['Workload']
        dp = gcm.interventional_samples(model, interventions={'Workload': lambda x, w=wl_val: w}, num_samples_to_draw=N_PROJ)
        y_true.append(row['Target'])
        y_pred.append(dp['Target'].mean())

    yt, yp = np.array(y_true), np.array(y_pred)
    thresh = df_train['Target'].mean() + 0.5 * df_train['Target'].std()
    yt_bin, yp_bin = (yt >= thresh).astype(int), (yp >= thresh).astype(int)
    return {
        'mape_pct': _mape(yt, yp), 'rmse': np.sqrt(mean_squared_error(yt, yp)),
        'r2': r2_score(yt, yp), 'f1': f1_score(yt_bin, yp_bin, average='binary', zero_division=1),
    }


def main():
    df_multi = load_multi_service_data(None, system_type='sockshop')
    rows = []
    for metric_name, metric_col, unit, scale in METRICS:
        for svc in SERVICES:
            wlc, tgc = f'{svc}_workload', f'{svc}_{metric_col}'
            if wlc not in df_multi.columns or tgc not in df_multi.columns:
                continue
            df = df_multi[[wlc, tgc]].dropna()
            df.columns = ['Workload', 'Target']
            if len(df) < 200:
                continue
            if len(df) > 2000:
                df = df.sample(2000, random_state=42)
            df = df.sort_values('Workload').reset_index(drop=True)
            split = int(len(df) * 0.67)
            df_train, df_test = df.iloc[:split], df.iloc[split:]

            print(f"\n[{metric_name} | {svc}]")
            for mech_name in MECHANISMS:
                try:
                    res = fit_and_eval(mech_name, df_train, df_test)
                except Exception as e:
                    print(f"    {mech_name:<12}: LOI - {e}")
                    continue
                print(f"    {mech_name:<12}: MAPE={res['mape_pct']:7.1f}%  RMSE={res['rmse']:9.4f}  "
                      f"R2={res['r2']:6.3f}  F1={res['f1']:.3f}")
                rows.append({'service': svc, 'metric': metric_name, 'mechanism': mech_name, **res})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, 'log_transform_trial_quantile.csv'), index=False)

    print("\n" + "=" * 78)
    piv = df_out.pivot_table(index=['metric', 'service'], columns='mechanism', values='mape_pct')
    piv['delta'] = piv['log_linear'] - piv['linear_pos']
    print(piv.round(2).to_string())
    print("\nTONG HOP:")
    print(df_out.groupby('mechanism')['mape_pct'].mean().round(2).to_string())
    print("\nSo node CAI THIEN (delta<0) vs XAU DI (delta>0):",
          (piv['delta'] < -0.01).sum(), 'vs', (piv['delta'] > 0.01).sum())


if __name__ == '__main__':
    main()
