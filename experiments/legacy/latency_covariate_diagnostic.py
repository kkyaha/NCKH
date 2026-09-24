# -*- coding: utf-8 -*-
"""
CHAN DOAN: them covariate nao giai thich duoc carts_latency-50/orders_latency-50?
====================================================================================
Boi canh: da xac nhan (hoi thoai truoc, khong phai suy doan) carts_latency-50 va
orders_latency-50 co MAPE OOD (quantile: train LOW 67% -> test HIGH 33%) lan
luot la 961.6% va 100.4% voi QueueingLatencyRegressor -- GIONG HET baseline
hang so (LinearRegression(positive=True) tu fit he so=0). Tuong quan Workload-
Latency rat yeu (pearson carts=-0.30, orders=-0.14). Ket luan: khong phai sai
dang ham, ma la THIEU BIEN -- workload cua CHINH service do khong mang du tin
hieu.

Khac voi H3 trong call_chain_neighbor_diagnostic.py (kiem dinh CPU caller ->
CPU callee, dung huong "nghen tai nguyen tu nguoi goi"), o day dung HUONG
NGUOC LAI cho Latency: do latency cua mot service = thoi gian xu ly rieng +
TONG latency cac cuoc goi ha nguon (distributed-tracing intuition: neu
service B goi C, latency cua B bi keo dai boi latency cua C). Theo dung canh
HTTP that (src/graph/sockshop_agent_graph.json):
  orders -> carts, orders -> user, orders -> payment, orders -> shipping
  front-end -> carts, front-end -> orders

  => carts KHONG co callee nao trong 7 core services (chi goi carts-db, ngoai
     pham vi) -- thu covariate CALLER thay vi callee (front-end, orders).
  => orders CO 4 callee trong 7 core services -- thu latency cua tung callee
     lam covariate thu 2, dung logic "response time = own + downstream calls".

Phuong phap: GIU NGUYEN protocol quantile OOD (train LOW 67% -> test HIGH
33%) va do MAPE tren bucket trung binh -- CHINH XAC nhu da dung xuyen suot
hoi thoai nay, khong dung R2 in-sample (de tranh vacuity da canh bao trong
docs/HE_THONG.md (muc 5)). Bao gom LinearRegression KHONG rang buoc dau (positive=
False) o ca 2 bien, vi da biet rang buoc duong triet tieu tin hieu am.

Output: data/processed/scm_results/latency_covariate_diagnostic.csv
"""

import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data

CANDIDATES = {
    'carts_latency-50': ['front-end_cpu', 'front-end_latency-50', 'orders_cpu', 'orders_workload'],
    'orders_latency-50': ['carts_latency-50', 'user_latency-50', 'payment_latency-50', 'shipping_latency-50'],
}
OWN_WORKLOAD = {'carts_latency-50': 'carts_workload', 'orders_latency-50': 'orders_workload'}


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def eval_ood(df, feature_cols, target_col, positive):
    cols = feature_cols + [target_col]
    d = df[cols].dropna()
    if len(d) > 2000:
        d = d.sample(2000, random_state=42)
    d = d.sort_values(feature_cols[0]).reset_index(drop=True)
    split = int(len(d) * 0.67)
    dtr, dte = d.iloc[:split], d.iloc[split:]
    if dtr[feature_cols].nunique().min() < 2:
        return None
    m = LinearRegression(positive=positive).fit(dtr[feature_cols], dtr[target_col])
    dte2 = dte.copy()
    dte2['bkt'] = pd.qcut(dte2[feature_cols[0]], q=min(8, dte2[feature_cols[0]].nunique()), duplicates='drop')
    bkt = dte2.groupby('bkt', observed=True)[feature_cols + [target_col]].mean()
    pred = m.predict(bkt[feature_cols])
    return {
        'mape': _mape(bkt[target_col], pred),
        'coefs': dict(zip(feature_cols, np.round(m.coef_, 4))),
        'n': len(d),
    }


def main():
    df = load_multi_service_data(None, system_type='sockshop')
    rows = []
    for target, extras in CANDIDATES.items():
        own_wl = OWN_WORKLOAD[target]
        for positive in (True, False):
            base = eval_ood(df, [own_wl], target, positive)
            if base is None:
                continue
            print(f"\n[{target}] baseline (own_workload only, positive={positive}): "
                  f"MAPE={base['mape']:.1f}%  coef={base['coefs']}")
            rows.append({'target': target, 'features': own_wl, 'positive': positive,
                         'mape': round(base['mape'], 1), 'coefs': base['coefs']})
            for extra in extras:
                if extra not in df.columns:
                    print(f"    [SKIP] {extra} khong co trong data")
                    continue
                res = eval_ood(df, [own_wl, extra], target, positive)
                if res is None:
                    continue
                tag = 'CAI THIEN' if res['mape'] < base['mape'] - 1.0 else (
                    'XAU DI' if res['mape'] > base['mape'] + 1.0 else 'khong doi')
                print(f"    +{extra:<22} MAPE={res['mape']:7.1f}%  coef={res['coefs']}  [{tag}]")
                rows.append({'target': target, 'features': f'{own_wl}+{extra}', 'positive': positive,
                             'mape': round(res['mape'], 1), 'coefs': res['coefs']})

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'latency_covariate_diagnostic.csv')
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu: {out_path}")


if __name__ == '__main__':
    main()
