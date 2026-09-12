# -*- coding: utf-8 -*-
"""
"TRAN LY THUYET" cua CPU MAPE: neu dung ham workload->CPU TOT NHAT CO THE
(trung binh cuc bo thuc trong du lieu, khong phai bat ky dang ham tham so
nao) thi MAPE thap nhat dat duoc la bao nhieu? So sanh voi MAPE cua model
hien tai (LinearRegression(positive=True)) de biet:
  - Neu model hien tai GAN san (floor): phan con lai la NHIEU THUC (khong
    lien quan workload), doi ham/them covariate KHONG giup duoc nua tru
    khi tim duoc covariate MOI that su co tin hieu (nhu backpressure).
  - Neu model hien tai XA san: co du dia de cai thien BANG CACH DOI HAM
    (piecewise/spline/nonlinear) MA KHONG can covariate moi.

Phuong phap floor: chia workload thanh N bin theo quantile (dam bao du
diem/bin de trung binh on dinh), du bao = trung binh CPU THUC trong bin
CHUA cai lay diem do (leave-one-out trong bin, tranh ro ri), tinh MAPE.
Day la uoc luong E[CPU | Workload=w] tot nhat co the tu du lieu, khong
gia dinh dang ham.

Output: data/processed/scm_results/cpu_mape_theoretical_floor.csv
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
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data, SERVICES, TRAINTICKET_SERVICES

N_BINS = 25


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def eval_node(df, wlc, cpuc):
    sub = df[[wlc, cpuc]].dropna()
    if len(sub) < 300:
        return None
    if len(sub) > 6000:
        sub = sub.sample(6000, random_state=42)
    sub = sub.sort_values(wlc).reset_index(drop=True)

    # --- Model hien tai: OOD protocol (train low 67% -> test high 33%), full-resolution ---
    split = int(len(sub) * 0.67)
    train, test = sub.iloc[:split], sub.iloc[split:]
    if train[wlc].nunique() < 3:
        return None
    m = LinearRegression(positive=True).fit(train[[wlc]], train[cpuc])
    yp_model = m.predict(test[[wlc]]).ravel()
    mape_model = _mape(test[cpuc].values, yp_model)

    # --- Tran ly thuyet: BIN-MEAN LEAVE-ONE-OUT tren TOAN BO du lieu (in-distribution,
    # vi tran ly thuyet la ve do NHIEU noi tai, khong phai ve kha nang ngoai suy) ---
    n_bins = min(N_BINS, sub[wlc].nunique())
    if n_bins < 3:
        return None
    sub = sub.copy()
    sub['bin'] = pd.qcut(sub[wlc], q=n_bins, duplicates='drop')
    grp = sub.groupby('bin', observed=True)[cpuc]
    bin_sum = grp.transform('sum')
    bin_n = grp.transform('count')
    # leave-one-out mean = (tong bin - gia tri hien tai) / (n bin - 1)
    loo_mean = (bin_sum - sub[cpuc]) / (bin_n - 1).replace(0, np.nan)
    valid = bin_n > 1
    mape_floor = _mape(sub.loc[valid, cpuc].values, loo_mean.loc[valid].values)

    # MAPE cua model hien tai do TREN CUNG PHAN BO in-distribution (de so sanh cong bang
    # voi floor, tach biet voi so OOD o tren)
    m_full = LinearRegression(positive=True).fit(sub[[wlc]], sub[cpuc])
    yp_full = m_full.predict(sub[[wlc]]).ravel()
    mape_model_indist = _mape(sub[cpuc].values, yp_full)

    return {
        'n': len(sub), 'n_bins': n_bins,
        'mape_model_OOD': round(mape_model, 2),
        'mape_model_indist': round(mape_model_indist, 2),
        'mape_floor_indist': round(mape_floor, 2),
        'gap_model_minus_floor': round(mape_model_indist - mape_floor, 2),
    }


if __name__ == '__main__':
    rows = []

    df_ss = load_multi_service_data(None, system_type='sockshop')
    for svc in SERVICES:
        res = eval_node(df_ss, f'{svc}_workload', f'{svc}_cpu')
        if res:
            res.update({'system': 'SockShop', 'service': svc})
            rows.append(res)

    df_tt = load_multi_service_data(os.path.join(BASE_DIR, 'data', 'raw', 'trainticket'), system_type='trainticket')
    for svc in TRAINTICKET_SERVICES:
        res = eval_node(df_tt, f'{svc}_workload', f'{svc}_cpu')
        if res:
            res.update({'system': 'TrainTicket', 'service': svc})
            rows.append(res)

    out = pd.DataFrame(rows)
    cols = ['system', 'service', 'n', 'mape_model_OOD', 'mape_model_indist', 'mape_floor_indist', 'gap_model_minus_floor']
    out = out[cols].sort_values(['system', 'gap_model_minus_floor'], ascending=[True, False])
    out_path = os.path.join(OUT_DIR, 'cpu_mape_theoretical_floor.csv')
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu: {out_path}")

    print("\n" + "=" * 70)
    print("  TONG HOP theo he thong")
    print("=" * 70)
    print(out.groupby('system')[['mape_model_indist', 'mape_floor_indist', 'gap_model_minus_floor']].mean().round(2))
