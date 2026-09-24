# -*- coding: utf-8 -*-
"""
THU NGHIEM 2: Doi mechanism co thuc su cai thien CHAT LUONG QUYET DINH khong?
==============================================================================
Boi canh: thu nghiem truoc (nonlinear_mechanism_trial.py) do tren BUCKET-AVERAGE
(8 diem/cap, khop voi cach CapacityAgent.get_metrics_for_service mo phong that)
va cho thay linear_pos > auto_gcm ro ret. Nhung RQ1 cua paper do tren
FULL-RESOLUTION (moi diem test tho, ~20k diem/cap) va cho ket qua NGUOC LAI:
linear_pos co MAPE cao hon auto_gcm ro ret (CPU Sock Shop 46.8% vs 35.7%).

Hai ket qua trai nguoc vi do 2 thu khac nhau:
  - Bucket-average MAPE  ~ do sai so theo XU HUONG TRUNG BINH (ky vong co dieu
    kien theo workload) -- dung voi muc dich CapacityAgent: du bao MUC TIEU HAO
    TAI NGUYEN KY VONG khi can thiep do(workload), khong phai 1 diem do rieng le.
  - Full-resolution MAPE ~ do sai so DIEM-DOI-DIEM, gom ca nhieu von co (CPU/Socket
    con bi anh huong boi nhieu tien trinh khac, GC, network jitter... ma 1 bien
    dau vao Workload khong the nao du bao duoc) -- phat hien nay TU BAN THAN
    paper da neu (R2 am o da so cap, "SCM khong giai thich duoc phan lon phuong
    sai ngoai mau").

Cau hoi thuc su can tra loi: mechanism nao giup CapacityAgent RA QUYET DINH
CANH BAO (SAFE/WARNING/CRITICAL) dung hon, do TREN TOAN BO diem test tho (khong
chi 8 bucket qua it de tin cay F1) -- ket hop duoc do phan giai day du cua RQ1
VOI cau hoi quyet dinh cua Fast Path.

Phuong phap: voi moi cap (service, metric), fit CA HAI mechanism tren cung
df_train (LOW 67%), du bao closed-form (khong Monte Carlo, nhanh) tren TOAN
BO df_test tho. Threshold = train_mean + 0.5*train_std (dung quy uoc da co
trong capacity_agent.py::train_fast_path). Do Precision/Recall/F1 tren
~20k diem/cap thay vi 8 bucket.

Output: data/processed/scm_results/mechanism_decision_quality_{system}.csv
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
from dowhy.gcm import AdditiveNoiseModel, EmpiricalDistribution
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data, SERVICES, METRICS, TRAINTICKET_SERVICES


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def _fit_auto(df_train):
    g = nx.DiGraph([('Workload', 'Target')])
    m = gcm.InvertibleStructuralCausalModel(g)
    gcm.auto.assign_causal_mechanisms(m, df_train)
    gcm.fit(m, df_train)
    return m


def _fit_linear_pos(df_train):
    g = nx.DiGraph([('Workload', 'Target')])
    m = gcm.InvertibleStructuralCausalModel(g)
    m.set_causal_mechanism('Target', AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))
    gcm.auto.assign_causal_mechanisms(m, df_train)  # override_models=False -> chi dien Workload (root)
    gcm.fit(m, df_train)
    return m


def evaluate_pair(df, min_rows=200):
    if df is None or len(df) < min_rows:
        return None
    df = df.sort_values('Workload').reset_index(drop=True)
    if len(df) > 4000:
        df = df.sample(4000, random_state=42).sort_values('Workload').reset_index(drop=True)
    split = int(len(df) * 0.67)
    df_train, df_test = df.iloc[:split], df.iloc[split:]
    if len(df_test) < 30 or df_train['Workload'].nunique() < 3:
        return None

    m_auto = _fit_auto(df_train)
    m_lin = _fit_linear_pos(df_train)

    X_test = df_test[['Workload']].values
    yt_full = df_test['Target'].values
    yp_auto = m_auto.causal_mechanism('Target').prediction_model.predict(X_test).ravel()
    yp_lin = m_lin.causal_mechanism('Target').prediction_model.predict(X_test).ravel()

    thresh = df_train['Target'].mean() + 0.5 * df_train['Target'].std()
    yt_bin = (yt_full >= thresh).astype(int)
    if yt_bin.sum() == 0 or yt_bin.sum() == len(yt_bin):
        return None  # threshold khong chia tach duoc lop nao tren test -> P/R/F1 vo nghia

    out = {'n_test': len(df_test), 'thresh': thresh, 'pos_rate': yt_bin.mean()}
    for name, yp in [('auto', yp_auto), ('linear_pos', yp_lin)]:
        yp_bin = (yp >= thresh).astype(int)
        out[f'{name}_mape'] = _mape(yt_full, yp)
        out[f'{name}_precision'] = precision_score(yt_bin, yp_bin, zero_division=0)
        out[f'{name}_recall'] = recall_score(yt_bin, yp_bin, zero_division=0)
        out[f'{name}_f1'] = f1_score(yt_bin, yp_bin, zero_division=0)
    return out


def run_sockshop():
    df_multi = load_multi_service_data(None, system_type='sockshop')
    rows = []
    for metric_name, metric_col, unit, scale in METRICS:
        for svc in SERVICES:
            wlc, tgc = f'{svc}_workload', f'{svc}_{metric_col}'
            if wlc not in df_multi.columns or tgc not in df_multi.columns:
                continue
            df = df_multi[[wlc, tgc]].dropna().rename(columns={wlc: 'Workload', tgc: 'Target'})
            res = evaluate_pair(df)
            if res is None:
                print(f"  [SKIP] {svc}/{metric_name}")
                continue
            res.update({'system': 'SockShop', 'service': svc, 'metric': metric_name})
            rows.append(res)
            print(f"  {svc:<12} {metric_name:<7} n={res['n_test']:>6} pos_rate={res['pos_rate']:.2f} | "
                  f"F1 auto={res['auto_f1']:.3f} linear={res['linear_pos_f1']:.3f} | "
                  f"MAPE auto={res['auto_mape']:6.1f}% linear={res['linear_pos_mape']:6.1f}%")
    return pd.DataFrame(rows)


def run_trainticket():
    df_all = load_multi_service_data(os.path.join(BASE_DIR, 'data', 'raw', 'trainticket'), system_type='trainticket')
    if df_all is None or df_all.empty:
        print("  [WARN] Khong load duoc du lieu Train Ticket, bo qua.")
        return pd.DataFrame()
    rows = []
    for metric_name, metric_col, unit, scale in METRICS:
        for svc in TRAINTICKET_SERVICES:
            wlc, tgc = f'{svc}_workload', f'{svc}_{metric_col}'
            if wlc not in df_all.columns or tgc not in df_all.columns:
                continue
            df = df_all[[wlc, tgc]].dropna().rename(columns={wlc: 'Workload', tgc: 'Target'})
            res = evaluate_pair(df, min_rows=300)
            if res is None:
                continue
            res.update({'system': 'TrainTicket', 'service': svc, 'metric': metric_name})
            rows.append(res)
            print(f"  {svc:<32} {metric_name:<7} n={res['n_test']:>6} | "
                  f"F1 auto={res['auto_f1']:.3f} linear={res['linear_pos_f1']:.3f}")
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("=" * 90)
    print("  SOCK SHOP")
    print("=" * 90)
    df_ss = run_sockshop()
    df_ss.to_csv(os.path.join(OUT_DIR, 'mechanism_decision_quality_sockshop.csv'), index=False)

    print("\n" + "=" * 90)
    print("  TRAIN TICKET")
    print("=" * 90)
    df_tt = run_trainticket()
    if not df_tt.empty:
        df_tt.to_csv(os.path.join(OUT_DIR, 'mechanism_decision_quality_trainticket.csv'), index=False)

    for name, df in [('Sock Shop', df_ss), ('Train Ticket', df_tt)]:
        if df.empty:
            continue
        print(f"\n{'=' * 90}\n  TONG HOP — {name} (n={len(df)} cap)\n{'=' * 90}")
        print(df[['auto_precision', 'auto_recall', 'auto_f1', 'auto_mape',
                  'linear_pos_precision', 'linear_pos_recall', 'linear_pos_f1', 'linear_pos_mape']].mean().round(3))
        f1_win_auto = (df['auto_f1'] > df['linear_pos_f1']).sum()
        f1_win_lin = (df['linear_pos_f1'] > df['auto_f1']).sum()
        f1_tie = (df['auto_f1'] == df['linear_pos_f1']).sum()
        mape_win_auto = (df['auto_mape'] < df['linear_pos_mape']).sum()
        mape_win_lin = (df['linear_pos_mape'] < df['auto_mape']).sum()
        print(f"F1 win:   auto={f1_win_auto}  linear_pos={f1_win_lin}  tie={f1_tie}  (n={len(df)})")
        print(f"MAPE win: auto={mape_win_auto}  linear_pos={mape_win_lin}  (n={len(df)})")
