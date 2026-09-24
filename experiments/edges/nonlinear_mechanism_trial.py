# -*- coding: utf-8 -*-
"""
THU NGHIEM: Co che hoi quy (mechanism) nao chinh xac hon trong DAI TAI THUC TE?
================================================================================
Boi canh: CapacityAgent that su chi van hanh trong injection_delta_pct thuoc
[5%, 50%] (xem src/agents/parser_agent.py: MIN_DELTA_PCT/MAX_DELTA_PCT) -- cac
kich ban +150%/+300% trong evaluation_suite.py chi la benchmark OOD rieng cho
RQ4/RQ6/G7, KHONG phai dai van hanh thuc te cua agent. Theo thao luan, uu tien
lan nay la DO CHINH XAC va CHAT LUONG CANH BAO trong dai tai thuc/co nguy co,
khong phai tinh dung dau (sign) o cuc bien ngoai suy.

Cau hoi: doi co che hoi quy trong Bivariate Fast Path
(capacity_agent.py::train_fast_path -- day la duong nuoi truc tiep
get_metrics_for_service() -> saturated_services -> SAFE/WARNING/CRITICAL)
sang mot mo hinh phi tuyen co rang buoc don dieu
(HistGradientBoostingRegressor(monotonic_cst=[1])) co cai thien do chinh xac
va F1 canh bao vuot nguong so voi hien trang khong?

Phuong phap: GIU NGUYEN protocol Train(LOW 67% workload) -> Test(HIGH 33%
workload) ma train_fast_path() dang dung (day la vung tai "cao/co nguy co"
thuc su quan sat duoc trong du lieu, khong phai OOD nhan tao), roi do
MAPE/RMSE/R2/F1(vuot nguong) cho 3 co che tren CUNG 21 cap (service, metric):

  A) auto_gcm   : gcm.auto.assign_causal_mechanisms -- DANG DUNG trong
                  production Fast Path (capacity_agent.py::train_fast_path).
  B) linear_pos : LinearRegression(positive=True) -- DANG DUNG trong Global
                  DAG (capacity_agent.py::train_accurate_path, CPU/Mem/Tier-1).
  C) hgbr_mono  : HistGradientBoostingRegressor(monotonic_cst=[1]) -- DE XUAT
                  (phi tuyen + van dam bao dh/dWorkload >= 0).

Output: data/processed/scm_results/mechanism_trial_fastpath.csv
        + bang tong hop in ra console.
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

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


MECHANISMS = {
    'auto_gcm': None,  # dac biet: dung gcm.auto.assign_causal_mechanisms
    'linear_pos': lambda: LinearRegression(positive=True),
    'hgbr_mono': lambda: HistGradientBoostingRegressor(
        monotonic_cst=[1], max_depth=4, max_iter=150, random_state=42
    ),
}


def fit_and_eval(mech_name, df_train, df_test):
    g = nx.DiGraph()
    g.add_edge('Workload', 'Target')
    model = gcm.InvertibleStructuralCausalModel(g)

    if mech_name == 'auto_gcm':
        gcm.auto.assign_causal_mechanisms(model, df_train)
    else:
        estimator = MECHANISMS[mech_name]()
        model.set_causal_mechanism(
            'Target', AdditiveNoiseModel(SklearnRegressionModel(estimator))
        )
        # override_models=False (mac dinh) -> chi gan phan phoi thuc nghiem cho
        # node goc 'Workload', giu nguyen mechanism 'Target' vua set thu cong.
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

    return {
        'mape_pct': _mape(yt, yp),
        'mae': mean_absolute_error(yt, yp),
        'rmse': np.sqrt(mean_squared_error(yt, yp)),
        'r2': r2_score(yt, yp),
        'f1_threshold_cross': f1_score(yt_bin, yp_bin, average='binary', zero_division=1),
        'n_test_buckets': len(bkts),
    }


def main(system_type='sockshop', split_mode='quantile'):
    data_dir = None
    df_multi = load_multi_service_data(data_dir, system_type=system_type)
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
                # Protocol thuc te cua Fast Path: train LOW 67% -> test HIGH 33%
                # (van la 1 dang ngoai suy NHE, khong phai in-distribution thuan)
                df = df.sort_values('Workload').reset_index(drop=True)
                split = int(len(df) * 0.67)
                df_train, df_test = df.iloc[:split], df.iloc[split:]
            else:
                # Protocol doi chung: xao tron ngau nhien -> in-distribution thuan,
                # de tach bach "misfit phi tuyen" khoi "extrapolation boundary".
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
                print(f"    {mech_name:<12}: MAPE={res['mape_pct']:6.1f}%  "
                      f"RMSE={res['rmse']:9.4f}  R2={res['r2']:6.3f}  "
                      f"F1_vuot_nguong={res['f1_threshold_cross']:.3f}")
                rows.append({'service': svc, 'metric': metric_name, 'mechanism': mech_name, **res})

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f'mechanism_trial_fastpath_{split_mode}.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu chi tiet: {out_path}")

    print("\n" + "=" * 78)
    print("  TONG HOP (trung binh qua 21 cap service x metric)")
    print("=" * 78)
    summary = df_out.groupby('mechanism').agg(
        mape_pct=('mape_pct', 'mean'),
        rmse=('rmse', 'mean'),
        r2=('r2', 'mean'),
        f1_threshold_cross=('f1_threshold_cross', 'mean'),
        n_pairs=('service', 'count'),
    ).round(4)
    print(summary.to_string())
    summary_path = os.path.join(OUT_DIR, f'mechanism_trial_fastpath_{split_mode}_summary.csv')
    summary.to_csv(summary_path)
    print(f"\n[OK] Da luu tong hop: {summary_path}")


if __name__ == '__main__':
    split_mode = sys.argv[1] if len(sys.argv) > 1 else 'quantile'
    main('sockshop', split_mode=split_mode)
