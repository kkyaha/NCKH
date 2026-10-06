# -*- coding: utf-8 -*-
"""DOI CHUNG DANG CO CHE cho Bivariate Fast Path (Workload -> Target).

Gop tu ba script, nguyen van trong `archive/` -- xem muc "Nguon goc" duoi.
Ba script do la CUNG mot khung do (cung protocol, cung 21 cap service x metric,
cung `fit_and_eval`), chi khac bo co che dem ra so. Do trung lap do la 73% / 54%
/ 50% so dong. Gop lai thanh mot registry + co `--co-che`.

CAU HOI: doi dang hoi quy trong Fast Path (capacity_agent.py::train_fast_path --
duong nuoi truc tiep get_metrics_for_service() -> saturated_services ->
SAFE/WARNING/CRITICAL) co cai thien do chinh xac va F1 canh bao vuot nguong
trong DAI TAI THUC TE khong?

Dai tai thuc te, khong phai cuc bien: CapacityAgent chi van hanh trong
injection_delta_pct [5%, 50%] (src/agents/parser_agent.py: MIN_DELTA_PCT /
MAX_DELTA_PCT). Cac kich ban +150%/+300% trong evaluation_suite.py la benchmark
OOD rieng cho RQ4/RQ6/G7, KHONG phai dai van hanh.

CAC CO CHE
----------
  auto_gcm    : gcm.auto.assign_causal_mechanisms -- DANG DUNG trong Fast Path
  linear_pos  : LinearRegression(positive=True)   -- DANG DUNG trong Global DAG
                (capacity_agent.py::train_accurate_path, CPU/Mem/Tier-1)
  hgbr_mono   : HistGradientBoostingRegressor(monotonic_cst=[1]) -- phi tuyen,
                van dam bao dh/dWorkload >= 0
  sigmoid_sat : duong cong bao hoa 3 tham so, tran L uoc luong tu du lieu
  log_linear  : LinearRegression(positive=True) tren log1p(y) roi expm1 lai

Ca nam deu GIU tinh don dieu tang -- dieu kien tien quyet cua Proposition 2
(Certified Capacity Envelope). Do la rang buoc khong thuong luong duoc, nen
moi ung vien deu phai thoa truoc khi duoc dem ra do.

PROTOCOL
--------
  --chia quantile : train LOW 67% -> test HIGH 33% (ngoai suy NHE, la vung tai
                    "cao/co nguy co" that quan sat duoc). Day la protocol
                    `train_fast_path()` dang dung.
  --chia random   : xao tron 70/30 -> in-distribution thuan. Doi chung de TACH
                    "sai dang ham" khoi "bien ngoai suy".

TINH TAI LAP -- doc truoc khi vien dan bat ky con so nao
--------------------------------------------------------
Ba ban goc KHONG gieo hat cho `gcm.interventional_samples`, nen chung KHONG
tai lap duoc. Da do: chay `nonlinear_mechanism_trial.py quantile` hai lan lien
tiep cho MAPE trung binh 13.5006% roi 13.5660% -- lech trung binh 0.64 pp,
cao nhat 3.03 pp, ca 21/21 cap deu khac nhau.

Hau qua cho ket luan: khoang cach "linear_pos 12.90% vs hgbr_mono 13.67%" ma
ba ban goc dung de ket luan la 0.77 pp -- NHO HON BIEN DO NHIEU CHAY LAI. Noi
cach khac, so sanh MAPE tong hop mot minh KHONG CHUNG MINH duoc dieu gi. Dung
vien dan no nhu bang chung.

Ket luan "giu linear_pos" van dung, nhung dua tren hai thu khac, manh hon:
  (1) LAP LUAN CAU TRUC: mo hinh dua tren cay khong ngoai suy duoc qua pham vi
      Workload da thay trong train -- moi leaf ngoai vung train tra ve hang so,
      nen duong du bao phang ra thay vi tiep tuc tang, tuc danh gia THAP nguy co
      o vung tai cao. Day la tinh chat cua lop ham, khong phu thuoc mau nao.
  (2) THAT BAI DONG DEU: ca ba ho ham cung thua tren dung nhung node te nhat
      (catalogue_cpu > 90% MAPE o CA BA co che). Khi doi ca lop ham ma loi khong
      doi thi nguyen nhan khong phai "sai dang ham hoi quy".

Ban gop GIEO HAT (`HAT = 42`, `set_random_seed` + `np.random.seed` truoc moi lan
fit) nen tai lap duoc. Vi vay so cua ban gop KHONG trung so cua ban goc -- va do
la y muon, khong phai loi gop.

KET LUAN DA CHOT (dung lai tu ba ban goc, de khong phai chay lai moi hoi)
-----------------------------------------------------------------------
  * hgbr_mono thua linear_pos o protocol quantile theo SO CAP THANG: 10/21 vs
    7/21. (Khoang cach MAPE tong hop 12.90% vs 13.67% thi nam trong nhieu --
    xem muc TINH TAI LAP; dung vien dan no.)
  * sigmoid_sat cung thua linear_pos tren toan bo 21 cap.
  * Ca ba ho ham deu that bai GIONG NHAU tren dung nhung node te nhat
    (catalogue_cpu > 90% MAPE o CA BA co che; front-end_socket 52.0%;
    carts_socket 51.9%; carts_cpu 26.6%). Khi doi ca lop ham ma loi khong doi
    thi nguyen nhan KHONG phai "sai dang ham hoi quy".
  * Gia thuyet thay the: phan phoi lech nang / nhieu NHAN (multiplicative),
    tuc LECH PHA giua ham mat mat luc train (squared error, thang goc) va thuoc
    do luc danh gia (MAPE, thang tuong doi). `log_linear` la phep vá toi thieu
    cho gia thuyet do.
  => Ket qua: GIU linear_pos. Day la mot ket luan AM va no co gia tri -- no la
     ly do bai bao dung Linear ANM chu khong phai mot bo uoc luong hoc may.

NGUON GOC (ban goc nguyen van, de tai lap)
------------------------------------------
  archive/nonlinear_mechanism_trial.py   auto_gcm | linear_pos | hgbr_mono
  archive/saturating_mechanism_trial.py  linear_pos | hgbr_mono | sigmoid_sat
  archive/log_transform_trial.py         linear_pos | log_linear  (chi quantile)

Ban gop la TAP HOP TREN chat cua ba ban: giu nhanh dac biet `auto_gcm`, giu
truong `used_sigmoid` va `mae`. `log_transform_trial.py` khong ghi `mae` --
ban gop ghi them, khong bot gi.

Chay:
  python papers/p1_du_phong/experiments/edges/dang_co_che_doi_chung.py
  python .../dang_co_che_doi_chung.py --co-che linear_pos,log_linear --chia quantile
  python .../dang_co_che_doi_chung.py --co-che linear_pos,hgbr_mono,sigmoid_sat
"""
import argparse
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel
from dowhy.gcm.ml import SklearnRegressionModel
from dowhy.gcm.util.general import set_random_seed
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data, SERVICES, METRICS   # noqa: E402

N_PROJ = 500
HAT = 42        # xem ghi chu TINH TAI LAP trong docstring


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def _sigmoid(x, L, k, x0, b):
    return L / (1.0 + np.exp(-k * (x - x0))) + b


class SigmoidSaturationRegressor(BaseEstimator, RegressorMixin):
    """Duong cong bao hoa 3 tham so: y = L/(1+exp(-k(x-x0))) + b.

    L (tran tiem can) duoc UOC LUONG, khong co dinh = 100: du lieu CPU o day
    (xem data_processor.METRICS) khong chuan hoa ve thang % 0-100 dong deu qua
    cac service (vd catalogue_cpu max quan sat ~49 nhung p99 chi ~0.23 -- phan
    phoi lech nang; xem papers/p1_du_phong/docs/HE_THONG.md muc "Gioi han da
    biet" #4). Tran bi rang buoc trong [1.05, 6.0] x max(y_train) -- du rong de
    duong cong khong bi ep bao hoa som hon du lieu train da thay, nhung van HUU
    HAN (khac linear_pos ngoai suy vo han).

    fit() thu 3 diem khoi tao k (0.01, 0.1, 1.0 x 1/std(X)) vi curve_fit voi
    sigmoid rat nhay khoi tao; neu ca 3 khong hoi tu, hoac residual te hon
    linear don thuan tren TAP TRAIN, fallback ve LinearRegression(positive=True).
    """

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
                bounds = ([y_max * 1.05, 1e-6, X.min() - 3 * x_std, 0.0],
                          [y_max * 6.0, 10.0 / x_std, X.max() + 3 * x_std, y_max])
                popt, _ = curve_fit(_sigmoid, X, y, p0=p0, bounds=bounds, maxfev=5000)
                sse = np.sum((_sigmoid(X, *popt) - y) ** 2)
                if best is None or sse < best[1]:
                    best = (popt, sse)
            except Exception:
                continue

        lin = LinearRegression(positive=True).fit(X.reshape(-1, 1), y)
        lin_sse = np.sum((lin.predict(X.reshape(-1, 1)) - y) ** 2)

        if best is not None and best[1] < lin_sse:
            self.is_sigmoid_, self.params_ = True, best[0]
        else:
            self.is_sigmoid_, self.fallback_ = False, lin
        # coef_ ao, chi de tuong thich voi cac check doc `.coef_` ben ngoai (vd
        # _check_monotone_precondition trong capacity_agent.py) -- duong cong nay
        # LUON don dieu tang theo thiet ke (L>0, k>0).
        self.coef_ = np.array([1.0])
        return self

    def predict(self, X):
        X = np.asarray(X).ravel()
        return _sigmoid(X, *self.params_) if self.is_sigmoid_ else self.fallback_.predict(X.reshape(-1, 1))


class LogLinearRegressor(BaseEstimator, RegressorMixin):
    """y = expm1(a + b*x), b >= 0 (LinearRegression(positive=True) tren log1p(y)).
    Don dieu tang (exp la ham tang), nen GIU duoc precondition cua Proposition 2."""

    def fit(self, X, y):
        X = np.asarray(X).reshape(-1, 1)
        self.model_ = LinearRegression(positive=True)
        self.model_.fit(X, np.log1p(np.clip(np.asarray(y).ravel(), 0, None)))
        self.coef_ = self.model_.coef_      # >=0 tren thang log -> don dieu tren thang goc
        return self

    def predict(self, X):
        return np.expm1(self.model_.predict(np.asarray(X).reshape(-1, 1)))


CO_CHE = {
    'auto_gcm': None,                       # dac biet: de gcm.auto tu chon
    'linear_pos': lambda: LinearRegression(positive=True),
    'hgbr_mono': lambda: HistGradientBoostingRegressor(
        monotonic_cst=[1], max_depth=4, max_iter=150, random_state=42),
    'sigmoid_sat': lambda: SigmoidSaturationRegressor(),
    'log_linear': lambda: LogLinearRegressor(),
}


def fit_and_eval(ten, df_train, df_test, hat=HAT):
    # GIEO HAT truoc moi lan fit+sample. Ban goc KHONG gieo, va do la mot loi
    # that: chay lai cung mot script cho MAPE lech trung binh 0.64 pp, cao nhat
    # 3.03 pp, ca 21/21 cap deu khac. Xem muc TINH TAI LAP trong docstring.
    set_random_seed(hat)
    np.random.seed(hat)
    g = nx.DiGraph()
    g.add_edge('Workload', 'Target')
    model = gcm.InvertibleStructuralCausalModel(g)

    estimator = None
    if ten == 'auto_gcm':
        gcm.auto.assign_causal_mechanisms(model, df_train)
    else:
        estimator = CO_CHE[ten]()
        model.set_causal_mechanism('Target', AdditiveNoiseModel(SklearnRegressionModel(estimator)))
        # override_models=False (mac dinh) -> chi gan phan phoi thuc nghiem cho node
        # goc 'Workload', giu nguyen mechanism 'Target' vua set thu cong.
        gcm.auto.assign_causal_mechanisms(model, df_train)
    gcm.fit(model, df_train)

    df_test = df_test.copy()
    df_test['bkt'] = pd.qcut(df_test['Workload'],
                             q=min(8, df_test['Workload'].nunique()), duplicates='drop')
    bkts = df_test.groupby('bkt', observed=True)[['Workload', 'Target']].mean()

    y_true, y_pred = [], []
    for _, row in bkts.iterrows():
        dp = gcm.interventional_samples(
            model, interventions={'Workload': lambda x, w=row['Workload']: w},
            num_samples_to_draw=N_PROJ)
        y_true.append(row['Target'])
        y_pred.append(dp['Target'].mean())

    yt, yp = np.array(y_true), np.array(y_pred)
    thresh = df_train['Target'].mean() + 0.5 * df_train['Target'].std()
    yt_bin, yp_bin = (yt >= thresh).astype(int), (yp >= thresh).astype(int)
    return {
        'mape_pct': _mape(yt, yp),
        'mae': mean_absolute_error(yt, yp),
        'rmse': np.sqrt(mean_squared_error(yt, yp)),
        'r2': r2_score(yt, yp),
        'f1_threshold_cross': f1_score(yt_bin, yp_bin, average='binary', zero_division=1),
        'n_test_buckets': len(bkts),
        'used_sigmoid': getattr(estimator, 'is_sigmoid_', None),
    }


def chia_tap(df, cach):
    """quantile: train LOW 67% -> test HIGH 33% (protocol that cua Fast Path).
    random:   xao tron 70/30 -> in-distribution thuan (doi chung)."""
    if cach == 'quantile':
        d = df.sort_values('Workload').reset_index(drop=True)
        k = int(len(d) * 0.67)
    else:
        d = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        k = int(len(d) * 0.70)
    return d.iloc[:k], d.iloc[k:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--co-che', default='auto_gcm,linear_pos,hgbr_mono,sigmoid_sat,log_linear',
                    help='danh sach cach nhau boi dau phay; chon trong ' + ','.join(CO_CHE))
    ap.add_argument('--chia', default='quantile', choices=('quantile', 'random'))
    ap.add_argument('--he', default='sockshop')
    ap.add_argument('--hau-to', default='', help='hau to cho ten tep ra, de khong ghi de ket qua cu')
    a = ap.parse_args()

    ds = [x.strip() for x in a.co_che.split(',') if x.strip()]
    if [x for x in ds if x not in CO_CHE]:
        raise SystemExit(f'co che khong biet: {[x for x in ds if x not in CO_CHE]}')
    if a.he != 'sockshop':
        raise SystemExit(f"he '{a.he}' chua duoc ho tro (SERVICES chi dinh nghia cho sockshop)")

    df_multi = load_multi_service_data(None, system_type=a.he)
    rows = []
    for metric_name, metric_col, _unit, _scale in METRICS:
        for svc in SERVICES:
            wlc, tgc = f'{svc}_workload', f'{svc}_{metric_col}'
            if wlc not in df_multi.columns or tgc not in df_multi.columns:
                continue
            df = df_multi[[wlc, tgc]].dropna()
            df.columns = ['Workload', 'Target']
            if len(df) < 200:
                print(f'  [BO QUA] {svc}/{metric_name}: khong du du lieu ({len(df)} mau)')
                continue
            if len(df) > 2000:
                df = df.sample(2000, random_state=42)
            df_train, df_test = chia_tap(df, a.chia)

            print(f'\n[{metric_name} | {svc}] n_train={len(df_train)} n_test={len(df_test)}')
            for ten in ds:
                try:
                    r = fit_and_eval(ten, df_train, df_test)
                except Exception as e:
                    print(f'    {ten:<12}: LOI - {e}')
                    continue
                tag = ''
                if ten == 'sigmoid_sat':
                    tag = ' [sigmoid]' if r['used_sigmoid'] else ' [fallback=linear]'
                print(f"    {ten:<12}: MAPE={r['mape_pct']:6.1f}%  RMSE={r['rmse']:9.4f}  "
                      f"R2={r['r2']:6.3f}  F1_vuot_nguong={r['f1_threshold_cross']:.3f}{tag}")
                rows.append({'service': svc, 'metric': metric_name, 'co_che': ten, **r})

    d = pd.DataFrame(rows)
    ht = a.hau_to or a.chia
    p1 = os.path.join(OUT_DIR, f'dang_co_che_{ht}.csv')
    d.to_csv(p1, index=False)
    print(f'\n[OK] chi tiet: {p1}')

    print('\n' + '=' * 78)
    print(f'  TONG HOP (trung binh qua {d.groupby(["service", "metric"]).ngroups} cap service x metric)')
    print('=' * 78)
    s = d.groupby('co_che').agg(mape_pct=('mape_pct', 'mean'), rmse=('rmse', 'mean'),
                                r2=('r2', 'mean'),
                                f1_threshold_cross=('f1_threshold_cross', 'mean'),
                                n_pairs=('service', 'count')).round(4)
    print(s.to_string())
    p2 = os.path.join(OUT_DIR, f'dang_co_che_{ht}_summary.csv')
    s.to_csv(p2)
    print(f'\n[OK] tong hop: {p2}')
    print('\nSo cap moi co che THANG (MAPE thap nhat):')
    thang = d.loc[d.groupby(['service', 'metric']).mape_pct.idxmin(), 'co_che'].value_counts()
    print(thang.to_string())


if __name__ == '__main__':
    main()
