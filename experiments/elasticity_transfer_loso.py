# -*- coding: utf-8 -*-
"""
LEAVE-ONE-SYSTEM-OUT: DO CO GIAN (ELASTICITY) CO CHUYEN GIAO GIUA CAC HE THONG?
================================================================================
Cau hoi: mot he thong dich MOI (chua co lich su telemetry du de fit co che)
co the dung PRIOR do co gian hoc tu CAC HE KHAC de du bao khong -- hay moi
trien khai buoc phai tu do?

Day la dung kich ban "Deploying to a New Target System" (paper sec:general-spec):
khach hang moi co do thi phu thuoc + vai gio telemetry nen, KHONG co 3 thang
lich su tai cao de fit hoi quy tung node.

Don vi phan tich: co che Tier-2 CUNG SERVICE  <svc>_workload -> <svc>_cpu.
Chon no vi (a) co o MOI service cua ca 3 he (khong can do thi), (b) dung dai
luong RQ4 da do tren Alibaba, (c) do co gian gamma = b*W_bar/y_bar la KHONG
THU NGUYEN nen so sanh/pool duoc xuyen service lan xuyen he thong -- he so
hoi quy tho b thi khong (don vi phu thuoc thang do tung metric).

QUAN TRONG: gamma la do co gian DIEM cua chinh mo hinh tuyen tinh tai trung
binh, nen KHONG doi dang ham co che. Mo hinh van tuyen tinh (positive=True)
nhu dang trien khai; chi co THAM SO duoc chuyen sang khong gian khong thu
nguyen de pool. Ket luan o day khong dung/khong pha Kingman.

Giao thuc danh gia (tai su dung dung cong admissible cua RQ4):
  - Chia OOD theo tai: sort theo workload, train = 67% THAP, test = 33% CAO
    (giong evaluate_node_stability / Fast Path OOD Gold Standard) -- do dung
    cau hoi capacity forecasting: du bao o muc tai CHUA TUNG THAY.
  - Chi dung du lieu giai doan BINH THUONG (time < inject_time).
  - skill = 1 - MSE_model / MSE_constant  (constant = trung binh tap train).
    skill <= 0 nghia la khong hon du bao hang so.
  - negative control: hoan vi workload trong tap train -> skill PHAI sup <= 0.

Cac nhanh so sanh tren tap test:
  constant     : y_hat = mean(y_train)                       (skill == 0)
  own_fit      : LinearRegression(positive=True) tren du lieu CUA CHINH no
  transfer     : gamma_prior tu HAI HE KIA (khong he nao cua chinh no) +
                 trung binh nen cua chinh no  -> y_hat = y_bar + g*(y_bar/W_bar)*(W - W_bar)
  pooled       : gamma = lambda*gamma_own + (1-lambda)*gamma_prior,
                 lambda = tau^2/(tau^2 + se^2)  (empirical-Bayes shrinkage)
  neg_control  : own_fit sau khi hoan vi workload

Ket cuc BAC BO gia thuyet chuyen giao: transfer co skill <= 0 tren he bi giu
lai => do co gian KHONG chuyen giao duoc, moi trien khai phai tu do (va do
la mot hang moi cho bang General Specification, khong phai that bai cua bai).

Output: data/processed/scm_results/elasticity_transfer_loso.csv
Dung:   python experiments/elasticity_transfer_loso.py
"""

import os
import sys
import glob
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')

TRAIN_FRAC = 0.67
MIN_ROWS = 200          # toi thieu de fit mot co che
RNG = np.random.default_rng(42)

# (ten he, glob toi thu muc case, ten file metric)
SYSTEMS = {
    'SockShop':       (os.path.join(BASE_DIR, 'data', 'raw', 'RE2-SS', '*', '*'), 'simple_metrics.csv'),
    'TrainTicket':    (os.path.join(BASE_DIR, 'data', 'raw', 'trainticket', '*'), 'metrics.parquet'),
    'OnlineBoutique': (os.path.join(BASE_DIR, 'data', 'raw', 'RE2-OB', '*'),      'metrics.parquet'),
}


def _read_metrics(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)


def load_system_normal(system: str) -> pd.DataFrame:
    """Gom du lieu giai doan BINH THUONG (truoc inject_time) cua MOI case
    trong mot he thong thanh 1 DataFrame (cot giu nguyen ten <svc>_<metric>).
    """
    case_glob, metric_file = SYSTEMS[system]
    frames = []
    for case_dir in sorted(glob.glob(case_glob)):
        mp = os.path.join(case_dir, metric_file)
        ip = os.path.join(case_dir, 'inject_time.txt')
        if not (os.path.exists(mp) and os.path.exists(ip)):
            continue
        try:
            with open(ip) as f:
                inject_t = int(f.read().strip())
            df = _read_metrics(mp)
        except Exception:
            continue
        tcol = next((c for c in df.columns if c.lower() in ('time', 'timestamp', 'imte')), None)
        if tcol is None:
            continue
        df = df.rename(columns={tcol: 'time'})
        pre = df[df['time'] < inject_t]
        if len(pre) > 0:
            frames.append(pre)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def services_of(df: pd.DataFrame) -> list:
    w = {c[:-len('_workload')] for c in df.columns if c.endswith('_workload')}
    c = {c[:-len('_cpu')] for c in df.columns if c.endswith('_cpu')}
    return sorted(w & c)


def ood_split(w: np.ndarray, y: np.ndarray):
    """Sort theo workload, train = 67% THAP, test = 33% CAO."""
    order = np.argsort(w, kind='stable')
    w, y = w[order], y[order]
    k = int(len(w) * TRAIN_FRAC)
    return w[:k], y[:k], w[k:], y[k:]


def fit_linear(w_tr: np.ndarray, y_tr: np.ndarray):
    """Tra ve (b, a, se_b) cua hoi quy tuyen tinh khong am -- dung lop mo
    hinh DANG trien khai trong san pham (LinearRegression(positive=True)).
    """
    X = w_tr.reshape(-1, 1)
    m = LinearRegression(positive=True).fit(X, y_tr)
    b, a = float(m.coef_[0]), float(m.intercept_)
    resid = y_tr - m.predict(X)
    dof = max(len(w_tr) - 2, 1)
    sxx = float(((w_tr - w_tr.mean()) ** 2).sum())
    se_b = float(np.sqrt((resid ** 2).sum() / dof / sxx)) if sxx > 0 else np.inf
    return b, a, se_b


def skill(y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: float) -> float:
    mse_model = float(np.mean((y_true - y_pred) ** 2))
    mse_const = float(np.mean((y_true - y_train_mean) ** 2))
    return 1.0 - mse_model / mse_const if mse_const > 0 else np.nan


def predict_from_elasticity(w: np.ndarray, gamma: float, w_bar: float, y_bar: float) -> np.ndarray:
    """Dung do co gian gamma + trung binh nen cua CHINH he dich de dung lai
    mot du bao tuyen tinh: y_hat = y_bar + gamma*(y_bar/w_bar)*(w - w_bar).

    Day la tat ca nhung gi mot trien khai NGUOI (chua co lich su tai cao)
    co the cung cap: vai gio telemetry nen -> w_bar, y_bar.
    """
    return y_bar + gamma * (y_bar / w_bar) * (w - w_bar)


def collect_node_stats() -> pd.DataFrame:
    """Voi moi (he thong, service): fit rieng, tinh gamma va se(gamma), giu
    lai ca tap test de cham diem o buoc sau."""
    rows = []
    for system in SYSTEMS:
        df = load_system_normal(system)
        if df.empty:
            print(f"  [BO QUA] {system}: khong load duoc du lieu.")
            continue
        svcs = services_of(df)
        print(f"  [{system}] {len(df)} dong binh thuong, {len(svcs)} service co ca workload+cpu")
        for svc in svcs:
            sub = df[[f'{svc}_workload', f'{svc}_cpu']].dropna()
            if len(sub) < MIN_ROWS:
                continue
            w = sub[f'{svc}_workload'].to_numpy(dtype=float)
            y = sub[f'{svc}_cpu'].to_numpy(dtype=float)
            w_tr, y_tr, w_te, y_te = ood_split(w, y)
            if len(w_te) < 20 or w_tr.std() == 0 or y_tr.mean() == 0 or w_tr.mean() == 0:
                continue
            b, a, se_b = fit_linear(w_tr, y_tr)
            w_bar, y_bar = float(w_tr.mean()), float(y_tr.mean())
            gamma = b * w_bar / y_bar
            se_gamma = se_b * w_bar / y_bar
            rows.append({
                'system': system, 'service': svc, 'n': len(sub),
                'gamma_own': gamma, 'se_gamma': se_gamma,
                'b': b, 'a': a, 'w_bar': w_bar, 'y_bar': y_bar,
                '_w_tr': w_tr, '_y_tr': y_tr, '_w_te': w_te, '_y_te': y_te,
            })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("  LEAVE-ONE-SYSTEM-OUT: do co gian workload->CPU co chuyen giao khong?")
    print("=" * 78)

    stats = collect_node_stats()
    if stats.empty:
        print("[LOI] Khong thu duoc node nao.")
        return

    print(f"\n  Tong cong {len(stats)} (he thong, service) fit duoc.\n")

    results = []
    for _, r in stats.iterrows():
        held_out_system = r['system']
        # PRIOR chi tu HAI HE KIA -- khong bao gio thay he dang danh gia.
        prior_pool = stats[stats['system'] != held_out_system]
        gamma_prior = float(np.median(prior_pool['gamma_own']))

        # tau^2 (phuong sai GIUA cac service cua prior) uoc luong kieu
        # method-of-moments: var tong - trung binh se^2, san o 0.
        finite_se = prior_pool['se_gamma'].replace([np.inf, -np.inf], np.nan).dropna()
        tau2 = max(float(np.var(prior_pool['gamma_own'])) - float(np.mean(finite_se ** 2)), 0.0)
        se2 = float(r['se_gamma'] ** 2)
        lam = tau2 / (tau2 + se2) if np.isfinite(se2) and (tau2 + se2) > 0 else 0.0

        w_tr, y_tr = r['_w_tr'], r['_y_tr']
        w_te, y_te = r['_w_te'], r['_y_te']
        y_train_mean = float(y_tr.mean())

        # 1. own_fit
        pred_own = r['a'] + r['b'] * w_te
        s_own = skill(y_te, pred_own, y_train_mean)

        # 2. transfer (prior tu he khac + trung binh nen cua chinh no)
        pred_tr = predict_from_elasticity(w_te, gamma_prior, r['w_bar'], r['y_bar'])
        s_transfer = skill(y_te, pred_tr, y_train_mean)

        # 3. pooled (shrinkage)
        gamma_pooled = lam * r['gamma_own'] + (1 - lam) * gamma_prior
        pred_pool = predict_from_elasticity(w_te, gamma_pooled, r['w_bar'], r['y_bar'])
        s_pooled = skill(y_te, pred_pool, y_train_mean)

        # 4. negative control: hoan vi workload trong TRAIN roi fit lai
        w_perm = RNG.permutation(w_tr)
        b_n, a_n, _ = fit_linear(w_perm, y_tr)
        s_neg = skill(y_te, a_n + b_n * w_te, y_train_mean)

        results.append({
            'system': held_out_system, 'service': r['service'], 'n': r['n'],
            'gamma_own': r['gamma_own'], 'gamma_prior': gamma_prior,
            'gamma_pooled': gamma_pooled, 'lambda': lam,
            'skill_own': s_own, 'skill_transfer': s_transfer,
            'skill_pooled': s_pooled, 'skill_negcontrol': s_neg,
        })

    res = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, 'elasticity_transfer_loso.csv')
    res.to_csv(out_path, index=False)

    print("=" * 78)
    print("  KET QUA (skill trung vi; skill <= 0 = khong hon du bao hang so)")
    print("=" * 78)
    cols = ['skill_own', 'skill_transfer', 'skill_pooled', 'skill_negcontrol']
    summary = res.groupby('system')[cols].median().round(4)
    counts = res.groupby('system')['service'].count().rename('n_service')
    print(pd.concat([counts, summary], axis=1).to_string())

    print("\n  Ty le service co skill > 0:")
    frac = res.groupby('system')[cols].apply(lambda g: (g > 0).mean().round(3))
    print(frac.to_string())

    print("\n  Tren TOAN BO 3 he:")
    print(f"    skill trung vi : own={res['skill_own'].median():.4f}  "
          f"transfer={res['skill_transfer'].median():.4f}  "
          f"pooled={res['skill_pooled'].median():.4f}  "
          f"neg_control={res['skill_negcontrol'].median():.4f}")
    print(f"    %skill>0       : own={(res['skill_own']>0).mean():.1%}  "
          f"transfer={(res['skill_transfer']>0).mean():.1%}  "
          f"pooled={(res['skill_pooled']>0).mean():.1%}  "
          f"neg_control={(res['skill_negcontrol']>0).mean():.1%}")
    print(f"\n  gamma_prior theo tung he bi giu lai: "
          f"{res.groupby('system')['gamma_prior'].first().round(4).to_dict()}")
    print(f"  lambda (trong so giu uoc luong rieng) trung vi: {res['lambda'].median():.4f}")
    print(f"\n[OK] -> {out_path}")


if __name__ == '__main__':
    main()
