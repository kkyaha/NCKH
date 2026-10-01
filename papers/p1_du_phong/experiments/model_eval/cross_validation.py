# -*- coding: utf-8 -*-
"""
KIEM CHUNG CHEO (cross-validation) -- MO HINH CO TOT VA CO PHU HOP KHONG?  [HOI CUU]
======================================================================================
KHONG thay the nhanh dong bang (tien cuu). Day la bang chung HOI CUU: no tra loi
"dang mo hinh + quy trinh fit co tong quat khong?", con nhanh dong bang tra loi
"du doan truoc khi do co dung khong?". Hai cau hoi khac nhau, ca hai deu can.

BA TANG, moi tang mot cach CHIA KHAC NHAU -- vi du lieu co cau truc phan tang:

  [A] CO CHE (alpha, beta moi service) -- bo-mot-BAC-TAI
      608 hang cua SS-TRAIN KHONG phai 608 quan sat doc lap: cac hang trong cung mot
      bac tai dung chung container, phien, nhieu nen. Chia NGAU NHIEN theo hang thi
      moi fold deu co "anh em sinh doi" nam trong tap huan luyen -> R2 cao gia.
      Them nua: cong dung THAT cua mo hinh la NGOAI SUY (fit <=150 rps, du doan diem
      gay o 200+), ma k-fold ngau nhien lai la NOI SUY -> tra loi sai cau hoi.
      Tang nay chay CA HAI de do khoang cach giua chung.

  [B] DIEM GAY -- bo-mot-TINH-NANG, FIT LAI P2 BEN TRONG MOI FOLD
      15 o nhung chi 8 tinh nang doc lap: x1 va x2 cua cung tinh nang dung chung k,
      chung chain, chung ban cai -> chia fold theo O la ro ri (pseudo-replication).
      Va tham so P2 (c_s, x) dang fit tren promo/recs, nen neu khong fit lai trong
      fold thi MOI fold deu da nhin thay promo/recs.

  [C] SO SANH BASELINE duoi CUNG mot luat chia cua [A]
      statistical_rigor.py hien chia MOT lan (67% tai thap -> 33% tai cao). Tang nay
      lap lai tren nhieu fold de co phuong sai, thay vi mot con so don.

    python papers/p1_du_phong/experiments/model_eval/cross_validation.py
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while BASE != os.path.dirname(BASE) and not os.path.isdir(os.path.join(BASE, 'src')):
    BASE = os.path.dirname(BASE)
assert os.path.isdir(os.path.join(BASE, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
sys.path.insert(0, os.path.join(BASE, 'experiments'))  # noi _paths.py (goc repo)
sys.path.insert(0, os.path.join(BASE, 'src', 'scm'))
import _paths  # noqa: F401,E402  -- dua cac nhom con khac vao sys.path
import feasibility_predictor as FP  # noqa: E402
import evaluate_frozen as EF  # noqa: E402
from fit_feature_costs import build_rows  # noqa: E402
from queueing_regressor import QueueingLatencyRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # noqa: E402

SEED = 0
FROZEN = os.path.join(BASE, 'data', 'processed', 'frozen')
pd.set_option('display.width', 200)


def mape(y, p):
    m = np.abs(y) > 1e-9
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else np.nan


def baselines():
    return {
        'LinearReg': LinearRegression(),
        'GradBoost': GradientBoostingRegressor(random_state=SEED),
        'RandForest': RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
        'GaussProc': GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-2), random_state=SEED, normalize_y=True),
        'Queueing': QueueingLatencyRegressor(),
    }


def scm_fit_predict(w_tr, c_tr, w_te):
    """Dung DUNG dang co che cua bai: nnls tren [1, w] -> alpha + beta*w (khong am)."""
    from scipy.optimize import nnls
    (a, b), _ = nnls(np.column_stack([np.ones_like(w_tr), w_tr]), c_tr)
    return a + b * w_te


# ------------------------------------------------------------------ [A] co che
def tier_a(df):
    """Do RO RI cua chia ngau nhien -- so tren CUNG MOT tap kiem tra.

    Voi moi bac tai L: tap kiem tra = mot NUA so hang o bac L (cung hang cho ca hai nhanh).
      chan  : huan luyen tren cac bac KHAC  (khong he thay bac L)
      ro ri : huan luyen tren cung tung ay hang, nhung co ca nua con lai CUA BAC L
    Chenh lech giua hai cot = loi ich mo hinh nhan duoc chi vi da "nhin trom" bac dang kiem.
    So sanh o lan chay truoc (trung vi fold ngau nhien vs trung vi fold theo bac) la SAI:
    hai ben gop tren don vi khac nhau nen khong so duoc.
    """
    lv = FP.level_of(df).to_numpy(float)
    levels = np.unique(lv)
    rs = np.random.RandomState(SEED)
    rows = []
    for s in FP.SCORED:
        cw, cc = f'{s}_workload', f'{s}_cpu'
        if cw not in df or cc not in df:
            continue
        w, c = df[cw].to_numpy(float), df[cc].to_numpy(float)
        ok = np.isfinite(w) & np.isfinite(c)
        w, c, g = w[ok], c[ok], lv[ok]
        e_blocked, e_leaky, e_extrap = [], [], []
        for L in levels:
            idx_L = np.flatnonzero(g == L)
            idx_other = np.flatnonzero(g != L)
            if len(idx_L) < 6 or len(idx_other) < 10:
                continue
            rs.shuffle(idx_L)
            te, rest = idx_L[:len(idx_L) // 2], idx_L[len(idx_L) // 2:]
            n_tr = len(idx_other)                       # giu KICH THUOC tap huan luyen bang nhau
            tr_leaky = np.concatenate([rs.choice(idx_other, n_tr - len(rest), replace=False), rest])
            e_blocked.append(mape(c[te], scm_fit_predict(w[idx_other], c[idx_other], w[te])))
            e_leaky.append(mape(c[te], scm_fit_predict(w[tr_leaky], c[tr_leaky], w[te])))
        # NGOAI SUY that -- cong dung thuc cua mo hinh: fit tai thap, du doan tai cao
        cut = np.median(g)
        if (g > cut).sum() >= 3 and (g <= cut).sum() >= 10:
            e_extrap = mape(c[g > cut], scm_fit_predict(w[g <= cut], c[g <= cut], w[g > cut]))
        else:
            e_extrap = np.nan
        rows.append({'service': s, 'n_hang': len(w), 'n_bac_dung': len(e_blocked),
                     'MAPE_ro_ri': np.nanmedian(e_leaky),
                     'MAPE_chan': np.nanmedian(e_blocked),
                     'MAPE_ngoai_suy': e_extrap})
    return pd.DataFrame(rows).set_index('service')


# ------------------------------------------------------------------ [B] diem gay
def tier_b(ramp_dirs):
    """Bo-mot-TINH-NANG tren toan bo 8 tinh nang; FIT LAI P2 trong moi fold."""
    fz = json.load(open(os.path.join(FROZEN, 'predictions_frozen_RE2.json'), encoding='utf-8'))
    mech, cores, u_star = fz['mechanism'], fz['params']['cores'], fz['params']['u_star']
    feats_all = EF.DEV[1:] + EF.LOCKED + EF.INDEP + EF.PROSP        # bo 'base'

    runs, seen = [], set()
    for d in ramp_dirs:
        for r in EF.load_ramps(d, feats_all + ('base',)):
            key = (r['feature'], r['scale'], r['run'], d)
            if r['feature'] != 'base' and r['bp'].get('lo') is not None and key not in seen:
                seen.add(key)
                runs.append(r)
    # chain lay tu taxonomy (giong het ban dong bang) -> khong can nhieu file frozen
    preds = {('P1', f, sc): {'chain': FP.SOCKSHOP_CALL_CHAINS[FP.FEATURE_ARCHETYPE[f]]['services']}
             for f in feats_all for sc in (1.0, 2.0)}
    rows = build_rows(runs, mech, cores, preds)
    feats = sorted(rows.feature.unique())

    meas = {}
    for r in runs:
        meas.setdefault((r['feature'], r['scale']), []).append(r['bp']['lo'])

    out = []
    for held in feats:
        fit = FP.fit_feature_cost(rows[rows.feature != held], mech)   # <- P2 fit KHONG co tinh nang bi bo
        P_out = FP.FeasibilityPredictor(mech, cores, u_star, feature_cost=fit)
        P_in = FP.FeasibilityPredictor(mech, cores, u_star,           # doi chung: P2 fit tren TAT CA (co ro ri)
                                       feature_cost=FP.fit_feature_cost(rows, mech))
        for sc in sorted(rows[rows.feature == held].scale.unique()):
            lo = float(np.median(meas[(held, sc)]))
            rec = {'tinh_nang': held, 'scale': sc, 'do_duoc': lo, 'n_lap': len(meas[(held, sc)])}
            for nm, P, mode in (('P1', P_out, 'P1'), ('P2_ngoai_fold', P_out, 'P2'), ('P2_ro_ri', P_in, 'P2')):
                r_pred = P.breakpoint(mode=mode, feature=held, scale=sc)[0]
                rec[nm] = round(r_pred, 1)
                rec[f'saiso_{nm}'] = round(100 * (r_pred - lo) / lo, 1)
            out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ [C] baseline
def tier_c(df):
    """Cung luat chia [A] (chan theo bac tai) cho SCM va moi baseline."""
    lv = FP.level_of(df).to_numpy(float)
    levels = np.unique(lv)
    rows = []
    for s in FP.SCORED:
        cw, cc = f'{s}_workload', f'{s}_cpu'
        if cw not in df or cc not in df:
            continue
        w, c = df[cw].to_numpy(float), df[cc].to_numpy(float)
        ok = np.isfinite(w) & np.isfinite(c)
        w, c, g = w[ok], c[ok], lv[ok]
        per_model = {nm: [] for nm in ['SCM(nnls)'] + list(baselines())}
        for L in levels:
            tr, te = (g != L), (g == L)
            if te.sum() < 3 or tr.sum() < 10 or np.std(c[tr]) == 0:
                continue
            per_model['SCM(nnls)'].append(mape(c[te], scm_fit_predict(w[tr], c[tr], w[te])))
            for nm, mdl in baselines().items():
                try:
                    mdl.fit(w[tr].reshape(-1, 1), c[tr])
                    per_model[nm].append(mape(c[te], np.asarray(mdl.predict(w[te].reshape(-1, 1))).ravel()))
                except Exception:
                    per_model[nm].append(np.nan)
        if per_model['SCM(nnls)']:
            rows.append({'service': s, **{k: np.nanmedian(v) for k, v in per_model.items()}})
    return pd.DataFrame(rows).set_index('service')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', default=os.path.join(BASE, 'data', 'raw', 'SS-TRAIN'))
    ap.add_argument('--ramp-dirs', default='SS-LIMITS,SS-LIMITS-CLEAN',
                    help='thu muc ramp duoi data/raw/ cho tang [B]')
    a = ap.parse_args(argv)

    df = FP.select_train(FP.load_runs(a.train_dir), train_max=10 ** 9)
    print('=' * 100)
    print('  [A] CO CHE -- RO RI vs CHAN, tren CUNG tap kiem tra; va NGOAI SUY that')
    print('=' * 100)
    A = tier_a(df)
    print(A.round(2).to_string())
    infl = (A['MAPE_chan'] / A['MAPE_ro_ri']).median()
    print(f"\n  Chan theo bac tai lam MAPE x{infl:.2f} lan so voi de ro ri (cung tap kiem tra,")
    print('  cung kich thuoc tap huan luyen). Chenh lech = loi ich gia do "nhin trom" bac dang kiem.')

    print('\n' + '=' * 100)
    print('  [B] DIEM GAY -- bo-mot-TINH-NANG (n=8), P2 fit LAI trong moi fold')
    print('=' * 100)
    B = tier_b([os.path.join(BASE, 'data', 'raw', x.strip()) for x in a.ramp_dirs.split(',')])
    print(B[['tinh_nang', 'scale', 'n_lap', 'do_duoc', 'P1', 'P2_ngoai_fold', 'P2_ro_ri',
             'saiso_P1', 'saiso_P2_ngoai_fold', 'saiso_P2_ro_ri']].to_string(index=False))
    print('\n  TONG HOP theo TINH NANG (moi tinh nang = 1 don vi doc lap, lay trung vi cac scale):')
    per_feat = B.groupby('tinh_nang')[['saiso_P1', 'saiso_P2_ngoai_fold', 'saiso_P2_ro_ri']].median()
    agg = pd.DataFrame({'|sai so| trung vi': per_feat.abs().median(),
                        '|sai so| trung binh': per_feat.abs().mean(),
                        'so tinh nang du doan MUON (nguy hiem)': (per_feat > 0).sum(),
                        'n tinh nang': len(per_feat)})
    print(agg.round(2).to_string())

    print('\n' + '=' * 100)
    print('  [C] SO SANH BASELINE duoi CUNG luat chia chan-theo-bac-tai (MAPE %, thap = tot)')
    print('=' * 100)
    C = tier_c(df)
    print(C.round(2).to_string())
    print('\n  trung vi tren cac service: ' + ', '.join(f'{c}={C[c].median():.2f}%' for c in C.columns))
    print(f'\n  ⚠ n = {len(C)} service -- KHONG du luc kiem dinh de bac bo gia thuyet nao. '
          'Bang nay la MO TA, khong phai kiem dinh.')

    out = os.path.join(BASE, 'data', 'processed', 'scm_results', 'cross_validation.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    B.to_csv(out, index=False)
    print(f'\n[OK] chi tiet tang [B] -> {out}')


if __name__ == '__main__':
    main()
