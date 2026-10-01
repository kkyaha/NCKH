# -*- coding: utf-8 -*-
"""CO CHE CUA VI PHAM SLO KHI THEM TINH NANG -- phan tich tai lap duoc.

Cau hoi: dai luong nao quyet dinh tinh kha thi, va dai luong nao DU PHONG DUOC?

Chay:  python experiments/model_eval/mechanism_analysis.py [--part all|signal|control|burst|auc]

Cac phan:
  signal   S1  tin hieu cua 7 tai nguyen x 7 service tren du lieu do he binh thuong
  control  S2  DOI CHUNG AM: khong tran (SS-TRAIN) vs tran RE2 (ramp) o cung tai
  burst    S3  GIA THUYET DO BUNG NO -- da BAC BO, giu lai de khong ai lam lai
  auc      S4  so sanh suc phan biet vi pham, 6 dac ta + bootstrap tren HIEU SO

Ket luan da chot (xem docs/CO_CHE_KHA_THI.md): throttling hon muc su dung o moi
dac ta, va phan hon KHONG suy duoc tu muc su dung bang bat ky phep don dieu nao.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
sys.path[:0] = [os.path.join(_P, 'src', 'scm')]
import measured_data as MD                                          # noqa: E402
import feasibility_predictor as FP                                  # noqa: E402

LIMITS = json.load(open(os.path.join(_P, 'deploy', 'sockshop', 'limits.json'), encoding='utf-8'))
CAPS = {s: 100.0 * v for s, v in LIMITS['configs']['RE2'].items() if not s.startswith('_')}
CONT = list(CAPS)
WIN_NS = 100_000_000                       # 100 ms = chu ky CFS mac dinh
RAMP_GLOB = os.path.join(_P, 'data', 'raw', 'SS-*', '**', 'ramp_*', 'run*', 'simple_metrics.csv')


def _r2(x, y):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 30 or x.std() == 0 or y.std() == 0:
        return np.nan
    p = np.polyval(np.polyfit(x, y, 1), x); st = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - p) ** 2).sum() / st if st > 0 else np.nan


# ============================================================ S1
def part_signal():
    """Tai nguyen nao co tin hieu theo tai? Ba dieu kien de mot tai nguyen DU PHONG DUOC:
    (1) co tin hieu theo tai, (2) co tran doc duoc tu telemetry, (3) la thu lam vo SLO."""
    d, _ = MD.load('baseline')
    mets = ['cpu', 'mem', 'socket', 'diskio', 'error', 'latency-50', 'latency-99']
    print(f'\n=== S1. R2 cua <metric> ~ <cung service>_workload | {len(d)} dong do he binh thuong ===')
    print(f"{'service':11s} " + ' '.join(f'{m[:9]:>10s}' for m in mets))
    for s in MD.SERVICES:
        w = d[f'{s}_workload'].values
        print(f'{s:11s} ' + ' '.join(
            (f'{_r2(w, d[f"{s}_{m}"].values):10.3f}' if f'{s}_{m}' in d else f'{"-":>10s}') for m in mets))
    print('\n  -> CHI cpu co tin hieu. mem/socket/diskio/latency ~ 0.')
    print('  -> telemetry chuan chi co tran cho CPU (container-spec-cpu-quota); khong co cot')
    print('     tran cho mem/socket/disk. Vay chi CPU thoa ca (1) va (2).')


# ============================================================ S2
def part_control():
    """DOI CHUNG AM: bo tran di thi hien tuong vo p99 co mat khong?"""
    def load(pat, tag):
        out = []
        for p in sorted(glob.glob(pat, recursive=True)):
            x = pd.read_csv(p)
            if 'gt_feature' in x.columns and str(x['gt_feature'].iloc[0]) != 'base':
                continue
            if 'gt_step_warm' in x.columns:
                x = x[x.gt_step_warm == 0]
            x['__tag'] = tag
            out.append(x)
        return pd.concat(out, ignore_index=True)

    no_cap = load(os.path.join(_P, 'data/raw/SS-TRAIN/level_*/run*/simple_metrics.csv'), 'KHONG TRAN')
    cap = load(os.path.join(_P, 'data/raw/SS-*/**/ramp_base/run*/simple_metrics.csv'), 'TRAN RE2')
    both = pd.concat([no_cap, cap], ignore_index=True)
    both['bin'] = (both['front-end_workload'] // 25 * 25).astype('Int64')
    print(f'\n=== S2. DOI CHUNG AM | khong tran {len(no_cap)} dong | tran RE2 {len(cap)} dong ===')
    for c in ['front-end_latency-99', 'front-end_latency-50', 'front-end_socket', 'front-end_cpu']:
        if c not in both:
            continue
        t = both.groupby(['bin', '__tag'])[c].median().unstack('__tag').dropna()
        if t.shape[1] < 2:
            continue
        t = t[['KHONG TRAN', 'TRAN RE2']]
        t['ti le'] = t['TRAN RE2'] / t['KHONG TRAN'].replace(0, np.nan)
        print(f'\n-- {c} (trung vi theo bin tai gateway) --')
        print(t.round(4).to_string())
    print('\n  -> p99 khong tran PHANG suot dai tai; co tran len toi ~10x. CPU/request nhu nhau.')
    print('  -> p50 gan nhu khong doi: p50 la PHAN VI SAI de do hien tuong nay.')


# ============================================================ S3
def part_burst():
    """GIA THUYET DA BAC BO: do bung no (phan tan so request moi 100ms) khong du doan
    throttling lan vi pham. Giu lai de khong ai mat thoi gian lam lai.

    LUU Y ve pham vi bac bo: bien do duoc la PHAN TAN LUOT DEN (tu so dong log moi
    100ms). CFS throttle theo NHU CAU CPU trong chu ky (muc song song x thoi gian phuc
    vu), khong theo so luot den -- mot container 4 thread x 12ms CPU co the dot het
    quota 50ms tu MOT lượt. Nen ban 'bung no theo muc song song' VAN CHUA duoc kiem.
    """
    runs = {}
    for p in sorted(glob.glob(os.path.join(_P, 'data/raw/SS-*/**/ramp_*/run*/logs.csv'), recursive=True)):
        sm = os.path.join(os.path.dirname(p), 'simple_metrics.csv')
        if not os.path.exists(sm):
            continue
        try:
            f = str(pd.read_csv(sm, usecols=['gt_feature'], nrows=1)['gt_feature'].iloc[0])
        except Exception:
            continue
        runs.setdefault(f, p)
    rows = []
    for f, p in sorted(runs.items()):
        dirn = os.path.dirname(p)
        sm = pd.read_csv(os.path.join(dirn, 'simple_metrics.csv'))
        if 'gt_step_idx' not in sm.columns:
            continue
        steps = (sm[sm.get('gt_step_warm', 0) == 0].groupby('gt_step_idx')
                 .agg(t0=('time', 'min'), t1=('time', 'max'), viol=('gt_step_violated', 'max')))
        lg = pd.read_csv(p, usecols=['timestamp', 'container_name'])
        lg = lg[lg.container_name.isin(CONT)].dropna()
        lg['sec'] = (lg.timestamp // 1_000_000_000).astype('int64')
        lg['w'] = (lg.timestamp // WIN_NS).astype('int64')
        for idx, st in steps.iterrows():
            seg = lg[(lg.sec >= st.t0) & (lg.sec <= st.t1)]
            for s, g in seg.groupby('container_name'):
                cnt = g.groupby('w').size().reindex(range(int(g.w.min()), int(g.w.max()) + 1), fill_value=0)
                if len(cnt) < 50:
                    continue
                m = cnt.mean()
                rows.append(dict(feature=f, service=s, viol=bool(st.viol), rate=m,
                                 cv=cnt.std() / max(m, 1e-9), burst_p99=cnt.quantile(.99) / max(m, 1e-9)))
    B = pd.DataFrame(rows)
    from sklearn.metrics import roc_auc_score
    print(f'\n=== S3. GIA THUYET DO BUNG NO -- DA BAC BO | {len(B)} (tinh nang x container x bac) ===')
    for c in ('rate', 'cv', 'burst_p99'):
        g = B[[c, 'viol']].dropna()
        print(f'  AUC({c:10s} -> vi pham) = {roc_auc_score(g.viol, g[c]):.3f}')
    print('  -> cv 0.50 / burst_p99 0.52 = bang doan ngau nhien. Bac bo.')


# ============================================================ S4
def _collect():
    rows = []
    for p in sorted(glob.glob(RAMP_GLOB, recursive=True)):
        sm = pd.read_csv(p)
        if 'gt_step_idx' not in sm.columns:
            continue
        try:
            mt = pd.read_csv(p.replace('simple_metrics.csv', 'metrics.csv'))
        except Exception:
            continue
        if 'time' not in mt.columns:
            continue
        T = {}
        for s in CONT:
            a = f'{s}_container-cpu-cfs-throttled-periods-total'
            b = f'{s}_container-cpu-cfs-periods-total'
            if a in mt and b in mt:
                T[f'T_{s}'] = (mt[a].diff() / mt[b].diff().replace(0, np.nan)).clip(0, 1)
        if not T:
            continue
        J = pd.DataFrame(T); J['time'] = mt['time'].values
        d = sm.merge(J, on='time', how='left')
        for idx, g in d.groupby('gt_step_idx'):
            gs = g[g.get('gt_step_warm', 0) == 0]
            if len(gs) < 3:
                continue
            r = dict(run=p, feature=str(sm['gt_feature'].iloc[0]), step=int(idx),
                     rps=float(gs['gt_target_rps'].median()) if 'gt_target_rps' in gs else np.nan,
                     pct=float(gs['gt_feature_pct'].median()) if 'gt_feature_pct' in gs else 0.0,
                     L=float(gs['front-end_workload'].median()),
                     viol=int(g['gt_step_violated'].max() > 0))
            for s in CONT:
                if f'{s}_cpu' in gs:
                    r[f'u_{s}'] = float(gs[f'{s}_cpu'].median()) / CAPS[s]
                if f'T_{s}' in gs:
                    r[f'T_{s}'] = float(gs[f'T_{s}'].median())
            rows.append(r)
    R = pd.DataFrame(rows)
    R['u_max'] = R[[c for c in R if c.startswith('u_')]].max(axis=1)
    R['T_max'] = R[[c for c in R if c.startswith('T_')]].max(axis=1)
    return R


def part_auc():
    """So sanh suc phan biet vi pham. Ba diem PHUONG PHAP bat buoc:
      1. don vi phan tich = MOT QUYET DINH = (tinh nang, muc tai). Nhieu run cua cung
         cau hinh la do lap lai, phai gop -- neu khong, AUC bi thoi len.
      2. isotonic u->throttle phai fit NGOAI FOLD (GroupKFold theo tinh nang). Fit va
         danh gia cung du lieu tung cho AUC 0.912 lac quan.
      3. so sanh hai AUC tren CUNG mau phai bootstrap tren HIEU SO, khong phai xem hai
         khoang tin cay co chong nhau khong.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    R = _collect()

    base, _ = MD.load('baseline')
    wg = base['front-end_workload'].to_numpy(float)
    M = {}
    for s in CONT:
        if f'{s}_workload' not in base or f'{s}_cpu' not in base:
            continue
        w = base[f'{s}_workload'].to_numpy(float); c = base[f'{s}_cpu'].to_numpy(float)
        ok = np.isfinite(wg) & np.isfinite(w) & np.isfinite(c)
        if ok.sum() < 50:
            continue
        rho = 1.0 if s == 'front-end' else float(wg[ok] @ w[ok] / (wg[ok] @ wg[ok]))
        lr = LinearRegression(positive=True).fit(np.ascontiguousarray(w[ok].reshape(-1, 1)),
                                                 np.ascontiguousarray(c[ok]))
        M[s] = (rho, float(lr.intercept_), float(lr.coef_[0]))
    K = {}
    for f in ['k_measured.json', 'k_measured_v2.json', 'k_measured_v3.json',
              'k_measured_prosp.json', 'k_measured_v3new.json', 'k_measured_v4new.json']:
        fp = os.path.join(_P, 'data', 'processed', 'frozen', f)
        if os.path.exists(fp):
            for k, v in json.load(open(fp))['features'].items():
                if v.get('measured_per_use'):
                    K[k] = v['measured_per_use']

    def u_hat(r):
        arch = FP.FEATURE_ARCHETYPE.get(r.feature, r.feature)
        chain = set(FP.SOCKSHOP_CALL_CHAINS[arch]['services']) if arch in FP.SOCKSHOP_CALL_CHAINS else set()
        L_bg = r.L / (1 + r.pct / 100.0); lam = r.L - L_bg
        out = {}
        for s, (rho, a, b) in M.items():
            k = 1.0 if s == 'front-end' else (float(K.get(r.feature, {}).get(s, 0.0)) if s in chain else 0.0)
            out[s] = (a + b * (rho * L_bg + k * lam)) / CAPS[s]
        return max(out.values()) if out else np.nan
    R['u_hat'] = [u_hat(r) for r in R.itertuples()]

    sub = R.dropna(subset=['u_max', 'T_max']).copy()
    for tr, te in GroupKFold(n_splits=5).split(sub, groups=sub.feature):
        for s in CONT:
            cu, ct = f'u_{s}', f'T_{s}'
            if cu not in sub or ct not in sub:
                continue
            a = sub.iloc[tr].dropna(subset=[cu, ct])
            if len(a) < 30 or a[ct].std() == 0:
                continue
            iso = IsotonicRegression(out_of_bounds='clip').fit(a[cu], a[ct])
            b = sub.iloc[te]
            sub.loc[b.index, f'Th_{s}'] = iso.predict(b[cu].fillna(0))
    th = [c for c in sub if c.startswith('Th_')]
    sub['T_hat'] = sub[th].max(axis=1) if th else np.nan
    R = R.merge(sub[['run', 'step', 'T_hat']], on=['run', 'step'], how='left')

    rng = np.random.default_rng(7)

    def diff_ci(y, x1, x2, n=4000):
        y, x1, x2 = map(np.asarray, (y, x1, x2))
        m = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
        y, x1, x2 = y[m], x1[m], x2[m]
        d = roc_auc_score(y, x1) - roc_auc_score(y, x2)
        bs = []
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) > 1:
                bs.append(roc_auc_score(y[i], x1[i]) - roc_auc_score(y[i], x2[i]))
        bs = np.array(bs)
        return d, np.percentile(bs, 2.5), np.percentile(bs, 97.5), float((bs <= 0).mean()), len(y)

    print('\n=== S4. SUC PHAN BIET VI PHAM -- 6 dac ta + bootstrap tren HIEU SO ===')
    for feat_only in (True, False):
        D0 = R[R.feature != 'base'] if feat_only else R
        for unit in ('quyet dinh', 'tho'):
            if unit == 'quyet dinh':
                A = D0.groupby(['feature', 'rps']).agg(
                    viol=('viol', 'max'), vmaj=('viol', 'mean'), u_max=('u_max', 'median'),
                    u_hat=('u_hat', 'median'), T_max=('T_max', 'median'),
                    T_hat=('T_hat', 'median')).reset_index()
                A['vmaj'] = (A.vmaj > 0.5).astype(int)
                labs = [('viol', 'bat ky run'), ('vmaj', 'da so run')]
            else:
                A = D0.copy(); labs = [('viol', 'bat ky run')]
            for lab, ln in labs:
                tag = 'chi tinh nang' if feat_only else 'gom base'
                print(f'\n-- {unit} | nhan {ln} | {tag} --')
                for c, nm in [('u_max', 'u DO DUOC'), ('u_hat', 'u DU DOAN'),
                              ('T_max', 'throttle DO DUOC'), ('T_hat', 'throttle SUY tu u')]:
                    x = A[[c, lab]].dropna()
                    if x[lab].nunique() > 1:
                        print(f'   AUC {nm:22s} {roc_auc_score(x[lab], x[c]):.3f}   n={len(x)}')
                for a, b, nm in [('T_max', 'u_max', 'throttle DO - u DO'),
                                 ('T_max', 'T_hat', 'throttle DO - throttle SUY tu u'),
                                 ('T_max', 'u_hat', 'throttle DO - u DU DOAN')]:
                    d, lo, hi, p, n = diff_ci(A[lab], A[a], A[b])
                    print(f'   HIEU {nm:32s} {d:+.3f} CI[{lo:+.3f},{hi:+.3f}] P(d<=0)={p:.4f}'
                          f'  -> {"CO Y NGHIA" if lo > 0 else "khong"}')


PARTS = dict(signal=part_signal, control=part_control, burst=part_burst, auc=part_auc)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['all'] + list(PARTS))
    a = ap.parse_args()
    for name, fn in (PARTS.items() if a.part == 'all' else [(a.part, PARTS[a.part])]):
        fn()
