# -*- coding: utf-8 -*-
"""
LOP LAM SACH DU LIEU TU DO (Sock Shop) -- MOT cua duy nhat vao data/raw/SS-*
===========================================================================
Vi sao can lop nay (moi muc duoi day la mot khiem khuyet DA DO, khong phai lo xa):

  1. HAI kieu long thu muc. `SS-TRAIN`, `SS-LIMITS*`, `SS-VERIFY` long 2 cap
     (`<scenario>/<run>/simple_metrics.csv`); `SS-PROSP2` long 3 cap
     (`<batch>/<scenario>/<run>/...`). Ca `data_processor.load_multi_service_data()`
     LAN `papers/p1_du_phong/experiments/collect/data_contract_check.py` (glob `*/*/simple_metrics.csv`)
     deu MU voi kieu 3 cap: goi loader tren 'data/raw/SS-PROSP2' tra ve None.

  2. Vai (role) la thuoc tinh CUA HANG, khong phai cua thu muc. `SS-LIMITS` chua
     836 dong base + 1970 dong co tinh nang trong cung mot goc. Tro thu muc de
     lay du lieu train la duong NHIEM BAN im lang: `data/raw/SS-PROSP2/SS-PROSP2`
     tra ve 4967 dong, trong do co ramp_login_x1, ramp_orderfull_x1... tuc chinh
     tap dap an cua RQ5.

  3. Hang warm-up. ~36-38% hang moi goc co `gt_step_warm == 1`. Hop dong du lieu
     (papers/p1_du_phong/docs/DATA_FRAMEWORK.md muc 1, dinh nghia nhan) tinh tren 60% CUOI moi bac.

  4. Hang sau vi pham SLO (`gt_step_violated == 1`): 12%-100% tuy goc. `SS-LIMITS-C1`
     co ti le 1.000 -- MOI hang deu sau vi pham.

  5. Cau hinh tran KHONG HOP LE. `deploy/sockshop/limits.json:_status` danh dau ca
     C1 va C2 `_INVALID`: han ngach nho (0.08-0.15 core) day vao che do CFS
     throttling, SLO vo o u ~20-45% thay vi ~88%, "ngoai pham vi bo du doan hien tai".

  6. Goc KHONG NHAN. `SS-CALIBRATION` khong co cot `gt_*` nao, lai co cot la
     `load_level` (data_contract_check --role train: 1 FAIL) -> khong xac dinh duoc
     tran hay tinh nang cua tung hang.

  7. Goc RONG. `SS-LOADSWEEP` khong co run nao, chi con mot tep collector_*.csv roi.

  8. DI BIET GIUA CAC PHIEN THU. Do tren 10 phien `ramp_base` cua SS-PROSP2: he so
     goc cpu~workload cua `orders` lech 2.3x (0.649-1.474), `shipping` 8x
     (0.081-0.677), trong khi `front-end`/`catalogue`/`user` chi lech +-4%. Loc
     warm/violated KHONG cuu duoc (orders R2 0.376 -> 0.425). Nen moi hang mang
     cot `__session` de tang tren co the xu ly (hoac chi bao cao) phan lech nay.

  9. Tap KHOA. `data_contract_check.py:38` khai LOCKED_FEATURES = (track, review):
     khong duoc dung khi chinh mo hinh.

NGUYEN TAC: du lieu tho la BAT BIEN. Module nay khong ghi, khong sua, khong xoa gi
trong data/raw. Moi phep loai bo deu tra ve kem LY DO co xuat xu (tep:dong) trong
`report`, de mot nguoi doc bao co the kiem lai tung quyet dinh.

Dung:
    import measured_data as MD
    df, rep = MD.load('train')      # hop dong role=train (chi SS-TRAIN qua duoc)
    df, rep = MD.load('baseline')   # moi run base duoi tran HOP LE, chi hang on dinh
    df, rep = MD.load('eval')       # run CO tinh nang (tap danh gia), bo tap KHOA
    MD.audit()                      # bang mot dong moi run + verdict, khong nap cot metric
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.abspath(os.path.join(_HERE, '..', '..', 'data', 'raw'))

SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
METRICS = ['workload', 'cpu', 'mem', 'socket', 'diskio', 'error',
           'latency-50', 'latency-90', 'latency-95', 'latency-99']
ENV_COLS = ['time', 'vm_cpu_util', 'vm_mem_avail_mb']

# --- luat loai bo o muc RUN, moi luat kem xuat xu de kiem lai duoc -------------
INVALID_LIMITS = {'C1', 'C2'}
UNLABELLED_ROOTS = {'SS-CALIBRATION'}
EMPTY_ROOTS = {'SS-LOADSWEEP'}
LOCKED_FEATURES = {'track', 'review'}          # data_contract_check.py:38

WHY = {
    'invalid_limits': 'cau hinh tran _INVALID (deploy/sockshop/limits.json:_status): '
                      'han ngach nho -> CFS throttling, SLO vo o u~20-45% thay vi ~88%, ngoai pham vi mo hinh',
    'unlabelled':     'khong co cot gt_* nao + co cot la load_level '
                      '(papers/p1_du_phong/experiments/collect/data_contract_check.py --role train: 1 FAIL) -> khong biet tran/tinh nang',
    'empty_root':     'goc khong co run nao (chi con mot tep collector_*.csv roi)',
    'locked':         'thuoc tap KHOA LOCKED_FEATURES (data_contract_check.py:38), khong dung khi chinh mo hinh',
    'has_feature':    'run co luu luong tinh nang -> khong phai du lieu hoc co che (papers/p1_du_phong/docs/DATA_FRAMEWORK.md muc 5 pha A)',
    'no_feature':     'run base -> khong thuoc tap danh gia tinh nang',
    'under_ceiling':  'chay duoi tran CPU: hop dong role=train yeu cau limits_cfg == none '
                      '(data_contract_check.py:96-103)',
    'saturated':      'run chua bac vi pham SLO -> khong con la vung chua bao hoa (data_contract_check.py:104)',
}


# ============================================================ kham pha run
def _runs_under(root_path: str) -> List[str]:
    """Moi thu muc chua simple_metrics.csv, BAT KE long 2 hay 3 cap.

    Dung rglob thay vi glob co do sau co dinh chinh la de khong lap lai diem mu
    cua data_processor/data_contract_check voi SS-PROSP2 (long 3 cap).
    """
    return sorted(os.path.dirname(p) for p in
                  glob.glob(os.path.join(root_path, '**', 'simple_metrics.csv'), recursive=True))


def _meta_of(run_dir: str, root: str) -> Dict:
    """Doc sieu du lieu cua MOT run ma khong giu cot metric (nhe, dung cho audit())."""
    p = os.path.join(run_dir, 'simple_metrics.csv')
    rel = os.path.relpath(run_dir, os.path.join(RAW_DIR, root)).replace(os.sep, '/')
    parts = rel.split('/')
    batch = parts[0] if len(parts) >= 3 else ''          # long 3 cap -> parts = [batch, scenario, run]
    scenario = parts[-2] if len(parts) >= 2 else rel
    d = pd.read_csv(p)
    def fr(col):
        return float(d[col].mean()) if col in d.columns else np.nan
    def one(col, default):
        if col not in d.columns:
            return default
        v = d[col].dropna().unique()
        return str(v[0]) if len(v) == 1 else ('|'.join(map(str, sorted(v))) if len(v) else default)
    return dict(
        root=root, batch=batch, scenario=scenario, run=parts[-1], rel=rel, path=run_dir,
        session=f'{root}/{batch}' if batch else root,
        n=len(d),
        feature=one('gt_feature', 'base'),
        limits=one('gt_limits_cfg', 'none'),
        warm_frac=fr('gt_step_warm'), viol_frac=fr('gt_step_violated'),
        vm_max=float(d['vm_cpu_util'].max()) if 'vm_cpu_util' in d else np.nan,
        t_gap_max=float(d['time'].diff().max()) if 'time' in d else np.nan,
        w_min=float(d['front-end_workload'].min()) if 'front-end_workload' in d else np.nan,
        w_max=float(d['front-end_workload'].max()) if 'front-end_workload' in d else np.nan,
        has_gt=any(c.startswith('gt_') for c in d.columns),
        has_steps=os.path.exists(os.path.join(run_dir, 'steps.json')),
    )


def audit(roots: Optional[List[str]] = None) -> pd.DataFrame:
    """Mot dong moi run tu do duoc, kem verdict cho ca ba purpose. KHONG nap cot metric."""
    roots = roots if roots is not None else sorted(
        d for d in os.listdir(RAW_DIR) if d.startswith('SS-') and os.path.isdir(os.path.join(RAW_DIR, d)))
    rows = []
    for root in roots:
        rp = os.path.join(RAW_DIR, root)
        found = _runs_under(rp)
        if not found:
            rows.append(dict(root=root, batch='', scenario='', run='', rel='', path=rp, session=root,
                             n=0, feature='-', limits='-', warm_frac=np.nan, viol_frac=np.nan,
                             vm_max=np.nan, t_gap_max=np.nan, w_min=np.nan, w_max=np.nan,
                             has_gt=False, has_steps=False))
            continue
        rows.extend(_meta_of(r, root) for r in found)
    df = pd.DataFrame(rows)
    for purpose in ('train', 'baseline', 'eval'):
        df[purpose] = [_run_verdict(r, purpose) for _, r in df.iterrows()]
    return df


def _run_verdict(m, purpose: str) -> str:
    """'' neu run duoc dung cho purpose nay, nguoc lai tra ve khoa ly do trong WHY."""
    if m['root'] in EMPTY_ROOTS or m['n'] == 0:
        return 'empty_root'
    if m['root'] in UNLABELLED_ROOTS or not m['has_gt']:
        return 'unlabelled'
    if str(m['limits']) in INVALID_LIMITS:
        return 'invalid_limits'
    is_base = str(m['feature']) == 'base'
    if purpose in ('train', 'baseline'):
        if not is_base:
            return 'has_feature'
        if purpose == 'train':
            if str(m['limits']) != 'none':
                return 'under_ceiling'
            if (m['viol_frac'] or 0) > 0:
                return 'saturated'
    else:                                   # eval
        if is_base:
            return 'no_feature'
        if str(m['feature']) in LOCKED_FEATURES:
            return 'locked'
    return ''


# ============================================================ nap du lieu
def load(purpose: str = 'baseline', roots: Optional[List[str]] = None,
         steady_only: bool = True, services: Optional[List[str]] = None,
         keep_locked: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tra ve (df, report).

    purpose:
      'train'    -- hop dong role=train (papers/p1_du_phong/docs/DATA_FRAMEWORK.md muc 4,
                    data_contract_check.py:96-103): base, KHONG tran, KHONG bac vi pham.
                    Chi SS-TRAIN qua duoc -> tap nho nhung dung hop dong tuyet doi.
      'baseline' -- moi run base duoi tran HOP LE (none hoac RE2). Rong hon 'train';
                    dung khi muon co ca vung gan tran (front-end cham u~0.94 o ramp RE2),
                    va PHAI bao cao la da noi long hop dong.
      'eval'     -- run CO tinh nang = tap danh gia. Tap KHOA bi loai tru khi
                    keep_locked=True (chi dung cho lan do cuoi, khong dung khi chinh mo hinh).

    steady_only: bo hang warm-up va hang sau vi pham SLO (gt_step_warm / gt_step_violated).
                 Voi purpose='eval' van nen de True: du doan so voi trang thai on dinh cua bac.

    report: mot dong moi run kem cot `used` va `why` -- de kiem lai tung quyet dinh loai bo.
    """
    if purpose not in ('train', 'baseline', 'eval'):
        raise ValueError(f"purpose phai la train|baseline|eval, nhan '{purpose}'")
    services = services if services is not None else SERVICES
    rep = audit(roots)
    rep = rep.rename(columns={purpose: 'why'})
    if purpose == 'eval' and keep_locked:
        rep.loc[rep['why'] == 'locked', 'why'] = ''
    rep['used'] = rep['why'] == ''
    rep['why_text'] = rep['why'].map(lambda k: WHY.get(k, ''))

    cols = [f'{s}_{m}' for s in services for m in METRICS]
    frames = []
    for _, r in rep[rep['used']].iterrows():
        d = pd.read_csv(os.path.join(r['path'], 'simple_metrics.csv'))
        if steady_only:
            if 'gt_step_warm' in d.columns:
                d = d[d['gt_step_warm'] == 0]
            if 'gt_step_violated' in d.columns:
                d = d[d['gt_step_violated'] == 0]
        keep = [c for c in ENV_COLS + cols if c in d.columns]
        out = d[keep].copy()
        for k in ('root', 'batch', 'scenario', 'run', 'session', 'feature', 'limits'):
            out[f'__{k}'] = r[k]
        out['__target_rps'] = d['gt_target_rps'].values if 'gt_target_rps' in d.columns else np.nan
        frames.append(out)

    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=ENV_COLS + cols))
    df.attrs['purpose'] = purpose
    df.attrs['steady_only'] = steady_only
    return df, rep[['root', 'batch', 'scenario', 'run', 'n', 'feature', 'limits',
                    'warm_frac', 'viol_frac', 'w_min', 'w_max', 'used', 'why', 'why_text']]


def summarize(rep: pd.DataFrame) -> pd.DataFrame:
    """Gop report thanh bang: moi (goc, ly do) mot dong, kem so run va so dong."""
    g = (rep.assign(why=rep['why'].replace('', '(DUNG)'))
            .groupby(['root', 'why'], as_index=False)
            .agg(runs=('n', 'size'), dong=('n', 'sum')))
    return g.sort_values(['root', 'why']).reset_index(drop=True)
