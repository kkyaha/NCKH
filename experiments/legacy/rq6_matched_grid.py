# -*- coding: utf-8 -*-
"""RQ6 — LUOI KHOP TONG QUAT: cam bao nhieu bo xep hang cung duoc, ba he.

Vi sao can script nay. `rq6_baro_matched_comparison.py` chi chay MOT he
(Sock Shop) voi HAI bo xep hang, va mot trong hai la ban CAI LAI luat cua BARO
chu khong phai code cua tac gia. Phan bien manh nhat voi phat hien khop tang la:
"co the day la tao tac cua cach cac tac gia cai lai BARO, va mot phuong phap
khong noi duoc gi ve ca linh vuc."

Script nay bo cai chan do: no tach phan DOC DU LIEU + CHAM DIEM + TINH CHI SO ra
khoi phan XEP HANG, nen them mot baseline = viet DUNG MOT ham adapter. Muc tieu
la chay 4-6 baseline da xuat ban bang CODE CUA CHINH TAC GIA (repo RCAEval),
trong cung mot luoi, tren ca ba he.

LUOI (moi to hop duoc cham tren CUNG run, CUNG dap an, CUNG cua so truoc/sau):

  tang     : chi_cpu | chi_latency | chi_workload | moi_cot
  ung vien : fault5 (dung 5 service bi tiem) | app (moi service ung dung) | all (moi service co trong cot)
  bo xep hang : moi ham trong REGISTRY

HOP DONG ADAPTER — them baseline chi can mot ham dang nay:

    def rank_<ten>(normal: pd.DataFrame, anomal: pd.DataFrame, cols: list) -> list[(cot, diem)]

  normal/anomal : hai cua so truoc/sau `inject_time`, da cat san
  cols          : CHI nhung cot thuoc tang + tap ung vien dang xet
  tra ve        : danh sach (ten_cot, diem) — diem CAO hon = nghi ngo hon.
                  Khong cho tra ve ten service: driver tu gop cot -> service.

  Dang ky bang:  REGISTRY['TEN'] = rank_<ten>

BASELINE CUA RCAEval. Neu da clone repo (xem cuoi file) va dat duoc bien moi
truong RCAEVAL_PATH, script tu tim va boc cac ham e2e cua ho. HOP DONG cua ho
KHONG duoc doan ngam: `_boc_rcaeval` thu goi va BAO LOI RO RANG neu chu ky khac,
kem goi y, thay vi tra ve rac.

Chay:
  python experiments/legacy/rq6_matched_grid.py                 # chi cac bo noi bo
  python experiments/legacy/rq6_matched_grid.py --kiem-tai-lap  # doi chieu voi CSV cu
  RCAEVAL_PATH=~/RCAEval python experiments/legacy/rq6_matched_grid.py   # + baseline tac gia

Ra: data/processed/scm_results/rq6_matched_grid.csv
    data/processed/scm_results/rq6_matched_grid_summary.csv
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RES = os.path.join(_P, 'data', 'processed', 'scm_results')
ENV = {'time', 'imte', 'vm_cpu_util', 'vm_mem_avail_mb'}
RESOURCE = ['cpu', 'mem', 'disk', 'socket']
KS = (1, 3, 5)

# ============================================================== BO XEP HANG


def rank_baro(normal, anomal, cols):
    """BARO (FSE'24): RobustScaler median/IQR tren cua so truoc, roi max|z| tren cua so sau.

    Cai lai theo luat trong RCAEval `e2e/baro.py`. LUU Y: day la BAN CAI LAI, khong
    phai code cua tac gia -- do la chinh dieu script nay ton tai de khac phuc.
    """
    out = []
    for c in cols:
        n, a = normal[c].to_numpy(float), anomal[c].to_numpy(float)
        n, a = n[np.isfinite(n)], a[np.isfinite(a)]
        if len(n) < 5 or len(a) < 5 or np.all(n == n[0]):
            continue
        z = RobustScaler().fit(n.reshape(-1, 1)).transform(a.reshape(-1, 1))
        out.append((c, float(np.max(np.abs(z)))))
    return out


def rank_meanshift(normal, anomal, cols):
    """|trung binh sau - trung binh truoc| / std truoc."""
    out = []
    for c in cols:
        n, a = normal[c].dropna(), anomal[c].dropna()
        if len(n) < 5 or len(a) < 5 or n.std() == 0:
            continue
        out.append((c, abs(a.mean() - n.mean()) / n.std()))
    return out


def rank_nothing(normal, anomal, cols):
    """DOI CHUNG AM: khong doc telemetry, xep theo ten cot.

    Bat buoc phai co. arXiv 2609.27069 chi ra rang mot bo xep hang KHONG doc gi
    van dat top-5 rat cao tren RE2 chi nho prior cua tap loi. Arm nay do dung
    muc san do bang thuc nghiem, thay vi tinh bang cong thuc.
    """
    return [(c, -i) for i, c in enumerate(sorted(cols))]


REGISTRY = {'BARO': rank_baro, 'MEANSHIFT': rank_meanshift, 'KHONG-DOC': rank_nothing}


def _boc_rcaeval(duong_dan):
    """Boc cac ham e2e cua RCAEval thanh adapter, BAO LOI RO neu chu ky khac."""
    duong_dan = os.path.expanduser(duong_dan)
    if not os.path.isdir(duong_dan):
        print(f'[RCAEval] khong thay {duong_dan} -- bo qua cac arm baseline tac gia')
        return
    sys.path.insert(0, duong_dan)
    try:
        import RCAEval.e2e as e2e
    except Exception as e:
        print(f'[RCAEval] import that bai: {type(e).__name__}: {e}')
        print('          -> kiem lai: pip install -e . trong repo RCAEval')
        return
    muon = ['baro', 'circa', 'rcd', 'micro_cause', 'e_diagnosis', 'nsigma']
    for ten in muon:
        fn = getattr(e2e, ten, None)
        if fn is None:
            print(f'[RCAEval] khong co e2e.{ten} -- bo qua')
            continue

        def adapter(normal, anomal, cols, _fn=fn, _ten=ten):
            d = pd.concat([normal[cols], anomal[cols]], ignore_index=True)
            t = len(normal)
            try:
                kq = _fn(d, t)
            except Exception as e:
                raise RuntimeError(
                    f'e2e.{_ten} khong nhan chu ky (data, inject_time): '
                    f'{type(e).__name__}: {e}\n'
                    f'  -> mo README cua RCAEval, xem chu ky THAT, roi sua adapter nay. '
                    f'KHONG doan: mot adapter sai se tra ve thu hang rac ma khong bao loi.')
            hang = kq.get('ranks') if isinstance(kq, dict) else kq
            if not hang:
                raise RuntimeError(f'e2e.{_ten} tra ve rong / khong co khoa "ranks": {type(kq)}')
            return [(str(h), -i) for i, h in enumerate(hang)]

        REGISTRY[f'RCAEval:{ten}'] = adapter
        print(f'[RCAEval] da dang ky arm RCAEval:{ten}')


# ============================================================== DOC DU LIEU


def cac_run():
    """(he, service bi tiem, loai loi, lan lap, metrics, thoi diem tiem). Dap an tu TEN THU MUC."""
    ss = os.path.join(_P, 'data', 'raw', 'RE2-SS')
    if os.path.isdir(ss):
        for scen in sorted(os.listdir(ss)):
            sp = os.path.join(ss, scen)
            if not os.path.isdir(sp):
                continue
            inj, _, ft = scen.rpartition('_')
            for r in sorted(os.listdir(sp)):
                mp, ip = os.path.join(sp, r, 'simple_metrics.csv'), os.path.join(sp, r, 'inject_time.txt')
                if os.path.exists(mp) and os.path.exists(ip):
                    try:
                        yield 'SockShop', inj, ft, r, pd.read_csv(mp), int(open(ip).read().strip())
                    except Exception:
                        continue
    for pref, root, name in (('re2ob_', 'RE2-OB', 'OnlineBoutique'),
                             ('re2tt_', 'trainticket', 'TrainTicket')):
        for sp in sorted(glob.glob(os.path.join(_P, 'data', 'raw', root, pref + '*'))):
            base = os.path.basename(sp)[len(pref):]
            run = base.rsplit('_', 1)[-1]
            inj, _, ft = base.rsplit('_', 1)[0].rpartition('_')
            mp, ip = os.path.join(sp, 'metrics.parquet'), os.path.join(sp, 'inject_time.txt')
            if os.path.exists(mp) and os.path.exists(ip):
                try:
                    yield name, inj, ft, run, pd.read_parquet(mp), int(open(ip).read().strip())
                except Exception:
                    continue


# ============================================================== CHI SO


def chi_so(ranks):
    r = pd.to_numeric(ranks, errors='coerce').dropna()
    if len(r) == 0:
        return {}
    o = {'n': len(r)}
    for k in KS:
        o[f'AC@{k}'] = 100 * (r <= k).mean()
    o['Avg@5'] = 100 * np.mean([(r <= k).mean() for k in range(1, 6)])
    o['MRR'] = (1.0 / r).mean()
    return o


def san(n_cand):
    k = float(pd.to_numeric(n_cand, errors='coerce').dropna().median())
    o = {'k': k}
    for j in KS:
        o[f'san AC@{j}'] = 100 * min(j, k) / k
    o['san Avg@5'] = 100 * np.mean([min(j, k) / k for j in range(1, 6)])
    o['san MRR'] = sum(1.0 / i for i in range(1, int(k) + 1)) / k
    return o


# ============================================================== LUOI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kiem-tai-lap', action='store_true',
                    help='doi chieu voi rq6_baro_matched_comparison.csv (Sock Shop)')
    ap.add_argument('--rcaeval', default=os.environ.get('RCAEVAL_PATH'),
                    help='duong dan repo RCAEval da clone (hoac bien moi truong RCAEVAL_PATH)')
    a = ap.parse_args()
    if a.rcaeval:
        _boc_rcaeval(a.rcaeval)

    # vong 1: gom tap service BI TIEM cua tung he (tu ten thu muc)
    du_lieu = list(cac_run())
    tap_tiem = {}
    for he, inj, *_ in du_lieu:
        tap_tiem.setdefault(he, set()).add(inj)
    print(f'\n{len(du_lieu)} run | tap bi tiem:')
    for h, s_ in sorted(tap_tiem.items()):
        print(f'  {h:15s} {len(s_)} service')

    rows = []
    for he, inj, ft, run, d, t in du_lieu:
        tc = 'imte' if 'imte' in d.columns else 'time'
        normal, anomal = d[d[tc] < t], d[d[tc] >= t]
        if len(normal) < 20 or len(anomal) < 20:
            continue
        cot_metric = [c for c in d.columns if c not in ENV and '_' in c]
        svc_tat_ca = {c.rsplit('_', 1)[0] for c in cot_metric}
        svc_app = {c[:-len('_workload')] for c in cot_metric if c.endswith('_workload')}
        for cand_lab, cand in (('fault5', tap_tiem[he]), ('app', svc_app), ('all', svc_tat_ca)):
            if inj not in cand:
                continue
            tangs = {
                'chi_cpu': [f'{s}_cpu' for s in cand if f'{s}_cpu' in d.columns],
                'chi_latency': [f'{s}_latency-50' for s in cand if f'{s}_latency-50' in d.columns],
                'chi_workload': [f'{s}_workload' for s in cand if f'{s}_workload' in d.columns],
                'moi_cot': [c for c in cot_metric if c.rsplit('_', 1)[0] in cand],
            }
            for tang_lab, sel in tangs.items():
                if len(sel) < 2:
                    continue
                for bo_lab, fn in REGISTRY.items():
                    r = fn(normal, anomal, sel)
                    if not r:
                        continue
                    r = sorted(r, key=lambda x: x[1], reverse=True)
                    hang_svc = []
                    for c, _ in r:
                        s_ = c.rsplit('_', 1)[0] if c not in svc_tat_ca else c
                        if s_ not in hang_svc:
                            hang_svc.append(s_)
                    if inj not in hang_svc:
                        continue
                    rows.append(dict(he=he, injected=inj, fault=ft, run=run,
                                     ung_vien=cand_lab, tang=tang_lab, bo_xep_hang=bo_lab,
                                     n_cand=len(hang_svc), rank=hang_svc.index(inj) + 1))
    D = pd.DataFrame(rows)
    D['nhom'] = np.where(D.fault.isin(RESOURCE), 'tai nguyen', 'mang')
    D.to_csv(os.path.join(RES, 'rq6_matched_grid.csv'), index=False)

    S = []
    for (he, nhom, cv, tang, bo), g in D.groupby(['he', 'nhom', 'ung_vien', 'tang', 'bo_xep_hang']):
        S.append(dict(he=he, nhom_loi=nhom, ung_vien=cv, tang=tang, bo_xep_hang=bo,
                      **chi_so(g['rank']), **san(g.n_cand)))
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(RES, 'rq6_matched_grid_summary.csv'), index=False)

    with pd.option_context('display.width', 240, 'display.max_rows', 400,
                           'display.float_format', lambda v: f'{v:,.2f}'):
        for nhom in ['tai nguyen', 'mang']:
            print(f'\n{"=" * 118}\n  LOI {nhom.upper()} — AC@1 theo (tang x ung vien x bo xep hang)\n{"=" * 118}')
            sub = S[S.nhom_loi == nhom]
            print(sub.pivot_table(index=['he', 'ung_vien', 'tang'],
                                  columns='bo_xep_hang', values='AC@1').to_string())

    print(f'\n{"=" * 118}\n  DOI CHUNG AM: bo KHONG-DOC-TELEMETRY dat bao nhieu?\n{"=" * 118}')
    kd = S[S.bo_xep_hang == 'KHONG-DOC']
    print(kd.groupby(['he', 'ung_vien'])[['AC@1', 'AC@3', 'AC@5', 'MRR', 'san AC@1', 'san MRR']]
          .mean().round(2).to_string())

    if a.kiem_tai_lap:
        print(f'\n{"=" * 118}\n  KIEM TAI LAP — doi chieu voi rq6_baro_matched_comparison.csv\n{"=" * 118}')
        old = pd.read_csv(os.path.join(RES, 'rq6_baro_matched_comparison.csv'))
        old = old[old.fault.isin(RESOURCE)]
        o = (old.groupby(['cand', 'layer', 'stat'])['rank']
             .apply(lambda s: 100 * (s == 1).mean()).rename('AC@1 cu').reset_index())
        o['ung_vien'] = o.cand.map({'app7': 'app', 'all': 'all'})
        n = S[(S.he == 'SockShop') & (S.nhom_loi == 'tai nguyen')]
        m = o.merge(n, left_on=['ung_vien', 'layer', 'stat'],
                    right_on=['ung_vien', 'tang', 'bo_xep_hang'], suffixes=('', '_moi'))
        m['lech'] = (m['AC@1 cu'] - m['AC@1']).abs()
        print(m[['ung_vien', 'tang', 'bo_xep_hang', 'AC@1 cu', 'AC@1', 'lech']].round(3).to_string(index=False))
        print(f'\n  lech lon nhat: {m.lech.max():.3f} diem  '
              f'{"-> TAI LAP KHOP" if m.lech.max() < 0.01 else "-> LECH, phai tim nguyen nhan"}')

    print(f'\n-> {RES}/rq6_matched_grid.csv  ({len(D)} dong)')
    print(f'-> {RES}/rq6_matched_grid_summary.csv  ({len(S)} dong)')
    print('\nDE THEM BASELINE CUA TAC GIA:')
    print('  git clone https://github.com/phamquiluan/RCAEval && cd RCAEval && pip install -e .')
    print('  RCAEVAL_PATH=<duong dan> python experiments/legacy/rq6_matched_grid.py')


if __name__ == '__main__':
    main()
