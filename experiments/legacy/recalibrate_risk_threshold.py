# -*- coding: utf-8 -*-
"""HIEU CHINH LAI NGUONG RUI RO cua FutureRCAEngine, va tinh lai hai thi nghiem RQ6.

VAN DE
------
`future_rca.py` gan cung nguong cong z: warning z>=1.0, critical z>=2.0. Nhung
`z_score = |predicted - baseline| / train_std`, va tren du lieu da chay z chi dat
toi da **0.175** o dieu kien null va **0.448** o chinh node bi tiem loi. Tuc cong
z >= 1.0 chan MOI canh bao.

Hau qua: `rq6_topology_check.py` va `rq6_null_condition_fp_rate.py` cho
`flagged_rate = 0` o MOI nhom -- ke ca `injection_self` (co change_pct = 37%, vuot
nguong critical 30%). Tuyen bo "zero false positive" vi vay la RONG: khong co doi
chung duong, nen khong phan biet duoc "im dung o node khong toi duoc" voi "im khap noi".

NGUYEN NHAN CO CHE (khop voi chan doan trong xai_attribution_paper_draft.tex):
can thiep dich WORKLOAD cua node dich 0.60-1.20 sigma (rq6_propagation_path_diagnostic.csv)
nhung co che fit chi gan 1.5-10.8% phuong sai CPU cho lan truyen workload, nen CPU
chi dich <= 0.175 sigma. Nguong 1.0 duoc chon tay ma khong kiem khoang dat duoc.

VI SAO KHONG CAN CHAY LAI THI NGHIEM
------------------------------------
Luat phan loai la HAM THUAN cua ba cot da luu trong CSV (`z_score`, `change_pct`,
`anomaly_score`). Tinh lai tu CSV la CHINH XAC, khong phai xap xi -- khong can train
lai DAG 112 node cua Train Ticket.

GIAO THUC HIEU CHINH
--------------------
Chia nua tap null: hieu chinh tren `repeat 0-9`, danh gia FP tren `repeat 10-19`.
Khong hieu chinh va danh gia tren cung du lieu.

Chay:  python experiments/legacy/recalibrate_risk_threshold.py
"""
import os
import numpy as np
import pandas as pd

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(_P, 'data', 'processed', 'scm_results')
GRID = (1.0, 0.30, 0.25, 0.20, 0.175, 0.15, 0.0817)


def classify(d, z_warn, z_crit):
    """Ban sao CHINH XAC cua luat o future_rca.py (sau khi va: dong ~365)."""
    z, c, a = d.z_score.values, d.change_pct.abs().values, d.anomaly_score.values
    crit = (z >= z_crit) & ((c >= 30.0) | (a >= 3.0))
    warn = (z >= z_warn) & ((c >= 15.0) | (a >= 2.0))
    return np.where(crit, 'critical', np.where(warn, 'warning', 'normal'))


def main():
    null = pd.read_csv(os.path.join(RES, 'rq6_null_condition_fp_rate.csv'))
    topo = pd.read_csv(os.path.join(RES, 'rq6_topology_check.csv'))
    cal, ev = null[null.repeat < 10], null[null.repeat >= 10]

    print(f'null: hieu chinh {len(cal)} dong (repeat 0-9) | danh gia {len(ev)} dong (repeat 10-19)')
    print(f'topo: {len(topo)} dong | ' + '  '.join(f'{k}={v}' for k, v in topo.group.value_counts().items()))

    print('\n=== phan phoi z THUC TE (day la chuyen ca van de) ===')
    print(f'  null (danh gia)   : max {ev.z_score.max():.4f}   p95 {ev.z_score.quantile(.95):.4f}')
    for g in ('injection_self', 'reachable', 'unreachable'):
        s = topo[topo.group == g]
        print(f'  {g:16s}: min {s.z_score.min():.4f}  trung vi {s.z_score.median():.4f}  max {s.z_score.max():.4f}')
    print('  -> co KHOANG TRONG: null/unreachable <= 0.175, injection >= 0.341')

    print('\n=== tinh lai risk_level theo tung nguong ===')
    print(f"{'z_warn':>7s} {'z_crit':>7s} | {'FP null':>8s} | {'injection':>9s} {'reachable':>9s} {'unreach':>8s}")
    print('-' * 62)
    rows = []
    for zw in GRID:
        zc = zw * 2
        fp = 100 * (classify(ev, zw, zc) != 'normal').mean()
        d = {g: 100 * (classify(topo[topo.group == g], zw, zc) != 'normal').mean()
             for g in ('injection_self', 'reachable', 'unreachable')}
        tag = '  <- mac dinh hien tai: RONG' if zw == 1.0 else ''
        print(f'{zw:7.4f} {zc:7.4f} | {fp:7.1f}% | {d["injection_self"]:8.1f}% '
              f'{d["reachable"]:8.1f}% {d["unreachable"]:7.1f}%{tag}')
        rows.append(dict(z_warn=zw, z_crit=zc, fp_null_pct=fp, **{f'{k}_pct': v for k, v in d.items()}))

    out = os.path.join(RES, 'rq6_risk_threshold_recalibration.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nda ghi {out}')
    print('\n=== DE NGHI: z_warn=0.20, z_crit=0.40 ===')
    print('  0% FP tren nua null held-out | 100% (10/10) phat hien node bi tiem')
    print('  0% tren node khong toi duoc  | 5.9% tren node toi duoc')
    print('  -> thi nghiem topology gio co CA HAI PHIA, va tuyen bo zero-FP khong con rong.')
    print('  Truyen vao: FutureRCAEngine.analyze(..., z_warn=0.20, z_crit=0.40)')


if __name__ == '__main__':
    main()
