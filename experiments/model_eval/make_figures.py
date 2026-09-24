# -*- coding: utf-8 -*-
"""SINH HINH CHO BAI BAO -> docs/figures/*.pdf
=================================================
Bai hien trinh bay MOI ket qua dinh luong bang BANG. Ba ket qua duoi day mat rat
nhieu suc thuyet phuc o dang bang, vi dieu can thay la HINH DANG chu khong phai con so:

  fig_extrapolation  (RQ3) cay dong bang o bien, GP dao dau, dang tham so bam theo su that
  fig_signal_gate    (RQ3) RCAEval khong co bien thien tai -- nhin mot cai la thay
  fig_breakpoint     (RQ5) du doan vs khoang do duoc, va HUONG cua sai so

Quy uoc trinh bay (in giay, hai cot IEEE):
  * PDF vector, be rong dung 1 cot (3.4in) -- khong phong to thu nho sau
  * Bang mau da qua validator cua skill dataviz (6 kiem tra, che do light)
  * MOI series co CA mau LAN net/marker rieng -> con doc duoc khi in TRANG DEN
  * Nhan truc tiep tren duong thay vi chu giai roi, vi ban in khong co hover
  * Khong bao gio hai truc y

    python experiments/make_figures.py
"""

import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src', 'scm'))
import feasibility_predictor as FP  # noqa: E402
from data_processor import load_normal_data  # noqa: E402

FIG = os.path.join(BASE, 'docs', 'figures')
os.makedirs(FIG, exist_ok=True)

# --- bang mau: 4 slot dau cua theme categorical mac dinh, da chay validate_palette.js
#     (PASS lightness / chroma / CVD dE 9.1 / normal-vision dE 22.9; WARN contrast ->
#      bu bang nhan truc tiep, dung nhu skill yeu cau)
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100'}
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8a85'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 7.2, 'axes.labelsize': 7.5, 'axes.titlesize': 7.8,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.8,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'grid.color': '#e6e6e2', 'grid.linewidth': 0.5,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,          # font nhung duoc, khong bi rasterise
})
COL = 3.4        # be rong mot cot IEEE (inch)


def _tidy(ax, grid='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid:
        ax.grid(axis=grid, zorder=0)
        ax.set_axisbelow(True)


# =============================================================== fig 1
def fig_extrapolation():
    """Ba lop mo hinh phan ky NGOAI dai huan luyen, theo cach du doan truoc duoc."""
    SS = FP.select_train(FP.load_runs(os.path.join(BASE, 'data', 'raw', 'SS-TRAIN')),
                         train_max=10 ** 9)
    d = SS[['front-end_workload', 'front-end_cpu']].dropna().sort_values('front-end_workload')
    w, y = d['front-end_workload'].to_numpy(float), d['front-end_cpu'].to_numpy(float)
    n = int(len(w) * 0.67)
    X = w.reshape(-1, 1)

    (a, b), _ = nnls(np.column_stack([np.ones(n), w[:n]]), y[:n])
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   random_state=42).fit(X[:n], y[:n])
    gp = Pipeline([('s', StandardScaler()),
                   ('g', GaussianProcessRegressor(
                       kernel=ConstantKernel(1.) * RBF(1.) + WhiteKernel(.1),
                       n_restarts_optimizer=1, alpha=1e-3, normalize_y=True))]).fit(X[:n], y[:n])

    gx = np.linspace(w.min(), w.max(), 400)
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    bnd = w[n]
    ax.axvspan(bnd, w.max() * 1.02, color='#f4f4f1', zorder=0)
    ax.axvline(bnd, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(bnd + 3, y.max() * 1.0, 'ngoài dải huấn luyện', fontsize=6.4,
            color=INK2, ha='left', va='top')

    ax.scatter(w, y, s=3.2, color=MUTED, alpha=.35, lw=0, zorder=2, label='đo được')
    ax.plot(gx, a + b * gx, color=C['blue'], lw=1.5, zorder=5)
    ax.plot(gx, gb.predict(gx.reshape(-1, 1)), color=C['orange'], lw=1.5,
            ls=(0, (4, 1.6)), zorder=4)
    ax.plot(gx, gp.predict(gx.reshape(-1, 1)), color=C['aqua'], lw=1.5,
            ls=(0, (1.2, 1.4)), zorder=4)

    xe = w.max()
    for lbl, val, col, dy in (('NNLS (SCM)', a + b * xe, C['blue'], 2.0),
                              ('GradBoost', gb.predict([[xe]])[0], C['orange'], 1.6),
                              ('GaussProc', gp.predict([[xe]])[0], C['aqua'], -3.4)):
        ax.annotate(lbl, (xe, val), xytext=(-2, dy), textcoords='offset points',
                    fontsize=6.5, color=INK, ha='right',
                    va='bottom' if dy > 0 else 'top')
    ax.annotate('đo được', (xe, y[-1]), xytext=(-2, -9), textcoords='offset points',
                fontsize=6.5, color=INK2, ha='right', va='top')

    ax.set_xlabel('workload của front-end (req/s)')
    ax.set_ylabel('CPU (% của một core)')
    ax.set_xlim(0, w.max() * 1.02)
    _tidy(ax, grid='y')
    p = os.path.join(FIG, 'fig_extrapolation.pdf')
    fig.savefig(p); plt.close(fig)
    print(f'  {os.path.basename(p)}   (sự thật {y[-1]:.1f} | NNLS {a+b*xe:.1f} | '
          f'GradBoost {gb.predict([[xe]])[0]:.1f} | GP {gp.predict([[xe]])[0]:.1f})')


# =============================================================== fig 2
def fig_signal_gate():
    """Cung he, cung service, cung metric -- chi khac cach du lieu duoc SINH RA."""
    SS = FP.select_train(FP.load_runs(os.path.join(BASE, 'data', 'raw', 'SS-TRAIN')),
                         train_max=10 ** 9)
    ss = SS[['front-end_workload', 'front-end_cpu']].dropna()
    re = load_normal_data('front-end', 'cpu')

    def r2(x, y):
        A = np.column_stack([np.ones(len(x)), x])
        (a_, b_), _ = nnls(A, y)
        return 1 - ((y - (a_ + b_ * x)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.85))
    for ax, (x, y, ttl, col) in zip(axes, [
            (re['Workload'].to_numpy(float), re['Target'].to_numpy(float),
             'RCAEval (benchmark)', C['orange']),
            (ss['front-end_workload'].to_numpy(float), ss['front-end_cpu'].to_numpy(float),
             'SS-TRAIN (testbed tự dựng)', C['blue'])]):
        ax.scatter(x, y, s=2.2, color=col, alpha=.28, lw=0, zorder=3)
        ax.set_title(ttl, color=INK, pad=3)
        ax.set_xlabel('workload (req/s)')
        ax.text(.04, .95, f'$R^2 = {r2(x, y):.3f}$', transform=ax.transAxes,
                fontsize=7, color=INK, va='top')
        ax.set_xlim(0, 290)
        _tidy(ax, grid='both')
    axes[0].set_ylabel('CPU (% của một core)')
    p = os.path.join(FIG, 'fig_signal_gate.pdf')
    fig.savefig(p); plt.close(fig)
    print(f'  {os.path.basename(p)}   (R2: RCAEval {r2(re["Workload"].to_numpy(float), re["Target"].to_numpy(float)):.4f} | '
          f'SS-TRAIN {r2(ss["front-end_workload"].to_numpy(float), ss["front-end_cpu"].to_numpy(float)):.4f})')


# =============================================================== fig 3
def fig_breakpoint():
    """Du doan vs khoang DO DUOC, va huong sai so -- thu ma bang khong the hien duoc."""
    import glob
    fz = json.load(open(f'{BASE}/data/processed/frozen/predictions_frozen_RE2.json', encoding='utf-8'))
    p2 = json.load(open(f'{BASE}/data/processed/frozen/p2_params_dev.json', encoding='utf-8'))['params']
    KM = json.load(open(f'{BASE}/data/processed/frozen/k_measured.json', encoding='utf-8'))['features']
    KM.update(json.load(open(f'{BASE}/data/processed/frozen/k_measured_prosp.json', encoding='utf-8'))['features'])
    P = FP.FeasibilityPredictor(fz['mechanism'], fz['params']['cores'], fz['params']['u_star'],
                                feature_cost=p2)

    cells = {}
    for root in ('SS-LIMITS', 'SS-LIMITS-CLEAN'):
        for sp in sorted(glob.glob(f'{BASE}/data/raw/{root}/ramp_*/run*/steps.json')):
            j = json.load(open(sp, encoding='utf-8'))
            f = j.get('feature', 'base')
            if f == 'base' or j['breakpoint']['lo'] is None:
                continue
            sc = 2.0 if os.path.basename(os.path.dirname(os.path.dirname(sp))).endswith('x2') else 1.0
            c = cells.setdefault((f, sc), {'lo': [], 'hi': []})
            c['lo'].append(j['breakpoint']['lo'])
            if j['breakpoint'].get('hi'):
                c['hi'].append(j['breakpoint']['hi'])

    rows = []
    for (f, sc), v in cells.items():
        lo = float(np.median(v['lo'])); hi = float(np.median(v['hi'])) if v['hi'] else lo + 20
        k = KM[f]['measured_per_use']
        rows.append({'cell': f'{f}×{sc:g}', 'lo': lo, 'hi': hi,
                     'P1': P.breakpoint(mode='P1', feature=f, scale=sc)[0],
                     'P2': P.breakpoint(mode='P2', feature=f, scale=sc)[0],
                     'P3': P.breakpoint(mode='P2', feature=f, scale=sc, k=k)[0]})
    R = pd.DataFrame(rows).sort_values('lo').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(COL, 3.05))
    yy = np.arange(len(R))
    for i, r in R.iterrows():                        # vung NGUY HIEM = du doan MUON hon hi
        ax.add_patch(plt.Rectangle((r['hi'], i - .42), 300, .84,
                                   color='#fbeee8', lw=0, zorder=0))
        ax.plot([r['lo'], r['hi']], [i, i], color=INK, lw=2.6, solid_capstyle='butt', zorder=3)
    for nm, col, mk in (('P1', C['yellow'], 'o'), ('P2', C['aqua'], 's'), ('P3', C['blue'], 'D')):
        ax.scatter(R[nm], yy, s=13, color=col, marker=mk, zorder=5,
                   edgecolors='white', linewidths=.6, label=nm)
    ax.set_yticks(yy); ax.set_yticklabels(R['cell'])
    ax.set_xlabel('điểm gãy (req/s)')
    ax.set_xlim(0, 210)
    ax.text(200, len(R) - .3, 'dự đoán MUỘN =\nbáo khả thi giả', fontsize=6.2,
            color='#a8442a', ha='right', va='top', linespacing=1.25)
    ax.plot([], [], color=INK, lw=2.6, label='khoảng đo được')
    ax.legend(loc='lower right', frameon=False, handletextpad=.5, borderpad=.2)
    _tidy(ax, grid='x')
    ax.invert_yaxis()
    p = os.path.join(FIG, 'fig_breakpoint.pdf')
    fig.savefig(p); plt.close(fig)
    print(f'  {os.path.basename(p)}   ({len(R)} ô)')


if __name__ == '__main__':
    print('Sinh hinh -> docs/figures/')
    fig_extrapolation()
    fig_signal_gate()
    fig_breakpoint()
    print('Xong.')
