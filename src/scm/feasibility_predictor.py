# -*- coding: utf-8 -*-
"""
BO DU DOAN KHA THI -- P0 / P1 (KHONG toi uu; dung de DONG BANG du doan truoc khi do dap an)
============================================================================================
Khung: docs/DATA_FRAMEWORK.md muc 1-2. Cau hoi: them mot tinh nang CHUA TUNG CO thi he thong con
dap ung SLO o tai dinh L khong, va node nao nghen truoc.

Co che (hoc CHI tu du lieu KHONG co tinh nang):
  * rho_s        : ti le workload service s so voi gateway theo co cau request nen (hoi quy qua goc)
  * CPU_s = a_s + b_s * W_s   (NNLS: a_s, b_s >= 0)  -- co che tung service theo workload cua chinh no
Can thiep len mot yeu cau moi (Delta = expected_delta_pct * scale / 100, chain = call chain cua archetype):
  * P0 (hanh vi hien tai cua CapacityAgent): gateway +Delta roi lan truyen theo co cau nen:
        W_s = rho_s * L * (1 + Delta)
  * P1 (chain lam DIEM DAT TAI): request tinh nang KHONG lan ra ngoai chain:
        W_s = L * (rho_s + Delta * k_s * [s in chain]),  k_s mac dinh 1
  * P1_ctrl (doi chung): nhu P1 nhung chain NGAU NHIEN cung kich thuoc -- neu khong te hon P1 thi chain
        chua chung minh duoc gia tri.
Phan quyet: u_s = CPU_s / C_s (C_s = 100 * cores, cung don vi cot `_cpu`), tren 5 node chinh;
        u* hieu chinh tu diem gay cua ramp BASELINE (khong dung tinh nang nao).

Tai co dinh mot L, W va CPU deu TUYEN TINH theo L nen diem gay R* co cong thuc dong (khong can quet).
Han che da biet (KHONG sua o giai doan nay): P1 chua tinh chi phi dieu phoi cao hon o gateway
(do duoc 1.3-2.1x tren backend-call), chua co khoang bat dinh, khong mo hinh queue/latency.
"""

import glob
import hashlib
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(__file__))
from request_router import SOCKSHOP_CALL_CHAINS  # noqa: E402  (nguon su that duy nhat cua taxonomy)

GATEWAY = 'front-end'
SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
SCORED = ['front-end', 'catalogue', 'user', 'carts', 'orders']      # docs/DATA_FRAMEWORK.md muc 3
FEATURE_ARCHETYPE = {'promo': 'APPLY_PROMO_CODE', 'recs': 'RECOMMEND_PRODUCTS',
                     'track': 'TRACK_PACKAGE', 'review': 'WRITE_PRODUCT_REVIEW'}
U_MARGIN = 0.10           # MARGINAL khi max_u trong 10% duoi u*


# ------------------------------------------------------------------ du lieu
def load_runs(root):
    """Doc moi run duoi `root` (cot gt_* duoc chuan hoa ve ten khong tien to), bo hang warmup."""
    frames = []
    for p in sorted(glob.glob(os.path.join(root, '*', '*', 'simple_metrics.csv'))):
        d = pd.read_csv(p)
        d = d.rename(columns={c: c[3:] for c in d.columns if c.startswith('gt_')})
        d['_run'] = os.path.relpath(os.path.dirname(p), root).replace(os.sep, '/')
        for wcol in ('warmup_flag', 'step_warm'):
            if wcol in d:
                d = d[d[wcol] == 0]
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f'khong co simple_metrics.csv trong {root}')
    return pd.concat(frames, ignore_index=True)


def level_of(df):
    """Nhan muc tai cua moi hang: load_level (quet) hoac target_rps (ramp)."""
    if 'load_level' in df:
        return df['load_level']
    if 'target_rps' in df:
        return df['target_rps']
    raise KeyError('thieu cot muc tai (load_level/target_rps)')


def select_train(df, train_max, allow_limits=False):
    """Chi hang KHONG tinh nang, chua bao hoa, muc tai <= train_max. Vi pham hop dong => bao loi to (khong am tham)."""
    feat = df['feature'] if 'feature' in df else pd.Series('base', index=df.index)
    if (feat != 'base').any():
        raise ValueError('du lieu train chua luu luong tinh nang (ro ri) -- dung khung du lieu')
    lim = df['limits_cfg'] if 'limits_cfg' in df else pd.Series('none', index=df.index)
    if not allow_limits and (lim != 'none').any():
        raise ValueError('du lieu train dat tran CPU (chi cho phep --dev)')
    m = level_of(df) <= train_max
    if 'step_violated' in df:
        m &= df['step_violated'] == 0
    return df[m]


# ------------------------------------------------------------------ co che
def fit_mechanism(df, services=SERVICES, gateway=GATEWAY):
    wg = df[f'{gateway}_workload'].to_numpy(dtype=float)
    mech = {}
    for s in services:
        d = df[[f'{s}_workload', f'{s}_cpu']].copy()
        d['wg'] = wg
        d = d.dropna()
        w, c, g = d[f'{s}_workload'].to_numpy(float), d[f'{s}_cpu'].to_numpy(float), d['wg'].to_numpy(float)
        rho = float(g @ w / (g @ g)) if s != gateway else 1.0
        (a, b), _ = nnls(np.column_stack([np.ones_like(w), w]), c)
        pred = a + b * w
        r2 = 1 - np.sum((c - pred) ** 2) / max(np.sum((c - c.mean()) ** 2), 1e-12)
        mech[s] = {'rho': rho, 'alpha': float(a), 'beta': float(b), 'r2': float(r2), 'n': int(len(w)),
                   'w_p99_train': float(np.percentile(w, 99)), 'w_max_train': float(w.max())}
    return mech


# ------------------------------------------------------------------ du doan
def wrong_chain(feature, seed=0):
    """Chain NGAU NHIEN cung kich thuoc, luon co gateway (moi request di qua gateway) -- doi chung cua P1."""
    true = SOCKSHOP_CALL_CHAINS[FEATURE_ARCHETYPE[feature]]['services']
    others = [s for s in SERVICES if s != GATEWAY]
    rng = random.Random(f'{feature}:{seed}')
    return [GATEWAY] + rng.sample(others, len(true) - 1)


class FeasibilityPredictor:
    def __init__(self, mech, cores, u_star, margin=U_MARGIN, scored=SCORED):
        self.mech, self.u_star, self.margin, self.scored = mech, float(u_star), margin, list(scored)
        self.cap = {s: 100.0 * float(cores[s]) for s in SERVICES if s in cores}
        missing = [s for s in self.scored if s not in self.cap]
        if missing:
            raise ValueError(f'thieu tran C_s cho {missing}')

    # --- can thiep
    def spec(self, feature, scale=1.0, chain=None):
        if feature in (None, 'base'):
            return 0.0, []
        arch = SOCKSHOP_CALL_CHAINS[FEATURE_ARCHETYPE[feature]]
        return arch['expected_delta_pct'] * scale / 100.0, list(chain if chain is not None else arch['services'])

    def workloads(self, L, mode='P1', feature=None, scale=1.0, chain=None, k=None):
        delta, ch = self.spec(feature, scale, chain)
        k = k or {}
        out = {}
        for s in SERVICES:
            rho = self.mech[s]['rho']
            if mode == 'P0':
                out[s] = rho * L * (1.0 + delta)
            elif mode == 'P1':
                out[s] = L * (rho + (delta * k.get(s, 1.0) if s in ch else 0.0))
            else:
                raise ValueError(mode)
        return out

    def cpu(self, W):
        return {s: self.mech[s]['alpha'] + self.mech[s]['beta'] * W[s] for s in SERVICES}

    def utilization(self, L, **kw):
        cpu = self.cpu(self.workloads(L, **kw))
        return {s: cpu[s] / self.cap[s] for s in self.cap}

    def verdict(self, L, **kw):
        W = self.workloads(L, **kw)
        u = {s: v for s, v in self.utilization(L, **kw).items() if s in self.scored}
        bott = max(u, key=u.get)
        mu = u[bott]
        v = 'INFEASIBLE' if mu > self.u_star else ('MARGINAL' if mu > self.u_star * (1 - self.margin) else 'FEASIBLE')
        extrap = [s for s in self.scored if W[s] > self.mech[s]['w_max_train']]
        return {'verdict': v, 'max_u': round(mu, 4), 'bottleneck': bott, 'extrapolating': bool(extrap),
                'extrap_nodes': extrap, 'u': {k: round(x, 4) for k, x in u.items()}}

    def breakpoint(self, **kw):
        """R* dong: tai L lon nhat de max_s u_s(L) <= u*. Tuyen tinh theo L nen giai dong tung node."""
        w1 = self.workloads(1.0, **kw)
        best, node = float('inf'), None
        for s in self.scored:
            slope = self.mech[s]['beta'] * w1[s]
            if slope <= 0:
                continue
            ls = (self.u_star * self.cap[s] - self.mech[s]['alpha']) / slope
            if ls < best:
                best, node = ls, s
        return float(best), node


# ------------------------------------------------------------------ hieu chinh u* tu ramp BASELINE
def calibrate_u_star(ramp_root, cores, scored=SCORED):
    """u* = muc su dung cua node nghen tai bac DAT SLO cuoi cung truoc diem gay (trung vi qua cac ramp baseline).
    Chi doc ramp `base`; tu choi neu thay ramp co tinh nang (se la nhin truoc dap an)."""
    vals, detail = [], []
    for sp in sorted(glob.glob(os.path.join(ramp_root, 'ramp_*', 'run*', 'steps.json'))):
        j = json.load(open(sp, encoding='utf-8'))
        if j.get('feature', 'base') != 'base':
            raise ValueError(f'hieu chinh u* chi dung ramp baseline, gap {j.get("feature")} o {sp}')
        lo = j['breakpoint']['lo']
        if lo is None:
            continue
        d = pd.read_csv(os.path.join(os.path.dirname(sp), 'simple_metrics.csv'))
        d = d.rename(columns={c: c[3:] for c in d.columns if c.startswith('gt_')})
        d = d[(d['step_warm'] == 0) & (d['target_rps'] == lo)]
        if d.empty:
            continue
        u = {s: float(d[f'{s}_cpu'].mean()) / (100.0 * cores[s]) for s in scored}
        b = max(u, key=u.get)
        vals.append(u[b])
        detail.append({'ramp': os.path.relpath(os.path.dirname(sp), ramp_root).replace(os.sep, '/'),
                       'lo': lo, 'bottleneck': b, 'u': round(u[b], 4)})
    if not vals:
        raise ValueError('khong co ramp baseline nao co diem gay de hieu chinh u*')
    return float(np.median(vals)), detail


def sha256_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(os.path.dirname(os.path.dirname(p))).encode())
        h.update(open(p, 'rb').read())
    return h.hexdigest()
