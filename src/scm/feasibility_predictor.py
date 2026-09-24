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
  * P2 (PHAT TRIEN, tham so hoc tu ramp promo/recs -- KHONG phai du doan truoc): nhu P1 nhung
        backend trong chain:  W_s = L*(rho_s + Delta*k_s*c_s)   c_s = he so chi phi MOI LAN GOI cua tinh nang so voi trung binh nen
        gateway            :  W_fe = L*(rho + Delta*(1 + x*n_calls)),  n_calls = |chain|-1 (taxonomy)  -- chi phi dieu phoi
    (do duoc: chain dung tuyen duong, k_s~1; sai so P1 gan nhu HOAN TOAN o chi phi moi lan goi: orders ~0.5x, gateway ~1.3x)
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
                     'track': 'TRACK_PACKAGE', 'review': 'WRITE_PRODUCT_REVIEW',
                     # tinh nang DOC LAP (cai boi agent rieng, khong thay taxonomy): REQ-06, REQ-05, REQ-10
                     'cartsum': 'VIEW_CART', 'quickadd': 'ADD_TO_CART', 'express': 'PLACE_ORDER',
                     # tinh nang KIEM DINH TIEN CUU cho P3 (mot agent DOC LAP khac, sau khi P0/P1/P2 da dong bang)
                     'browse': 'GET_CATALOGUE'}
FEATURE_SETS = {'main': ['promo', 'recs', 'track', 'review'], 'indep': ['cartsum', 'quickadd', 'express'], 'prosp': ['browse']}
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
    """Chain NGAU NHIEN cung kich thuoc, luon co gateway (moi request di qua gateway) -- doi chung cua P1.

    CANH BAO ve cach dung: day la MOT hien thuc cua mot doi chung NGAU NHIEN. Moi ket
    luan rut ra tu mot `seed` duy nhat deu la ket luan ve hien thuc do, khong phai ve
    doi chung. Do manh cua bang chung thay doi hoan toan theo seed: tren 200 lan boc
    (16 o tinh nang x node), Wilcoxon p trai tu 0.008 den 1.000, trung vi 0.19 -- va
    seed=0 (gia tri mac dinh, dung trong ban dong bang) cho p=0.59. Huong thi nhat
    quan (chain that tot hon o 99.5% lan boc) nhung 56% so o cho du doan Y HET nhau,
    vi chain ngau nhien cung kich thuoc thuong trung lap nhieu voi chain that.

    => Bao cao ket qua doi chung PHAI dung wrong_chain_draws() va trinh bay PHAN PHOI.
    Ham nay giu nguyen chu ky va hanh vi CHI de ban dong bang
    (predictions_frozen_RE2.json, SHA-256 f05c8eb5...bc8e2) con tai lap duoc bit-for-bit;
    khong duoc doi mac dinh seed=0.
    """
    true = SOCKSHOP_CALL_CHAINS[FEATURE_ARCHETYPE.get(feature, feature)]['services']
    others = [s for s in SERVICES if s != GATEWAY]
    rng = random.Random(f'{feature}:{seed}')
    return [GATEWAY] + rng.sample(others, len(true) - 1)


def wrong_chain_draws(feature, n_draws=200, start=0):
    """n_draws hien thuc DOC LAP cua doi chung chain-sai cho `feature`.

    Don vi ngau nhien hoa dung la MOT chain sai cho MOI TINH NANG, giu co dinh qua moi
    buoc tai cua tinh nang do -- boc lai o tung buoc tai se tron lan phuong sai giua
    cac lan boc voi trung binh trong mot lan boc, lam hep gia tao khoang bat dinh.
    """
    return [wrong_chain(feature, s) for s in range(start, start + n_draws)]


class FeasibilityPredictor:
    def __init__(self, mech, cores, u_star, margin=U_MARGIN, scored=SCORED, feature_cost=None,
                 feature_demand=None, slo_p99=None):
        self.mech, self.u_star, self.margin, self.scored = mech, float(u_star), margin, list(scored)
        self.feature_cost = feature_cost or {'c': {}, 'x': 0.0}      # chi dung cho mode P2
        # Cong nhu cau phuc vu: TAT theo mac dinh. Chi bat khi ca hai duoc cung cap, nen
        # moi noi goi cu giu nguyen hanh vi va ban dong bang van tai lap duoc bit-for-bit.
        self.feature_demand = dict(feature_demand or {})   # {ten tinh nang: D_feat (giay)}
        self.slo_p99 = float(slo_p99) if slo_p99 is not None else None
        self.cap = {s: 100.0 * float(cores[s]) for s in SERVICES if s in cores}
        missing = [s for s in self.scored if s not in self.cap]
        if missing:
            raise ValueError(f'thieu tran C_s cho {missing}')

    # --- can thiep
    def spec(self, feature, scale=1.0, chain=None):
        """`feature` nhan MOT trong hai dang: ten tinh nang da dat (khoa cua FEATURE_ARCHETYPE, vd 'promo')
        hoac TEN ARCHETYPE THAT (khoa cua SOCKSHOP_CALL_CHAINS, vd 'LOGIN') -- cho phep dung voi bat ky yeu
        cau moi nao ma ParserAgent phan loai duoc, khong chi 8 tinh nang da co bang chung thuc nghiem."""
        if feature in (None, 'base'):
            return 0.0, []
        archetype_name = FEATURE_ARCHETYPE.get(feature, feature)
        if archetype_name not in SOCKSHOP_CALL_CHAINS:
            raise KeyError(f"'{feature}' khong phai ten tinh nang da dat (FEATURE_ARCHETYPE) "
                          f"hay ten archetype hop le (SOCKSHOP_CALL_CHAINS: {sorted(SOCKSHOP_CALL_CHAINS)})")
        arch = SOCKSHOP_CALL_CHAINS[archetype_name]
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
            elif mode == 'P2':
                fc = self.feature_cost
                if s == GATEWAY:
                    # n_calls = TONG boi so goi backend (Sum k_m). Voi k mac dinh (rong) => k.get(m,1.0)=1.0
                    # moi node -> tong = len(ch)-1, GIONG HET cong thuc cu (tuong thich nguoc). Khi co k do
                    # duoc (P3), n_calls phan anh dung tong luot goi backend thuc (vd express: 13, khong phai 6).
                    n_calls = sum(k.get(m, 1.0) for m in ch if m != GATEWAY)
                    out[s] = L * (rho + delta * (1.0 + fc['x'] * n_calls))
                else:
                    out[s] = L * (rho + (delta * k.get(s, 1.0) * fc['c'].get(s, 1.0) if s in ch else 0.0))
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
        """R* dong: tai L lon nhat de max_s u_s(L) <= u*. Tuyen tinh theo L nen giai dong tung node.

        CHI tieu chi CPU -- giu nguyen hanh vi cu, bit-for-bit, vi ban dong bang
        (predictions_frozen_RE2.json) va 7 noi goi khac phu thuoc vao no.
        Tieu chi thu hai xem breakpoint_demand() / breakpoint_gated().
        """
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

    # ---------------------------------------------------------------- cong thu hai: nhu cau phuc vu
    # Ly do ton tai: tieu chi u_s <= u* gia dinh nut that LUON la CPU. Do tren 16 ramp
    # that cho thay dieu do sai o 4 o (quickadd x1/x2, express x1, promo x2): SLO vo khi
    # node nghen moi o u = 0.32-0.67, va muc su dung tai diem gay LECH CO HE THONG theo
    # cau truc goi cua tinh nang (nhom quat-ra nhieu 0.547 vs nhom thuong 0.867,
    # Mann-Whitney p=0.0011 sau Holm). Mot nguong CPU duy nhat vi vay khong the dung cho
    # moi loai tinh nang.
    #
    # Co so: dinh luat nhu cau phuc vu (Denning & Buzen 1978) -- D = Sum_s k_s*S_s la thoi
    # gian phuc vu tich luy doc chuoi goi, va thoi gian dap ung moi tram phinh theo
    # 1/(1-u_s) (Lazowska et al. 1984). D_feat do TRUC TIEP boi probe khi he RANH, khong
    # dung mot hat du lieu tai nao -- nen dong bang truoc ramp duoc.
    #
    # Tinh chat kiem tra duoc: khi L -> 0 moi u_s -> 0 va R_feat -> D_feat, tuc quy ve
    # dung gia tri probe do duoc. Khi Sum k = 1 va mot nut thong tri, cong nay lai gan
    # nhu tieu chi u* cu -- tuong thich nguoc ve mat co che.

    def demand_latency(self, L, **kw):
        """R_feat(L): do tre du bao cua CHINH request tinh nang o tai nen L.

        R_feat(L) = Sum_s (k_s / Sum k) * D_feat / (1 - u_s(L)),  s in chain + gateway.
        Tra ve None neu chua co D_feat cho tinh nang do (cong tat, khong doan bua).
        """
        D = self.feature_demand.get(kw.get('feature'))
        if D is None:
            return None
        _, ch = self.spec(kw.get('feature'), kw.get('scale', 1.0), kw.get('chain'))
        k = kw.get('k') or {}
        w = {s: (1.0 if s == GATEWAY else float(k.get(s, 1.0))) for s in set(ch) | {GATEWAY}}
        tot = sum(w.values()) or 1.0
        u = self.utilization(L, **kw)
        return sum((w[s] / tot) * D / max(1.0 - min(u.get(s, 0.0), 0.995), 5e-3) for s in w)

    def breakpoint_demand(self, hi=1000.0, tol=1e-3, **kw):
        """L lon nhat de R_feat(L) <= slo_p99. R_feat don dieu tang theo L nen chia doi.

        Tra ve None neu cong tat (thieu D_feat hoac slo_p99). Tra ve `hi` neu cong KHONG
        rang buoc trong dai [1, hi] -- khi do phep min() o breakpoint_gated() tu bo qua no.
        """
        if self.slo_p99 is None or self.demand_latency(1.0, **kw) is None:
            return None
        if self.demand_latency(1.0, **kw) > self.slo_p99:
            return 1.0
        lo = 1.0
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.demand_latency(mid, **kw) <= self.slo_p99:
                lo = mid
            else:
                hi = mid
        return float(lo)

    def breakpoint_gated(self, **kw):
        """Diem gay duoi CA HAI tieu chi: R* = min(R*_cpu, R*_demand).

        Tra ve dict, khong phai tuple, de noi ro cong nao RANG BUOC -- day la thong tin
        ky su can: 'cpu' nghia la he het CPU truoc, 'demand' nghia la do tre vuot SLO
        trong khi CPU con thua, va hai truong hop doi hoi hai cach xu ly khac nhau.
        """
        r_cpu, node = self.breakpoint(**kw)
        r_dem = self.breakpoint_demand(**kw)
        if r_dem is None:
            return {'R': r_cpu, 'R_cpu': r_cpu, 'R_demand': None,
                    'binding': 'cpu', 'bottleneck': node, 'demand_gate': 'off'}
        binding = 'demand' if r_dem < r_cpu else 'cpu'
        return {'R': min(r_cpu, r_dem), 'R_cpu': r_cpu, 'R_demand': r_dem,
                'binding': binding, 'bottleneck': node, 'demand_gate': 'on'}


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


def fit_feature_cost(rows, mech):
    """Hieu chinh he so chi phi cua P2 tu cac hang do duoc (CHI tap phat trien).
    rows: DataFrame cot feature, node, L, delta, chain(set|list), n_calls, w_meas, c_meas (moi hang = 1 buoc tai truoc diem gay).
      c_s : chi phi CPU do tinh nang gay ra / (beta * tai tinh nang DO DUOC)  -> chi phi MOI LAN GOI (khong lan voi boi so k)
      x   : chi phi gateway cua 1 request tinh nang = (1 + x*n_calls) lan request nen; hoi quy qua goc tren cac hang gateway."""
    c, detail = {}, {}
    be = rows[(rows['node'] != GATEWAY) & rows.apply(lambda r: r['node'] in r['chain'], axis=1)]
    for s, g in be.groupby('node'):
        m = mech[s]
        extra_cpu = g['c_meas'] - (m['alpha'] + m['beta'] * m['rho'] * g['L'])
        extra_w = g['w_meas'] - m['rho'] * g['L']
        den = float((m['beta'] * extra_w).sum())
        if den > 0:
            c[s] = float(extra_cpu.sum() / den)
            detail[s] = {'n': int(len(g)), 'features': sorted(g['feature'].unique().tolist())}
    gw = rows[rows['node'] == GATEWAY]
    m = mech[GATEWAY]
    base_unit = m['beta'] * gw['delta'] * gw['L']                        # chi phi neu request tinh nang tot nhu request nen
    extra = gw['c_meas'] - (m['alpha'] + m['beta'] * m['rho'] * gw['L'])
    z = base_unit * gw['n_calls']
    x = float(((extra - base_unit) * z).sum() / max((z * z).sum(), 1e-12))
    per_feature = {f: float(((g['c_meas'] - (m['alpha'] + m['beta'] * m['rho'] * g['L'])).sum()) /
                            max((m['beta'] * g['delta'] * g['L']).sum(), 1e-12)) for f, g in gw.groupby('feature')}
    return {'c': c, 'x': x, 'gateway_multiple_by_feature': per_feature, 'detail': detail}


def load_slo(path=None):
    """Doc SLO tu deploy/sockshop/slo.json -- KHONG cam cung nguong trong ma nguon.

    File do ghi ro 'CHOT sau Phase 0 va KHONG doi sau khi do dap an'; doc tu file giu
    dung rang buoc do, va mot lan sua nguong se lan toi moi noi dung no thay vi lech
    am tham giua cac script.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'deploy', 'sockshop', 'slo.json')
    with open(os.path.abspath(path), encoding='utf-8') as f:
        j = json.load(f)
    return {'p99_s': float(j['p99_s']), 'err_rate': float(j['err_rate'])}


def demand_from_probe(path):
    """{tinh nang: D_feat (giay)} tu file probe (--save cua probe_feature_chain.py).

    Doc truong `latency_idle.p99` ma probe do khi he RANH. File probe cu (truoc khi them
    bam gio) khong co truong nay -- khi do tinh nang do bi BO QUA, cong tu tat cho no,
    thay vi doan mot gia tri.
    """
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    return {k: v['latency_idle']['p99'] for k, v in feats.items()
            if isinstance(v.get('latency_idle'), dict) and 'p99' in v['latency_idle']}


def sha256_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(os.path.dirname(os.path.dirname(p))).encode())
        h.update(open(p, 'rb').read())
    return h.hexdigest()
