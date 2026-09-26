# -*- coding: utf-8 -*-
"""
NewFeatureFeasibilityAgent -- tich hop feasibility_predictor.py vao he thong
=============================================================================
KHONG dung chung class/pipeline voi CapacityAgent (train() tren du lieu RCAEval fault-injection,
phuc vu RQ1-RQ12 da cong bo). Day la mot agent RIENG cho dung mot cau hoi khac: "them tinh nang
CHUA TUNG CO thi he thong con dap ung SLO o tai dinh L khong" (docs/DATA_FRAMEWORK.md muc 1),
tach biet de KHONG dung cham vao pipeline/test da co (an toan tai hien cac RQ da cong bo).

Ba diem hoan thien so voi feasibility_predictor.py dung doc lap (script):
  1. C_s DOC TRUC TIEP tu container dang chay (`docker inspect`) thay vi file limits.json tinh --
     phan anh dung gioi han THAT tai thoi diem hoi, chiu duoc config drift. Fallback ve limits.json
     kem canh bao neu Docker khong goi duoc (vd moi truong CI khong co Docker).
  2. KHOANG BAT DINH cho diem gay: bootstrap KHONG tham so tren CHINH cac hang du lieu huan luyen
     (khong chi tren tham so da fit) -- resample co hoan lai, fit lai co che (rho/alpha/beta) moi
     lan, tinh lai diem gay, lay phan vi [5,95] thuc nghiem. Phan anh dung bat dinh do CO MAU HUU
     HAN, khong phai mot khoang gia dinh.
  3. `assess()` la API don, nhan mo ta archetype (hoac requirement da duoc ParserAgent phan loai)
     va tra ve FeasibilityVerdict co day du: phan quyet, node nghen, khoang tin cay, co ngoai suy.

Han che con lai (chua giai quyet, ghi vao Threats to Validity):
  * hieu ung hang doi/do tre duoi tu quat-ra nhieu backend KHONG duoc mo hinh (xem
    docs/DATA_FRAMEWORK.md muc 5h) -- assess() luon tra `latency_risk` de nguoi doc tu canh giac,
    khong tinh vao phan quyet CPU.
  * k_s (boi so goi) mac dinh 1 tru khi nguoi goi tu do va truyen vao (vd bang probe_feature_chain.py).
"""

import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_AGENT_DIR)
PROJECT_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, 'scm'))
import feasibility_predictor as FP  # noqa: E402

DEFAULT_FROZEN = os.path.join(PROJECT_ROOT, 'data', 'processed', 'frozen', 'predictions_frozen_RE2.json')
DEFAULT_P2 = os.path.join(PROJECT_ROOT, 'data', 'processed', 'frozen', 'p2_params_dev.json')
DEFAULT_LIMITS = os.path.join(PROJECT_ROOT, 'deploy', 'sockshop', 'limits.json')
DEFAULT_TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'SS-TRAIN')
DOCKER_PROJECT = 'sockshop'
BOOTSTRAP_B = 200
CI = (5, 95)
# hien tai P0-P3 chi mo hinh CPU; tinh nang co nhieu luot goi backend (Sigma k lon) co rui ro
# do tre hang doi khong duoc mo hinh (docs/DATA_FRAMEWORK.md muc 5h) -- nguong canh bao thuc nghiem.
LATENCY_RISK_N_CALLS = 4
# CONG THROTTLING (docs/DATA_FRAMEWORK.md muc 5c). Tien de "SLO vo quanh u* ~ 0.88" CHI duoc kiem
# chung o han ngach >= 0.5 core (front-end, cau hinh RE2). O han ngach nho hon, CFS throttling la
# co che rang buoc: cau hinh C1 (carts 0.15 core, JVM) va C2 (catalogue 0.08 core, Go) vo SLO o muc
# su dung TRUNG BINH 20-45%, khong phai ~88% -- catalogue bi throttle 0% (<=80 req/s) -> 32% (160
# req/s) va vo o 140 req/s thay vi ~223 nhu du doan. Duoi nguong nay bo du doan KHONG con hop le,
# nen phai tra UNDECIDED thay vi mot con so tu tin. Nguong lay tu CAU HINH han ngach (quyet dinh o
# Phase 0b), khong phai fit tren tinh nang nao.
QUOTA_MIN_VALIDATED_CORES = 0.5


@dataclass
class FeasibilityVerdict:
    verdict: str                      # FEASIBLE | MARGINAL | INFEASIBLE
    breakpoint_rps: float             # R* trung vi (bootstrap)
    breakpoint_ci: tuple              # (p5, p95) tren R*
    bottleneck: str
    max_utilization: float
    utilization_by_node: Dict[str, float]
    extrapolating: bool
    extrap_nodes: List[str]
    latency_risk: bool                 # Sigma k >= LATENCY_RISK_N_CALLS -> CPU khong du, xem muc 5h
    throttling_risk: bool              # han ngach cua node nghen < QUOTA_MIN_VALIDATED_CORES -> ngoai
                                       # pham vi da kiem chung (che do CFS throttling), xem muc 5c
    bottleneck_cores: float            # han ngach (core) cua node nghen, de nguoi doc tu kiem
    model: str                         # 'P2' hoac 'P3' (co k do duoc)
    k_source: str                      # 'gia dinh (k=1)' hoac 'do bang probe'
    cores_source: str                  # 'docker (song)' hoac 'limits.json (tinh, du phong)'
    n_bootstrap: int
    warnings: List[str] = field(default_factory=list)


def read_live_cores(services, project=DOCKER_PROJECT, fallback_limits=DEFAULT_LIMITS, fallback_config='RE2'):
    """Doc gioi han CPU (core) THAT tu container dang chay qua `docker inspect`. Neu Docker khong
    goi duoc hoac thieu container, du phong ve limits.json (tinh, co canh bao ro trong warnings)."""
    warnings, cores = [], {}
    for s in services:
        try:
            r = subprocess.run(['docker', 'inspect', f'{project}-{s}-1', '--format', '{{.HostConfig.NanoCpus}}'],
                               capture_output=True, text=True, timeout=10)
            nano = float(r.stdout.strip())
            if r.returncode != 0 or not r.stdout.strip():
                raise RuntimeError(r.stderr.strip() or 'trong')
            if nano <= 0:
                ncpu = subprocess.run(['docker', 'info', '--format', '{{.NCPU}}'], capture_output=True, text=True, timeout=10)
                nano = float(ncpu.stdout.strip() or 12) * 1e9
            cores[s] = nano / 1e9
        except Exception as e:
            warnings.append(f'khong doc duoc gioi han song cua {s} qua docker ({e}); dung du phong tinh')
    if len(cores) < len(services):
        cfg = json.load(open(fallback_limits, encoding='utf-8'))['configs'][fallback_config]
        for s in services:
            cores.setdefault(s, cfg.get(s, 12.0))
        return cores, 'limits.json (tinh, du phong)', warnings
    return cores, 'docker (song)', warnings


class NewFeatureFeasibilityAgent:
    """Dung: agent = NewFeatureFeasibilityAgent(); agent.assess('promo', scale=1.0, L_peak=150)."""

    def __init__(self, frozen_path=DEFAULT_FROZEN, p2_params_path=DEFAULT_P2, train_dir=DEFAULT_TRAIN_DIR,
                 docker_project=DOCKER_PROJECT, u_margin=FP.U_MARGIN):
        self.frozen = json.load(open(frozen_path, encoding='utf-8'))
        self.p2 = json.load(open(p2_params_path, encoding='utf-8'))['params'] if p2_params_path and os.path.exists(p2_params_path) else None
        self.train_dir = train_dir
        self.docker_project = docker_project
        self.u_margin = u_margin
        self._train_rows = None       # lazy: chi doc khi can bootstrap

    # ---------------------------------------------------------------- ha tang
    def _predictor(self, cores, u_star):
        return FP.FeasibilityPredictor(self.frozen['mechanism'], cores, u_star,
                                       margin=self.u_margin, feature_cost=self.p2)

    def _train_rows_cached(self):
        if self._train_rows is None:
            if not os.path.isdir(self.train_dir):
                raise FileNotFoundError(
                    f'{self.train_dir} khong ton tai -- can de tinh khoang bat dinh (bootstrap tren du lieu huan luyen goc). '
                    'Neu chi can diem uoc luong, goi assess(..., bootstrap=False).')
            df = FP.load_runs(self.train_dir)
            train_max = self.frozen['params']['train_max_rps']
            self._train_rows = FP.select_train(df, train_max)
        return self._train_rows

    # ---------------------------------------------------------------- API chinh
    def assess(self, feature, scale=1.0, L_peak=200.0, k=None, model=None, bootstrap=True,
               n_bootstrap=BOOTSTRAP_B, seed=0):
        """feature: ten tinh nang da dat (khoa FEATURE_ARCHETYPE, vd 'promo') HOAC ten archetype THAT
        (khoa SOCKSHOP_CALL_CHAINS, vd 'LOGIN', 'REGISTER' -- dung duoc voi bat ky archetype nao ParserAgent
        tra ve, khong chi 8 tinh nang da co bang chung thuc nghiem); None cho baseline.
        k: dict boi so goi do duoc (vd tu probe_feature_chain.py); None -> gia dinh k=1 (P2).
        model: ep 'P2' hoac 'P3'; mac dinh tu dong ('P3' neu co k, nguoc lai 'P2')."""
        cores, cores_src, warn = read_live_cores(FP.SERVICES, self.docker_project)
        mode = model or ('P2' if k else 'P2')     # P3 = P2 + k do duoc; cung 1 nhanh code (mode='P2', k=...)
        u_star = self.frozen['params']['u_star']
        P = self._predictor(cores, u_star)

        point_r, point_node = P.breakpoint(mode=mode, feature=feature, scale=scale, k=k)
        u = P.utilization(L_peak, mode=mode, feature=feature, scale=scale, k=k)
        W = P.workloads(L_peak, mode=mode, feature=feature, scale=scale, k=k)

        if bootstrap:
            try:
                rs = self._bootstrap_breakpoints(cores, u_star, mode, feature, scale, k, n_bootstrap, seed)
                lo, hi = np.percentile(rs, CI[0]), np.percentile(rs, CI[1])
                r_med = float(np.median(rs))
            except FileNotFoundError as e:
                warn.append(str(e))
                r_med, lo, hi, n_bootstrap = point_r, point_r, point_r, 0
        else:
            r_med, lo, hi, n_bootstrap = point_r, point_r, point_r, 0

        bott = max(u, key=u.get)
        mu = u[bott]
        verdict = ('INFEASIBLE' if mu > u_star else
                  ('MARGINAL' if mu > u_star * (1 - self.u_margin) else 'FEASIBLE'))
        # CONG THROTTLING: han ngach cua node nghen quyet dinh bo du doan co hop le hay khong.
        # Duoi nguong da kiem chung, CFS throttling (khong phai bao hoa CPU) lam vo SLO -> bo du doan
        # nay khong ap dung duoc; tra UNDECIDED chu KHONG doan mot con so. Xem muc 5c.
        bott_cores = float(cores.get(bott, float('nan')))
        throttling_risk = bool(bott_cores < QUOTA_MIN_VALIDATED_CORES)
        if throttling_risk:
            verdict = 'UNDECIDED'
            warn.append(f"node nghen '{bott}' co han ngach {bott_cores:g} core < "
                        f"{QUOTA_MIN_VALIDATED_CORES:g} core -- vung CFS throttling, NGOAI pham vi da "
                        f"kiem chung (docs/DATA_FRAMEWORK.md muc 5c); khong phan quyet kha thi.")
        mech = self.frozen['mechanism']
        extrap = [s for s in P.scored if W[s] > mech[s]['w_max_train']]
        arch = FP.SOCKSHOP_CALL_CHAINS[FP.FEATURE_ARCHETYPE.get(feature, feature)] if feature else None
        n_calls = sum((k or {}).get(m, 1.0) for m in (arch['services'] if arch else []) if m != FP.GATEWAY)

        return FeasibilityVerdict(
            verdict=verdict, breakpoint_rps=round(r_med, 1), breakpoint_ci=(round(float(lo), 1), round(float(hi), 1)),
            bottleneck=bott, max_utilization=round(mu, 4),
            utilization_by_node={s: round(v, 4) for s, v in u.items()},
            extrapolating=bool(extrap), extrap_nodes=extrap,
            latency_risk=bool(n_calls >= LATENCY_RISK_N_CALLS),
            throttling_risk=throttling_risk, bottleneck_cores=round(bott_cores, 4),
            model=('P3' if k else 'P2'), k_source=('do bang probe' if k else 'gia dinh (k=1)'),
            cores_source=cores_src, n_bootstrap=n_bootstrap, warnings=warn)

    # ---------------------------------------------------------------- bootstrap
    def _bootstrap_breakpoints(self, cores, u_star, mode, feature, scale, k, B, seed):
        rows = self._train_rows_cached()
        rng = np.random.RandomState(seed)
        n = len(rows)
        out = np.empty(B)
        for i in range(B):
            idx = rng.randint(0, n, n)
            mech_b = FP.fit_mechanism(rows.iloc[idx])
            Pb = FP.FeasibilityPredictor(mech_b, cores, u_star, margin=self.u_margin, feature_cost=self.p2)
            out[i], _ = Pb.breakpoint(mode=mode, feature=feature, scale=scale, k=k)
        return out
