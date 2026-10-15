# -*- coding: utf-8 -*-
"""
HE A (CapacityAgent) _apply_chain_correction(): PHAN QUYET DUNG HON HAY SAI HON?
=================================================================================
Nhiem vu: `_apply_chain_correction()` (them vao CapacityAgent, src/agents/capacity_agent.py)
hien CHI duoc bien minh bang MAE CPU giam 25.2% (Wilcoxon p=0.0234, n=8 tinh nang, HOI CUU --
xem evaluate_chain_awareness.py / chain_correction_params.json). Nhung do rieng (xem ham
_check_evidence_sanity() o duoi) cho thay correction nay DOI PHAN QUYET SAFE/WARNING/CRITICAL
o 4/10 archetype (PLACE_ORDER, APPLY_PROMO_CODE, RECOMMEND_PRODUCTS: SAFE->CRITICAL;
TRACK_PACKAGE: SAFE->WARNING). "MAE giam" khong chung minh "phan quyet tot hon" -- day la
khoang trong ma file nay do.

KHO KHAN PHUONG PHAP (doc truoc khi doc code): He B (FeasibilityPredictor, P1/P2/P3) du bao
mot DIEM GAY R* (req/s) -- so truc tiep duoc voi khoang do [do_duoc_lo, hi]. He A KHONG du
bao R*: `assess_capacity()` tra ve status in {SAFE,WARNING,CRITICAL} cho MOT delta_pct can
thiep tai gateway, o ĐÚNG MOT tai nen (trung binh workload SS-TRAIN). Hai dai luong khac
BAN CHAT nhau. Cach BAC CAU duoc dung o day (xem _status_off_on_at_L phia duoi):

  CONG THUC CAU NOI (THAT, khong doan): voi mot archetype co call-chain va expected_delta_pct
  (D) trong CALL_CHAINS, va mot tai nen L (dai luong CUNG DON VI voi do_duoc_lo/hi cua He B --
  req/s TRUOC KHI co tinh nang), workload gateway DUOI CA HAI he khi co tinh nang la:
      W_fe(L) = L * (1 + D_scaled/100)        (D_scaled = D * scale, giong He B o mode P0/P1 --
                                                xem feasibility_predictor.py:workloads(), nhanh
                                                P0/P1: out['front-end'] = L*(1+delta))
  simulate_intervention(service, delta_pct) cua He A dinh nghia:
      target_wl = base_wl * (1 + delta_pct/100),   base_wl = trung binh front-end_workload SS-TRAIN (CO DINH)
  Giai delta_pct de target_wl == W_fe(L) mong muon:
      delta_pct_eff(L) = (W_fe(L) / base_wl - 1) * 100
  Day KHONG phai mot xap xi: target_wl la HAM TUYEN TINH cua delta_pct trong chinh code cua
  CapacityAgent, nen giai nguoc dung CHINH XAC (da KIEM: tai L=base_wl, delta_pct_eff == D_scaled,
  tai lap dung 10/10 phan quyet da do truoc o 'evidence section' -- xem _check_evidence_sanity()).

  He A KHONG co khai niem "chi phi dieu phoi gateway" (x*n_calls) cua He B mode P2/P3 -- dung
  DUNG cong thuc P0/P1 (W_fe=L*(1+D)) cho ca hai he, vi do la TOAN BO nhung gi delta_pct cua
  CapacityAgent THAT SU bieu dien. Dung cong thuc P2/P3 se gan cho He A mot kha nang no KHONG CO.

  HAI PHEP DO DUA RA, CA HAI deu that (khong fabricate R*):
  [A] QUYET DINH TAI DUNG TAI NEN DA DO (chinh, it gia dinh nhat): lay LUOI muc_tai (L) ma
      He B DA duoc danh gia san (papers/.../evaluate_verdicts.py:nap_quyet_dinh(), tai su dung
      100% -- khong viet lai luat loai 'dai nhap nhang' L<lo/L>=hi), roi tinh status cua He A
      (bat/tat correction) tai DUNG nhung L do. So du_kha_thi (He A) voi that_kha_thi (do that,
      tai su dung cot co san) -- giong HET phuong phap evaluate_verdicts.py dung cho He B, chi
      ap dung THEM cho He A. KHONG CAN gia dinh don dieu/bisection gi ca -- day la phep do
      CHAT nhat trong file nay.
  [B] DIEM GAY SUY RA TU HE A (R*_A, phu, de doi chieu don vi voi bang p3_evaluation.csv):
      quet luoi L (doi so mu tu thap len cao) + bisection, GIA DINH status don dieu khong giam
      theo L (co so: moi co che CPU trong DAG la LinearRegression(positive=True), xem Menh de 2
      trong capacity_agent.py) -- NHUNG gia dinh nay DUOC KIEM LAI bang so (khong chi gia dinh)
      qua toan bo diem da quet, va BAO CAO RO neu vi pham. R*_A duoc phan loai MUON/SOM/TRONG
      khoang dung CHINH bo nhan cua p3_evaluation.csv (cong thuc sai_so_% = (R*-lo)/lo*100, suy
      nguoc tu du lieu p3_evaluation.csv va xac nhan khop tren nhieu dong truoc khi dung).

  Ca hai phep do dung CHUNG mot dinh nghia "kha thi gia" (false-feasible): mo hinh bao
  SAFE/kha-thi tai mot tai ma do THAT la khong kha thi (L>=hi, hoac status!=CRITICAL tai do).

DU LIEU / NGUON (chi doc, khong sua file nao trong src/ hay CSV/JSON da co):
  - CapacityAgent(llm=None, data_dir='data/raw/SS-TRAIN', graph_path='src/graph/sockshop_agent_graph.json',
    auto_train=True)  -- He A, nhanh heuristic (llm=None), KHONG goi ParserAgent/LLM thuc.
  - src/scm/request_router.py: CALL_CHAINS (= SOCKSHOP_CALL_CHAINS) -- 10 archetype, moi
    archetype co 'services' (chain) va 'expected_delta_pct' (D).
  - src/scm/feasibility_predictor.py: FEATURE_ARCHETYPE -- bang anh xa 18 ten tinh nang SS-LIMITS
    (express, track, cartsum, recs, quickadd, promo, review, browse, + 10 tinh nang vong 2) sang
    archetype -- dung lai NGUYEN BANG nay (khong tu dat lai), nhu evaluate_chain_awareness.py da lam.
  - data/processed/frozen/p3_evaluation.csv -- thanh tich He B: 27 cell (feature x scale) x 4
    model (P1 (k=1), P2 (c,x; k=1), P3 (k do duoc), P3+chain), cot do_duoc_lo/hi/phan_loai.
  - evaluate_verdicts.py:nap_quyet_dinh() -- luoi quyet dinh SACH (L, that_kha_thi) cua He B,
    doc tu data/processed/frozen/predictions_frozen_RE2*.json, da LOAI dai nhap nhang. TAI
    SU DUNG truc tiep (import), khong viet lai.

GIOI HAN DA BIET (khong che):
  - Luoi muc_tai cua predictor P3 (frozen) CHI phu 14/27 cell (7 tinh nang vong-2 x1/x2 + browse
    x1/x2 + cac tinh nang don x1 -- xem in luc chay); 13 cell con lai (tinh nang main+indep:
    promo/recs/track/review/cartsum/quickadd/express) KHONG co luoi P3 dong bang rieng (P3 trung
    P2 khi khong co k do duoc tu probe -- xac nhan: R*_pred P3==P2 tren toan bo 13 cell nay trong
    p3_evaluation.csv). Phan A dung predictor P2 (phu 27/27 cell) lam doi chung grid chinh, va
    P3 THEM VAO cho 14/27 cell co san -- KHONG suy dien P3 cho 13 cell con thieu.
  - `saturated_services` cua CapacityAgent LUON chua 'payment.latency'/'shipping.latency' voi
    status NO_OBSERVED_RANGE (P50==P99==0.0025s tren SS-TRAIN -- do gan nhu hang so / luong hoa
    dong ho do). Da XAC MINH (doc code summarize_headroom()): NO_OBSERVED_RANGE KHONG nang muc
    canh bao (khong doi status tong) -- chi gay nhieu khi HIEN THI danh sach saturated_services.
    File nay LOAI cac dong co '[NO_OBSERVED_RANGE]' khoi _loc_canh_bao_hien_thi() truoc khi in/
    ghi cot `canh_bao_mau` -- cot `status` (SAFE/WARNING/CRITICAL) KHONG bi anh huong (da doc
    code de xac nhan, khong doan).
  - delta_pct_eff(L) co the am (L nho) hoac rat lon (L lon) -- vuot xa [5,50] ma Layer 3/G2
    (parser_agent.py) se kep trong san pham THAT. Day la CHU Y: cau hoi o day la "co che cua
    CapacityAgent TU NO co don dieu/dung huong khong tai moi L", khong phai "request nao se lot
    qua G2". simulate_intervention() KHONG tu kep delta_pct (kep nam o parser_agent.py, lop
    KHONG duoc dung trong file nay theo yeu cau -- khong goi LLM/ParserAgent).
  - R*_A (Phep do B) chi co nghia khi tim duoc trong pham vi quet (L_FLOOR..L_CAP). Neu khong
    tim duoc (status SAFE xuyen suot toi L_CAP, hoac da khong-SAFE ngay tai L_FLOOR), cot
    `R_star` de NaN kem `note` giai thich -- KHONG suy dien/noi suy mot con so.

CHAY LAI:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
        papers/p1_du_phong/experiments/model_eval/evaluate_verdict_level_heA.py

RA:
    data/processed/scm_results/heA_verdict_decision_matrix.csv   (Phep do A, 1 dong/[cell,muc_tai])
    data/processed/scm_results/heA_breakpoint_comparison.csv    (Phep do B, 1 dong/[cell,cau_hinh])
    + bang tong hop in ra man hinh (so ca kha-thi-gia/qua-bao-thu moi cau hinh).
"""

import os
import re
import sys
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- duong dan goc
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = _HERE
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))
sys.path.insert(0, _HERE)  # de import evaluate_verdicts.py (cung thu muc)

from capacity_agent import CapacityAgent  # noqa: E402
from request_router import CALL_CHAINS  # noqa: E402  (= SOCKSHOP_CALL_CHAINS, nguon su that)
from feasibility_predictor import FEATURE_ARCHETYPE  # noqa: E402  (bang anh xa SS-LIMITS -> archetype)
import evaluate_verdicts as ev_verdicts  # noqa: E402  (tai su dung nap_quyet_dinh(), cp95())

FROZEN = os.path.join(BASE_DIR, 'data', 'processed', 'frozen')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

SS_TRAIN = os.path.join(BASE_DIR, 'data', 'raw', 'SS-TRAIN')
GRAPH_PATH = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')

STATUS_ORDINAL = {'SAFE': 0, 'WARNING': 1, 'CRITICAL': 2}


# ======================================================================
# 0. BANG ANH XA cell -> archetype (dung lai FEATURE_ARCHETYPE + CALL_CHAINS,
#    KHONG tu dat lai nhu evaluate_chain_awareness.py da lam)
# ======================================================================
def phan_tich_cell(cell: str):
    """'cartsumx2' -> (feature='cartsum', scale=2, archetype='VIEW_CART', spec=CALL_CHAINS[...])."""
    m = re.match(r'^(.+)x(\d+)$', cell)
    if not m:
        raise ValueError(f"khong doc duoc cell '{cell}' (ky vong dang '<feature>x<scale>')")
    feature, scale = m.group(1), int(m.group(2))
    if feature not in FEATURE_ARCHETYPE:
        raise KeyError(f"tinh nang '{feature}' (tu cell '{cell}') khong co trong FEATURE_ARCHETYPE "
                        f"(feasibility_predictor.py) -- khong the suy archetype")
    archetype = FEATURE_ARCHETYPE[feature]
    if archetype not in CALL_CHAINS:
        raise KeyError(f"archetype '{archetype}' khong co trong CALL_CHAINS (request_router.py)")
    spec = CALL_CHAINS[archetype]
    return feature, scale, archetype, spec


def nap_27_cell():
    """Doc 27 cell tu p3_evaluation.csv (nguon do THAT cua He B), kem (lo,hi) va suy archetype."""
    E = pd.read_csv(os.path.join(FROZEN, 'p3_evaluation.csv'))
    rows = []
    for r in E.drop_duplicates('cell').itertuples():
        feature, scale, archetype, spec = phan_tich_cell(r.cell)
        rows.append(dict(
            cell=r.cell, feature=feature, scale=scale, archetype=archetype,
            chain=tuple(spec['services']), expected_delta_pct=spec['expected_delta_pct'],
            delta_scaled_pct=spec['expected_delta_pct'] * scale,
            do_duoc_lo=float(r.do_duoc_lo), hi=float(r.hi),
        ))
    return pd.DataFrame(rows)


# ======================================================================
# 1. CAU NOI: status He A (bat/tat correction) TAI MOT TAI NEN L
# ======================================================================
class HeABridge:
    """Boc CapacityAgent + cong thuc bac cau (xem docstring module). Cache theo
    (archetype, round(L,6)) vi NHIEU cell dung CHUNG archetype (vd browsex1/x2,
    catsearchx1, relatedx1 deu la GET_CATALOGUE) nhung chain/Delta_scaled khac theo
    scale -- cache key phai gom CA delta_scaled_pct, khong chi archetype."""

    def __init__(self, agent: CapacityAgent):
        self.agent = agent
        self.base_wl = float(agent.df_baseline['front-end_workload'].mean())
        self._orig_cc = agent._chain_correction  # tham so correction DA HOC (khong sua)
        self._cache = {}
        self.n_calls = 0

    def _delta_pct_eff(self, L: float, delta_scaled_pct: float) -> float:
        target_wl = L * (1.0 + delta_scaled_pct / 100.0)
        return (target_wl / self.base_wl - 1.0) * 100.0

    def status_off_on(self, L: float, delta_scaled_pct: float, chain: tuple):
        """Tra ve dict: status_off, status_on, lines_off, lines_on (da loc NO_OBSERVED_RANGE),
        ood_confidence -- CA HAI che do (bat/tat correction) tu DUNG MOT lan goi
        simulate_intervention() (correction chi la buoc xu ly SAU, khong can goi lai DAG)."""
        key = (delta_scaled_pct, chain, round(float(L), 6))
        if key in self._cache:
            return self._cache[key]
        delta_pct_eff = self._delta_pct_eff(L, delta_scaled_pct)
        sim = self.agent.simulate_intervention('front-end', delta_pct_eff)
        self.n_calls += 1

        headroom_off = self.agent.compute_headroom(sim)
        status_off, lines_off = self.agent.summarize_headroom(headroom_off)

        sim_on = self.agent._apply_chain_correction(sim, set(chain))
        headroom_on = self.agent.compute_headroom(sim_on)
        status_on, lines_on = self.agent.summarize_headroom(headroom_on)

        out = dict(
            status_off=status_off, status_on=status_on,
            lines_off=_loc_canh_bao_hien_thi(lines_off),
            lines_on=_loc_canh_bao_hien_thi(lines_on),
            ood_confidence=self.agent._last_ood_confidence,
            delta_pct_eff=round(delta_pct_eff, 4),
        )
        self._cache[key] = out
        return out


def _loc_canh_bao_hien_thi(lines):
    """Loai cac dong canh bao '[NO_OBSERVED_RANGE]' KHOI HIEN THI (xem docstring module --
    da xac nhan qua code summarize_headroom() rang trang thai nay KHONG nang muc canh bao,
    nen loai no khoi danh sach hien thi khong lam mat thong tin quyet dinh)."""
    return [l for l in lines if '[NO_OBSERVED_RANGE]' not in l]


def feasible_long(status: str) -> bool:
    """Doc 'long': kha thi <=> status != CRITICAL (WARNING van tinh la kha thi) --
    guong voi evaluate_verdicts.py: kha thi <=> verdict != INFEASIBLE (MARGINAL=kha thi)."""
    return status != 'CRITICAL'


def feasible_chat(status: str) -> bool:
    """Doc 'chat': kha thi <=> status == SAFE -- guong voi verdict == FEASIBLE."""
    return status == 'SAFE'


# ======================================================================
# 2. KIEM TRA DOI CHUNG VOI 'EVIDENCE' DA DO TRUOC (khong do lai, chi xac nhan cau noi DUNG)
# ======================================================================
_EVIDENCE_AT_BASE_WL = {
    'GET_CATALOGUE': ('SAFE', 'SAFE'), 'ADD_TO_CART': ('SAFE', 'SAFE'),
    'VIEW_CART': ('SAFE', 'SAFE'), 'REGISTER': ('SAFE', 'SAFE'), 'LOGIN': ('SAFE', 'SAFE'),
    'PLACE_ORDER': ('SAFE', 'CRITICAL'), 'APPLY_PROMO_CODE': ('SAFE', 'CRITICAL'),
    'RECOMMEND_PRODUCTS': ('SAFE', 'CRITICAL'), 'TRACK_PACKAGE': ('SAFE', 'WARNING'),
    'WRITE_PRODUCT_REVIEW': ('SAFE', 'SAFE'),
}


def _check_evidence_sanity(bridge: HeABridge) -> bool:
    """Tai L = base_wl (trung binh workload SS-TRAIN), delta_pct_eff phai == expected_delta_pct
    cua archetype (scale=1) VA phan quyet phai khop CHINH XAC voi 10 archetype da do truoc khi
    viet file nay (xem phan 'BANG CHUNG DA DO' trong yeu cau nhiem vu). Day la unit-test cho
    chinh cong thuc bac cau -- neu sai o day, moi so lieu phia duoi deu vo nghia."""
    ok = True
    for archetype, (exp_off, exp_on) in _EVIDENCE_AT_BASE_WL.items():
        spec = CALL_CHAINS[archetype]
        r = bridge.status_off_on(bridge.base_wl, spec['expected_delta_pct'], tuple(spec['services']))
        if abs(r['delta_pct_eff'] - spec['expected_delta_pct']) > 1e-6:
            print(f"  [CANH BAO] {archetype}: delta_pct_eff={r['delta_pct_eff']} != "
                  f"expected_delta_pct={spec['expected_delta_pct']} (cong thuc bac cau SAI)")
            ok = False
        if r['status_off'] != exp_off or r['status_on'] != exp_on:
            print(f"  [CANH BAO] {archetype}: tai lap KHONG khop evidence -- "
                  f"do duoc off={r['status_off']}/on={r['status_on']}, "
                  f"evidence off={exp_off}/on={exp_on}")
            ok = False
    print(f"  [Kiem tra doi chung evidence] {'KHOP 10/10' if ok else 'CO SAI LECH -- xem canh bao tren'}")
    return ok


# ======================================================================
# PHAN A: QUYET DINH TAI DUNG LUOI TAI MA HE B DA DUOC DANH GIA
#   (tai su dung evaluate_verdicts.nap_quyet_dinh() -- khong viet lai luat loai
#   dai nhap nhang / khong tu tinh that_kha_thi)
# ======================================================================
def phan_A_quyet_dinh_tai_luoi_do(bridge: HeABridge, cell_df: pd.DataFrame) -> pd.DataFrame:
    D = ev_verdicts.nap_quyet_dinh()  # 1297 dong: (cell, predictor, muc_tai) -> that_kha_thi, verdict
    D_ref = D[D.predictor.isin(['P0', 'P1', 'P2', 'P3'])].copy()

    spec_by_cell = {r.cell: r for r in cell_df.itertuples()}
    rows = []
    for (cell, muc_tai), g in D_ref.groupby(['cell', 'muc_tai']):
        if cell not in spec_by_cell:
            continue  # khong nen xay ra (D da duoc loc theo cell cua p3_evaluation.csv), phong thu
        spec = spec_by_cell[cell]
        that_vals = g['that_kha_thi'].unique()
        assert len(that_vals) == 1, (
            f"that_kha_thi khong nhat quan giua predictor cho cung (cell={cell}, muc_tai={muc_tai}) "
            f"-- vi pham gia dinh 'ground truth chi phu thuoc (L,lo,hi)'")
        that_kha_thi = bool(that_vals[0])

        r = bridge.status_off_on(float(muc_tai), spec.delta_scaled_pct, spec.chain)

        row = dict(cell=cell, archetype=spec.archetype, muc_tai=float(muc_tai),
                    do_duoc_lo=spec.do_duoc_lo, hi=spec.hi, that_kha_thi=that_kha_thi,
                    heA_off_status=r['status_off'], heA_on_status=r['status_on'],
                    heA_off_canh_bao=';'.join(r['lines_off']) or '-',
                    heA_on_canh_bao=';'.join(r['lines_on']) or '-',
                    ood_confidence=r['ood_confidence'], delta_pct_eff=r['delta_pct_eff'])
        for suf, st in (('off', r['status_off']), ('on', r['status_on'])):
            row[f'heA_{suf}_kha_thi_long'] = feasible_long(st)
            row[f'heA_{suf}_kha_thi_chat'] = feasible_chat(st)

        for _, rowB in g.iterrows():
            pr = rowB['predictor']
            v = str(rowB['verdict']).upper()
            row[f'heB_{pr}_verdict'] = v
            row[f'heB_{pr}_kha_thi_long'] = (v != 'INFEASIBLE')
            row[f'heB_{pr}_kha_thi_chat'] = (v == 'FEASIBLE')
        rows.append(row)
    return pd.DataFrame(rows)


def _cp95(k, n):
    return ev_verdicts.cp95(k, n)


def tom_tat_phan_A(A: pd.DataFrame):
    """Confusion-matrix style: so ca KHA THI GIA (false-feasible, du bao kha thi tai
    tai THAT SU khong kha thi) va SOM (qua bao thu) cho MOI cau hinh, 2 cach doc
    (long/chat), tren CUNG tap (cell, muc_tai) -- so sanh truc tiep duoc giua
    cac cau hinh vi mau so (n) giong nhau theo tung predictor Hệ B co san."""
    configs_A = [('heA_off', 'Hệ A - KHONG correction'), ('heA_on', 'Hệ A - CO correction')]
    print(f"\n{'='*112}\n  PHAN A -- QUYET DINH TAI DUNG {len(A)} (cell, muc_tai) MA He B DA DUOC DANH GIA\n{'='*112}")
    for doc in ('long', 'chat'):
        print(f"\n  -- cach doc '{doc}' ({'MARGINAL/WARNING tinh la kha thi' if doc == 'long' else 'CHI SAFE/FEASIBLE moi tinh la kha thi'}) --")
        print(f"  {'cau hinh':28s} {'n':>5s} {'dung':>7s} {'KHA THI GIA':>22s} {'bo thieu (qua bao thu)':>24s}")
        for key, label in configs_A:
            col = f'{key}_kha_thi_{doc}'
            du = A[col]
            that = A['that_kha_thi']
            dung = (du == that).mean() * 100
            kt = ~that  # thuc su KHONG kha thi
            fp = int((du & kt).sum())
            n_kt = int(kt.sum())
            lo, hi = _cp95(fp, n_kt)
            fn = int((~du & that).sum())
            n_t = int(that.sum())
            print(f"  {label:28s} {len(A):5d} {dung:6.1f}% {fp:4d}/{n_kt:<4d}={100*fp/max(n_kt,1):5.1f}%"
                  f" [{lo:4.1f},{hi:4.1f}]%   {fn:4d}/{n_t:<4d}={100*fn/max(n_t,1):5.1f}%")
        for pr in ('P0', 'P1', 'P2', 'P3'):
            col = f'heB_{pr}_kha_thi_{doc}'
            if col not in A.columns:
                continue
            sub = A[A[col].notna()]
            if sub.empty:
                continue
            du, that = sub[col], sub['that_kha_thi']
            dung = (du == that).mean() * 100
            kt = ~that
            fp = int((du & kt).sum())
            n_kt = int(kt.sum())
            lo, hi = _cp95(fp, n_kt)
            fn = int((~du & that).sum())
            n_t = int(that.sum())
            print(f"  {'Hệ B - ' + pr:28s} {len(sub):5d} {dung:6.1f}% {fp:4d}/{n_kt:<4d}={100*fp/max(n_kt,1):5.1f}%"
                  f" [{lo:4.1f},{hi:4.1f}]%   {fn:4d}/{n_t:<4d}={100*fn/max(n_t,1):5.1f}%")


# ======================================================================
# PHAN B: DIEM GAY R*_A SUY TU HE A (bisection + kiem don dieu BANG SO)
# ======================================================================
L_FLOOR = 1.0
L_CAP = 2.0e5          # 200k req/s -- vuot xa moi do that (hi max = 240); xem docstring
GROWTH = 2.0
MAX_EXPAND = 30
BISECT_ITERS = 50
REL_TOL = 1e-5


def _tim_khung(s, lo0, lo_ok_fn, hi_bad_fn, L_cap):
    """Mo rong tu lo0 (da biet lo_ok_fn(status(lo0))==True) len theo GROWTH toi khi
    gap diem co hi_bad_fn(status)==True, hoac het L_cap. Tra ve (lo,hi) voi hi=None
    neu khong gap trong pham vi."""
    lo, hi, L = lo0, None, lo0
    for _ in range(MAX_EXPAND):
        L = min(L * GROWTH, L_cap)
        st = s(L)
        if hi_bad_fn(st):
            hi = L
            break
        lo = L
        if L >= L_cap:
            break
    return lo, hi


def _bisect(s, lo, hi, la_ben_hi_fn):
    """Bisection chuan: giu bat bien lo=ben 'tot', hi=ben 'xau' (la_ben_hi_fn=True)."""
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        st = s(mid)
        if la_ben_hi_fn(st):
            hi = mid
        else:
            lo = mid
        if (hi - lo) < REL_TOL * max(hi, 1.0):
            break
    return lo, hi


def _tim_diem_gay(eval_status, L_floor=L_FLOOR, L_cap=L_CAP):
    """eval_status(L) -> 'SAFE'|'WARNING'|'CRITICAL'. Tra ve dict:
      R_long  : sup{L : status==SAFE}        (None neu khong xac dinh duoc trong [L_floor, L_cap])
      R_chat  : sup{L : status!=CRITICAL}
      don_dieu_ok: True/False (kiem BANG SO tren CHINH cac diem da quet trong lan goi nay,
                   khong gia dinh suong -- Menh de 2 cua CapacityAgent chi dam bao don dieu
                   cho CO CHE CPU/mem truoc correction, KHONG tu dong suy ra cho `status`
                   tong hop SAU headroom+correction, nen PHAI do lai o day)
      vi_pham: danh sach (L_a,status_a,L_b,status_b) nguoc thu tu neu co
      n_calls: so lan goi eval_status (= so lan goi simulate_intervention(), tru cache-hit)

    3 truong hop tai chinh san L_floor (status(L_floor)):
      SAFE     -> tim R_long (mo rong + bisect) roi R_chat (tiep tu R_long)
      WARNING  -> R_long <= L_floor (khong uoc luong duoc, chi ghi chu); tim R_chat tu L_floor
      CRITICAL -> CA R_long VA R_chat <= L_floor (khong uoc luong duoc, chi ghi chu)
    """
    path = []

    def s(L):
        st = eval_status(L)
        path.append((L, st))
        return st

    out = {'R_long': None, 'R_long_note': '', 'R_chat': None, 'R_chat_note': ''}
    duoi_san = (f"status da KHONG SAFE/CRITICAL (tuy truong hop) ngay tai san "
                f"L_floor={L_floor:g} -- diem gay <= {L_floor:g}, khong uoc luong duoc "
                f"trong pham vi quet [{L_floor:g}, {L_cap:g}]")

    st0 = s(L_floor)
    if st0 == 'CRITICAL':
        out['R_long_note'] = duoi_san
        out['R_chat_note'] = duoi_san
    else:
        if st0 == 'WARNING':
            out['R_long_note'] = duoi_san
            lo_long, hi_long = L_floor, L_floor   # khong co khung SAFE->khong-SAFE o tren san
        else:  # st0 == 'SAFE'
            lo_long, hi_long = _tim_khung(s, L_floor, lambda st: st == 'SAFE',
                                           lambda st: st != 'SAFE', L_cap)
            if hi_long is None:
                out['R_long_note'] = (f"status la SAFE xuyen suot toi L_cap={L_cap:g} "
                                       f"-- khong tim duoc diem gay trong pham vi quet")
            else:
                lo_long, hi_long = _bisect(s, lo_long, hi_long, lambda st: st != 'SAFE')
                out['R_long'] = 0.5 * (lo_long + hi_long)
                out['R_long_bracket'] = (lo_long, hi_long)

        # R_chat: tiep tuc tu diem cao nhat da biet la != CRITICAL (hi_long neu co, neu
        # khong thi L_floor -- vi WARNING cung != CRITICAL)
        diem_bat_dau = hi_long if (st0 == 'SAFE' and out['R_long'] is not None) else L_floor
        st_tai_do = s(diem_bat_dau)
        if st_tai_do == 'CRITICAL':
            lo_chat, hi_chat = diem_bat_dau, diem_bat_dau  # se duoc _tim_khung mo rong xuong? khong can:
            # da CRITICAL ngay tai diem_bat_dau -- khung la (diem truoc do, diem_bat_dau).
            # Dung lai gia tri da co (lo_long,hi_long) neu co, neu khong thi khong xac dinh them.
            if st0 == 'SAFE' and out['R_long'] is not None:
                lo_chat, hi_chat = lo_long, hi_long
            else:
                out['R_chat_note'] = duoi_san
                lo_chat = hi_chat = None
        else:
            lo_chat, hi_chat = _tim_khung(s, diem_bat_dau, lambda st: st != 'CRITICAL',
                                           lambda st: st == 'CRITICAL', L_cap)
            if hi_chat is None:
                out['R_chat_note'] = f"status khong bao gio CRITICAL toi L_cap={L_cap:g}"
        if hi_chat is not None:
            lo_chat, hi_chat = _bisect(s, lo_chat, hi_chat, lambda st: st == 'CRITICAL')
            out['R_chat'] = 0.5 * (lo_chat + hi_chat)
            out['R_chat_bracket'] = (lo_chat, hi_chat)

    # --- kiem don dieu BANG SO tren toan bo diem da quet trong lan goi nay (khong gia dinh)
    path_sorted = sorted(set(path))
    vi_pham = []
    for (La, sa), (Lb, sb) in zip(path_sorted, path_sorted[1:]):
        if STATUS_ORDINAL[sb] < STATUS_ORDINAL[sa]:
            vi_pham.append((round(La, 3), sa, round(Lb, 3), sb))
    out['don_dieu_ok'] = (len(vi_pham) == 0)
    out['vi_pham'] = vi_pham
    out['n_calls'] = len(path)
    return out


def phan_B_diem_gay(bridge: HeABridge, cell_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in cell_df.itertuples():
        for correction_on, tag in ((False, 'off'), (True, 'on')):
            def eval_status(L, _r=r, _on=correction_on):
                res = bridge.status_off_on(L, _r.delta_scaled_pct, _r.chain)
                return res['status_on'] if _on else res['status_off']

            kq = _tim_diem_gay(eval_status)
            for doc, key in (('long', 'R_long'), ('chat', 'R_chat')):
                R = kq.get(key)
                note = kq.get(f'{key}_note', '')
                rows.append(dict(
                    cell=r.cell, feature=r.feature, scale=r.scale, archetype=r.archetype,
                    config=f'heA_{tag}_{doc}', system='A', correction=tag, doc_verdict=doc,
                    R_star=round(R, 2) if R is not None else np.nan,
                    do_duoc_lo=r.do_duoc_lo, hi=r.hi,
                    sai_so_pct=round((R - r.do_duoc_lo) / r.do_duoc_lo * 100, 1) if R else np.nan,
                    phan_loai=(_phan_loai(R, r.do_duoc_lo, r.hi) if R is not None else 'KHONG_XAC_DINH'),
                    don_dieu_ok=kq['don_dieu_ok'], n_vi_pham_don_dieu=len(kq['vi_pham']),
                    n_calls_scm=kq['n_calls'], note=note,
                ))
    return pd.DataFrame(rows)


def _phan_loai(R, lo, hi):
    """DUNG CHINH xi quyet cua p3_evaluation.csv (suy nguoc tu du lieu, xac nhan khop
    tren nhieu dong truoc khi dung -- xem docstring module): R>hi -> MUON (kha thi gia,
    He noi kha thi lau hon thuc te); R<lo -> SOM (an toan, qua bao thu); con lai TRONG khoang."""
    if R > hi:
        return 'MUON (kha thi gia)'
    if R < lo:
        return 'SOM (an toan)'
    return 'TRONG khoang'


def gop_voi_he_B_diem_gay(cell_df: pd.DataFrame) -> pd.DataFrame:
    """Rows cua p3_evaluation.csv (4 model He B) o CUNG SCHEMA voi phan_B_diem_gay(),
    de ghep chung 1 bang 'config x cell' -- khong tinh lai, chi doi ten cot."""
    E = pd.read_csv(os.path.join(FROZEN, 'p3_evaluation.csv'))
    spec_by_cell = {r.cell: r for r in cell_df.itertuples()}
    rows = []
    for _, rB in E.iterrows():
        cell = rB['cell']
        if cell not in spec_by_cell:
            continue
        sp = spec_by_cell[cell]
        rows.append(dict(
            cell=cell, feature=sp.feature, scale=sp.scale, archetype=sp.archetype,
            config=f"heB_{rB['model']}", system='B', correction='n/a', doc_verdict='n/a',
            R_star=rB['R*_pred'], do_duoc_lo=rB['do_duoc_lo'], hi=rB['hi'],
            sai_so_pct=rB['sai_so_%'], phan_loai=rB['phan_loai'],
            don_dieu_ok=np.nan, n_vi_pham_don_dieu=np.nan, n_calls_scm=np.nan,
            note='tu p3_evaluation.csv (khong tinh lai)',
        ))
    return pd.DataFrame(rows)


def tom_tat_phan_B(B_all: pd.DataFrame):
    print(f"\n{'='*112}\n  PHAN B -- DIEM GAY R*_A SUY RA (bisection) SO VOI KHOANG DO VA R*_pred CUA He B\n{'='*112}")
    print(f"  {'cau hinh':26s} {'n cell':>7s} {'MUON (kha thi gia)':>20s} {'SOM (an toan)':>16s} "
          f"{'TRONG khoang':>14s} {'khong xac dinh':>15s}")
    for cfg, g in B_all.groupby('config', sort=False):
        n = len(g)
        vc = g['phan_loai'].value_counts()
        muon = int(vc.get('MUON (kha thi gia)', 0))
        som = int(vc.get('SOM (an toan)', 0))
        trong = int(vc.get('TRONG khoang', 0))
        kxd = int(vc.get('KHONG_XAC_DINH', 0))
        print(f"  {cfg:26s} {n:7d} {muon:9d} ({100*muon/n:4.1f}%)   {som:5d} ({100*som/n:4.1f}%)  "
              f"{trong:5d} ({100*trong/n:4.1f}%)  {kxd:5d}")
    vi_pham = B_all[(B_all['system'] == 'A') & (B_all['don_dieu_ok'] == False)]  # noqa: E712
    if len(vi_pham):
        print(f"\n  [CANH BAO DON DIEU] {len(vi_pham)} cau hinh VI PHAM don dieu so voi tai L "
              f"(status khong tang khi L tang) -- xem cot n_vi_pham_don_dieu, R*_A o do la GAN DUNG:")
        print(vi_pham[['cell', 'config', 'n_vi_pham_don_dieu']].to_string(index=False))
    else:
        print(f"\n  [Kiem don dieu] KHONG co vi pham don dieu tren bat ky diem da quet nao (He A, {len(B_all[B_all.system=='A'])} cau hinh).")
    kxd = B_all[(B_all['system'] == 'A') & (B_all['phan_loai'] == 'KHONG_XAC_DINH')]
    if len(kxd):
        print(f"\n  [KHONG XAC DINH -- {len(kxd)} cau hinh] diem gay nam ngoai pham vi quet [{L_FLOOR:g}, {L_CAP:g}] req/s:")
        print(kxd[['cell', 'config', 'note']].to_string(index=False))


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("Khoi tao CapacityAgent (llm=None, data_dir=SS-TRAIN) -- KHONG goi LLM/ParserAgent...")
    agent = CapacityAgent(llm=None, data_dir=SS_TRAIN, graph_path=GRAPH_PATH, auto_train=True)
    bridge = HeABridge(agent)

    print("\nKiem tra cong thuc bac cau doi chung voi 10 phan quyet da do truoc (scale=1, L=base_wl):")
    _check_evidence_sanity(bridge)

    cell_df = nap_27_cell()
    print(f"\n{len(cell_df)} cell (feature x scale) tu p3_evaluation.csv, "
          f"{cell_df['archetype'].nunique()} archetype khac nhau.")

    print("\n--- PHAN A: quyet dinh tai dung luoi tai ma He B da duoc danh gia ---")
    A = phan_A_quyet_dinh_tai_luoi_do(bridge, cell_df)
    out_A = os.path.join(OUT_DIR, 'heA_verdict_decision_matrix.csv')
    A.to_csv(out_A, index=False)
    print(f"[OK] {len(A)} dong (cell x muc_tai) -> {out_A}")
    tom_tat_phan_A(A)

    print("\n--- PHAN B: diem gay R*_A suy ra (bisection + kiem don dieu) ---")
    B_heA = phan_B_diem_gay(bridge, cell_df)
    B_heB = gop_voi_he_B_diem_gay(cell_df)
    B_all = pd.concat([B_heA, B_heB], ignore_index=True)
    out_B = os.path.join(OUT_DIR, 'heA_breakpoint_comparison.csv')
    B_all.to_csv(out_B, index=False)
    print(f"[OK] {len(B_all)} dong (cell x cau_hinh) -> {out_B}")
    print(f"     (tong {bridge.n_calls} lan goi simulate_intervention() thuc su, da cache theo "
          f"(archetype,delta_scaled,L))")
    tom_tat_phan_B(B_all)

    print(f"\n{'='*112}\n  HET. Xem bao cao cuoi (tin nhan tra ve) de biet ket luan giu/go correction.\n{'='*112}")


if __name__ == '__main__':
    main()
