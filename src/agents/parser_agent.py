# -*- coding: utf-8 -*-
"""
ParserAgent — Requirement Interpreter
======================================
Phan tich yeu cau tinh nang moi -> injection_delta_pct + core_services.

classify_request() van chay truoc de sinh mot goi y rule-based (dung lam
neo/fallback khi LLM loi, va de tinh similarity_score), nhung KHONG con
duong tat bo qua LLM: MOI requirement, ke ca khi similarity keyword da ro,
deu di qua LLM full-parse (bao gom ca tu-khai-bao is_customer_facing_feature
o buoc 0). Ban truoc day co "fast-path" bo qua LLM khi similarity >= 0.6 --
da go bo sau khi phat hien 8/11 prompt trong bo test doi khang Scope Gate
tu thiet ke de co du khop tu khoa VOI MOT archetype SAI van vuot nguong nay,
khien Layer 1 khong bao gio duoc thuc thi cho chung (xem docstring cua
parse()).

Guards (bat buoc, kiem tra TRUOC khi tra ket qua):
  G1: injection_service phai la gateway (in-degree=0 tu do thi)
  G2: injection_delta_pct phai trong [MIN_DELTA, MAX_DELTA]
  G3: adjustment phai trong [-MAX_ADJ, +MAX_ADJ], vuot -> fallback ve anchor
  G4: core_services chi chua service ton tai trong graph
  G5: low similarity -> siet adjustment = 0
  G6: mot diem so logistic tong hop 4 tin hieu (keyword overlap, similarity
      lua chon cua LLM, similarity rule-based, tu-khai-bao is_customer_facing
      _feature) duoc so voi HAI nguong hieu chinh bang split-conformal
      (_SCOPE_GATE_CONFORMAL_TAU_PASS / _TAU_REFUSE): duoi nguong duoi ->
      cho qua, tren nguong tren -> REFUSED (is_out_of_scope=True), o giua ->
      needs_human_review=True thay vi tu dong quyet dinh. Thay the mot luat
      OR hai-tin-hieu truoc do (xem docs/paper_draft.tex, "Behavior at the
      Edge of the Declared Scope" de biet qua trinh dan den thiet ke nay).

Phu hop Q1 paper: "Grounded LLM Estimation anchored to empirical calibration table"
"""

import os
import re
import sys
import json
import unicodedata
import warnings
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

# Path setup
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.dirname(_AGENT_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, 'scm'))

from request_router import (CALL_CHAINS, classify_request, remove_accents,
                            default_priority_order)

# ============================================================
# CONSTANTS
# ============================================================
KNOWN_SERVICES = {'front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping'}

MAX_ADJUSTMENT_PCT  = 10.0   # LLM chi duoc +/-10% so voi anchor
MIN_DELTA_PCT       = 5.0    # san: khong co tinh nang nao duoi 5%
MAX_DELTA_PCT       = 50.0   # tran: LLM hallucinate neu vuot 50%
SIMILARITY_THRESHOLD = 0.6   # nguong match keyword: duoi nay -> siet adjustment=0

# ============================================================
# OUTPUT SCHEMA
# ============================================================
@dataclass
class ParsedRequirement:
    """
    Output chuan cua ParserAgent. Duoc CapacityAgent.get_metrics_for_service()
    va CapacityAgent.simulate_intervention() tieu thu (Fast/Accurate path).
    """
    # Core fields — dung cho ca 2 path
    request_type:         str         # "APPLY_PROMO_CODE" | "UNKNOWN"
    injection_service:    str         # gateway: node in-degree=0 tu do thi
    injection_delta_pct:  float       # da qua guard, san sang cho SCM
    core_services:        list        # services can sua code (LLM xac dinh)
    affected_services:    list        # blast radius (tu CALL_CHAINS)

    # Explainability — cho paper
    reasoning:            str
    confidence:           str         # "HIGH" | "MEDIUM" | "LOW" | "REFUSED"
    matched_template:     str         # template duoc chon lam anchor
    template_delta:       float       # anchor goc (chua dieu chinh)
    adjustment:           float       # delta_actual - template_delta (sau clamp)
    similarity_score:     float       # keyword match score [0, 1]
    llm_was_called:       bool        # True neu goi LLM (cho efficiency metric)
    is_out_of_scope:      bool = False  # G6: True neu KHONG archetype nao trung khop —
                                         # injection_delta_pct van co gia tri (de khong vo
                                         # pipeline) nhung KHONG duoc coi la dang tin cay;
                                         # noi tieu thu (vd report generation) phai kiem tra
                                         # co nay va tu choi dua ra phan quyet dinh luong.
    needs_human_review:   bool = False  # HITL: True neu Scope Gate khong du tin cay de tu
                                         # quyet dinh refuse hay cho qua (vd noi dung thu dong/
                                         # tinh, hoac 2 tin hieu bat dong) -- khac is_out_of_scope
                                         # o cho KHONG tu dong refuse, ma can nguoi xac nhan
                                         # truoc khi he thong dua ra phan quyet dinh luong.
    scope_gate_score:      float = None  # P(out-of-scope) tu bo tinh diem conformal-calibrated
                                         # dung de quyet dinh hard_refuse/needs_human_review
                                         # o tren (xem _scope_gate_conformal_score).
    scope_gate_raw_features: dict = None  # {'rb_similarity', 'llm_pick_similarity',
                                         # 'keyword_similarity_score', 'llm_is_customer_facing'}
                                         # -- luon duoc dien, de MOI request di qua parse() tu
                                         # dong gop them 1 dong du lieu co nhan (neu ground truth
                                         # duoc ghi lai rieng) cho lan hieu chinh conformal tiep
                                         # theo, thay vi phai chay lai LLM de lay lai cac dac
                                         # trung nay (dung y voi scope_gate_score o tren).


# ============================================================
# HELPER: similarity scoring
# ============================================================
def _compute_similarity(text: str, request_type: str, call_chains: dict = None) -> float:
    """Ti le keyword match trong [0, 1] voi request_type."""
    call_chains = call_chains if call_chains is not None else CALL_CHAINS
    if request_type not in call_chains:
        return 0.0
    keywords = call_chains[request_type].get('keywords', [])
    if not keywords:
        return 0.0
    text_clean = remove_accents(text.lower())
    matched = sum(
        1 for kw in keywords
        if re.search(r'\b' + re.escape(remove_accents(kw.lower())) + r'\b', text_clean)
    )
    # Nếu khớp >= 2 từ khóa thì coi như khớp tốt (>= 0.6)
    denominator = min(3, len(keywords)) if len(keywords) > 0 else 1
    return matched / denominator


# ============================================================
# CONFORMAL-CALIBRATED SCOPE GATE SCORER (the deployed Layer 1 mechanism)
# ============================================================
# Fitted offline on data/processed/scm_results/g6_scope_gate_layerA_validation.csv
# (61 labeled prompts: 50 legitimate RQ3 + 11 polysemous-keyword adversarial;
# see experiments/scope_gate_conformal_calibration.py for the full derivation).
# Coefficients below come from a logistic regression fit on ALL 61 labeled
# rows (features: keyword_similarity_score, llm_pick_similarity, rb_similarity,
# llm_is_customer_facing) -- appropriate for a deployed scorer, which should
# use every available label. The accuracy number reported in the paper
# (18.0% legitimate false-refusal at 100% recall) is instead the more honest
# 5-fold *out-of-fold* cross-validated estimate; the two differ slightly
# because this full-data fit is evaluated on its own training rows and is
# therefore mildly optimistic -- do not conflate the two numbers.
#
# KNOWN LIMITATION, stated plainly rather than buried: the underlying LR
# score above is still fit on the original 61-prompt calibration set (50
# legitimate + 11 out-of-scope) -- a live daily API quota limit blocked
# collecting raw features for a full refit on the larger set below. Only the
# TWO THRESHOLDS immediately below were recalibrated using the larger n, and
# both are evaluated on the SAME 81 points used to set them, not a further
# held-out set -- see docs/paper_draft.tex Section "Behavior at the Edge of
# the Declared Scope" ("Toward a Calibrated Threshold") for the full history
# (including the two-signal OR-rule this design replaced, and why).
_SCOPE_GATE_LR_COEF = {
    'keyword_similarity_score': 0.6169654821607805,
    'llm_pick_similarity':      0.2933483794392012,
    'rb_similarity':            0.6169654821607805,
    'llm_is_customer_facing':   -3.123588781264302,  # feature encoded as 1.0/0.0
}
_SCOPE_GATE_LR_INTERCEPT = 0.3548729348243096

# Two-threshold (reject-option) calibration, recalibrated on 81 labeled
# examples (21 out-of-scope, 60 legitimate) -- the original 61-prompt
# calibration set PLUS a genuinely fresh 20-prompt held-out set collected to
# check generalization (experiments/scope_gate_holdout_test.py). Only the
# thresholds use the expanded n; the LR coefficients above are unchanged
# (see limitation note above).
#
#   score <  TAU_PASS   -> confidently in-scope, auto-pass.
#     Calibrated on the 21 OUT-OF-SCOPE calibration scores: split-conformal
#     guarantees Pr[a new true out-of-scope prompt scores below this] <= 0.01.
#   score >= TAU_REFUSE  -> confidently out-of-scope, auto-refuse.
#     Calibrated on the 60 LEGITIMATE calibration scores (mirror-image
#     quantile): guarantees Pr[a new true legitimate prompt scores at or
#     above this] <= 0.05.
#   TAU_PASS <= score < TAU_REFUSE -> genuinely uncertain by BOTH guarantees
#     at once -> NEEDS_HUMAN_REVIEW, the same role the old OR-rule's HITL
#     escalation played, now driven by two calibrated bounds instead of a
#     hand-picked "confidence != HIGH" rule.
#
# On the 81-point calibration set this band gives: 0/21 out-of-scope missed
# in the auto-pass zone, 9 out-of-scope + 7 legitimate routed to review
# (16/81 ~ 20% of all cases), and only 3/60 (5.0%) legitimate prompts
# wrongly auto-refused -- down from 10/60 (16.7%) with a single threshold
# and no review band. This is evaluated on the SAME 81 points used to set
# the thresholds, not a further held-out set -- read as a validated design,
# not a proven one.
#
# A first attempt at deriving these two thresholds compared them in the
# WRONG direction (checked tau_refuse < tau_pass instead of tau_pass <
# tau_refuse), which would have silently produced an inverted, self-
# contradictory band; caught by printing an explicit sanity breakdown
# (missed / review / wrongly-refused counts) before hardcoding anything.
_SCOPE_GATE_CONFORMAL_TAU_PASS   = 0.5343128588727313  # alpha=0.01, n_pos=21
_SCOPE_GATE_CONFORMAL_TAU_REFUSE = 0.6907318736523925  # beta=0.05,  n_neg=60


def _scope_gate_conformal_score(keyword_similarity_score: float, llm_pick_similarity: float,
                                 rb_similarity: float, llm_is_customer_facing: bool) -> float:
    """P(out-of-scope) under the fitted logistic score. Higher = more suspicious."""
    z = (
        _SCOPE_GATE_LR_INTERCEPT
        + _SCOPE_GATE_LR_COEF['keyword_similarity_score'] * keyword_similarity_score
        + _SCOPE_GATE_LR_COEF['llm_pick_similarity'] * llm_pick_similarity
        + _SCOPE_GATE_LR_COEF['rb_similarity'] * rb_similarity
        + _SCOPE_GATE_LR_COEF['llm_is_customer_facing'] * (1.0 if llm_is_customer_facing else 0.0)
    )
    return 1.0 / (1.0 + pow(2.718281828459045, -z))


def _get_gateways(graph) -> set:
    """Gateway = node co in-degree=0 va khong phai database/queue/worker."""
    NON_GATEWAY_TYPES = {'database', 'message_queue', 'worker'}
    return {
        n for n in graph.nodes()
        if graph.in_degree(n) == 0
        and graph.nodes[n].get('type') not in NON_GATEWAY_TYPES
    }


# ============================================================
# GUARDS
# ============================================================
def _guard_injection_service(service: str, gateways: set,
                             archetype_services=None,
                             default_gateway: str = None) -> tuple:
    """G1 (Layer 2): dam bao injection_service la mot gateway hop le.

    Tra ve (gateway_da_chon, llm_de_xuat_dung_khong).

    Tren topology MOT gateway (SockShop) moi nhanh fallback deu hoi tu ve
    cung mot node, nen thu tu chon khong quan trong. Tren topology NHIEU
    gateway (Train Ticket co 14 node in-degree=0) thi no rat quan trong:
    ban truoc lay `sorted(gateways)[0]`, tuc `ts-admin-order-service` --
    mot dich vu QUAN TRI -- lam diem tiem cho MOI yeu cau khach hang. Do la
    loi that, bi che khuat vi SockShop chi co mot gateway.

    Thu tu uu tien bay gio:
      1. LLM de xuat dung mot gateway  -> nhan.
      2. Gateway nam trong call chain cua chinh archetype da khop -- archetype
         khai bao duong vao cua no, nen day la tin hieu co can cu nhat.
      3. `default_gateway` do he thong khai bao (vd ts-ui-dashboard).
      4. Alphabet -- chi con la luoi an toan cuoi cung, khong phai lua chon.
    """
    if service in gateways:
        return service, True

    if archetype_services:
        in_chain = [s for s in archetype_services if s in gateways]
        if in_chain:
            return in_chain[0], False

    if default_gateway and default_gateway in gateways:
        return default_gateway, False

    fallback = sorted(gateways)[0] if gateways else 'front-end'
    return fallback, False


def _guard_delta(delta: float, template_delta: float) -> tuple:
    """
    G2 + G3: Clamp delta va kiem tra adjustment.
    Tra ve (delta_clamped, adjustment_clamped, was_clamped).
    """
    # G3: adjustment vuot nguong -> fallback ve anchor
    actual_adj = delta - template_delta
    if abs(actual_adj) > MAX_ADJUSTMENT_PCT:
        delta      = template_delta
        actual_adj = 0.0
        clamped    = True
    else:
        clamped = False

    # G2: clamp vao khoang vat ly hop ly
    delta = max(MIN_DELTA_PCT, min(MAX_DELTA_PCT, delta))
    return delta, actual_adj, clamped


def _guard_core_services(services: list, known_services: set = None,
                         fallback_service: str = None) -> list:
    """G4 (Layer 2): chi giu service ton tai trong known_services.

    `known_services` mac dinh la module-level KNOWN_SERVICES (SockShop, giu de
    tuong thich nguoc cho cac script goi ham nay truc tiep khong qua mot
    ParserAgent instance) -- nhung parse() luon truyen self.known_services
    (tu chinh graph cua instance), de guard nay tong quat hoa theo he thong
    dang chay thay vi luon gia dinh SockShop.

    LOI DA SUA: ban truoc tra ve `['front-end']` khi danh sach rong sau khi
    loc. `front-end` la service CUA SOCKSHOP, viet cung. Tren Train Ticket
    (khong co node nao ten 'front-end') guard nay tu no CHEN VAO mot service
    khong ton tai -- tuc chinh no vi pham bat bien ma no ton tai de bao ve
    (S* subset V), lam Corollary "SHR = 0 by construction" SAI tren moi he
    thong khong co node ten 'front-end'. SockShop che lap loi nay vi o do
    'front-end' tinh co hop le.

    Fallback dung: gateway cua he thong dang chay (luon thuoc V theo dinh
    nghia), neu khong co thi tra ve danh sach RONG -- rong van thoa
    S* subset V, con mot service bia dat thi khong.
    """
    ks = known_services if known_services is not None else KNOWN_SERVICES
    valid = [s for s in services if s in ks]
    if valid:
        return valid
    if fallback_service and fallback_service in ks:
        return [fallback_service]
    return []


# ============================================================
# PARSER AGENT
# ============================================================
class ParserAgent:
    """
    Phan tich requirement -> ParsedRequirement.

    Chien luoc:
      1. Rule-based (classify_request): nhanh, deterministic -- chi de sinh
         goi y request_type va tinh similarity_score, KHONG con dung de bo
         qua LLM (fast-path da bi go bo, xem docstring parse()).
      2. LLM full-parse: luon chay cho MOI requirement
         -> LLM xac dinh request_type + core_services + adjustment +
            injection_service + is_customer_facing_feature (kem do tin cay)
         -> request_type tu rule-based duoc truyen vao LLM nhu goi y

    LLM KHONG duoc:
      - Tu xac dinh delta cho tung service
      - Tao gia tri ngoai calibration table +-10%
    """

    def __init__(self, llm, arch_agent, system_name: str = "SockShop",
                 domain_description: str = "an online sock e-commerce store",
                 call_chains: dict = None, default_gateway: str = None,
                 default_request_type: str = None):
        """
        system_name / domain_description: tham so hoa de Layer 1 (Scope Gate)
        khong con viet cung "SockShop" trong prompt -- xuat phat tu cau hoi
        "khung nay da tong quat hoa cho cac he thong khac chua" (chua kiem
        chung thuc nghiem tren Train Ticket, nhung co che gio da tham so hoa
        thay vi hardcode). Mac dinh giu nguyen gia tri SockShop de tuong thich
        nguoc voi moi noi goi ParserAgent() khong truyen 2 tham so nay.

        Layer 1 (Scope Gate): dung bo tinh diem logistic + HAI nguong
        split-conformal (TAU_PASS / TAU_REFUSE, xem docstring cac hang so o
        tren) -- day LA co che duy nhat tu phien ban nay, thay hoan toan cho
        luat OR/HITL rieng le truoc do (da go bo, xem lich su commit va
        docs/paper_draft.tex Section "Behavior at the Edge of the Declared
        Scope" de biet qua trinh dan den thiet ke nay). Vung giua 2 nguong
        tra ve NEEDS_HUMAN_REVIEW. Da kiem tra tren 20 prompt hoan toan moi
        (khong dung de hieu chinh): bo don-nguong ban dau (chi co TAU_REFUSE,
        khong co vung review) HOA CHUNG voi luat OR cu (10/10 recall, 1/10 tu
        choi oan ca hai) -- cai thien do tren 61 prompt goc KHONG lap lai
        tren du lieu chua thay. Sau khi gop ca 20 prompt do vao tap hieu
        chinh (61+20=81, 21 out-of-scope) va them lai vung review bang
        nguong thu hai, ket qua tren chinh 81 diem nay: 0/21 out-of-scope
        lot qua vung auto-pass, 16/81 (~20%) can nguoi xem, tu choi oan tu
        dong con 3/60 (5.0%) -- giam manh so voi 10/60 (16.7%) cua ban
        1-nguong khong co vung review.
        LUU Y CON LAI (chua giai quyet, khong nen giau): he so logistic
        regression VAN chi fit tren 61 prompt goc (chua refit duoc tren 81
        vi bi chan boi gioi han quota API hang ngay khi thu thap dac trung
        tho cho 20 prompt moi) -- chi 2 nguong duoc tinh lai voi n lon hon,
        khong phai toan bo mo hinh. Ket qua tren van danh gia tren CHINH 81
        diem dung de dat nguong, chua co tap test thu ba hoan toan doc lap.
        RQ3 (bang Table rq3 trong paper) duoc do TRUOC thay doi kien truc
        nay -- PBVR/SHR/GMR do o Layer 2/3, khong phu thuoc Layer 1 nen
        nhieu kha nang khong doi, nhung outcome NEEDS_HUMAN_REVIEW la nhom
        thu 3 chua co cot rieng trong bang RQ3 hien tai va chua duoc do lai
        bang live-LLM o dung quy mo 50-prompt x 3-lan-lap cua RQ3.
        """
        self.llm        = llm
        self.arch_agent = arch_agent
        self.system_name = system_name
        self.domain_description = domain_description
        # Taxonomy hieu chinh: tham so hoa (truoc day la module-level import cung,
        # khien khong the tro ParserAgent sang he thong thu hai). Mac dinh van la
        # SockShop de tuong thich nguoc.
        self.call_chains = call_chains if call_chains is not None else CALL_CHAINS
        self._priority_order = default_priority_order(self.call_chains)
        self._gateways  = _get_gateways(arch_agent.graph)
        # Gateway mac dinh do he thong khai bao, dung khi LLM de xuat sai va
        # archetype khong chua gateway nao (xem _guard_injection_service).
        # Chi co y nghia tren topology nhieu gateway.
        self.default_gateway = default_gateway
        # Archetype dung khi khong tu khoa nao khop. None -> classify_request tu
        # quyet dinh (SockShop giu GET_CATALOGUE de tuong thich nguoc).
        self.default_request_type = default_request_type

        # Build services context cho prompt, VA known_services (Layer 2 / G4)
        # tu CHINH graph duoc truyen vao thay vi doc module-level KNOWN_SERVICES
        # hardcode -- module constant van giu nguyen (nhieu script ben ngoai
        # dang import truc tiep de tinh SHR doc lap), nhung logic guard THAT
        # SU dung trong parse() gio theo dung graph cua instance nay.
        self._services_ctx = ""
        self.known_services = set()
        for node_id in arch_agent.graph.nodes:
            node_type = arch_agent.graph.nodes[node_id].get('type', '')
            if node_type in ('database', 'message_queue', 'worker'):
                continue
            desc = arch_agent.graph.nodes[node_id].get('description', '')
            self._services_ctx += f"  - {node_id}: {desc}\n"
            self.known_services.add(node_id)

        # Build calibration table string cho prompt
        self._calibration_ctx = ""
        for rt, info in self.call_chains.items():
            self._calibration_ctx += (
                f"  {rt:<25} | anchor={info['expected_delta_pct']}% "
                f"| {len(info['services'])} services "
                f"| {info['resource_profile']}\n"
            )

        print(f"[ParserAgent] Khoi tao xong ({self.system_name}). Gateways: {self._gateways}")

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------
    def parse(self, requirement: str) -> ParsedRequirement:
        """
        Phan tich requirement -> ParsedRequirement day du guard.
        Day la method duy nhat can goi tu orchestrator.

        KHONG con fast-path bypass. Ban truoc day, similarity keyword cao
        (rb_similarity >= SIMILARITY_THRESHOLD) se bo qua LLM hoan toan de
        tiet kiem chi phi (~34% cac prompt trong RQ3). Phat hien khi test
        Scope Gate qua dung parse() (khong phai goi thang _llm_full_parse):
        8/11 prompt doi khang duoc thiet ke de co du khop tu khoa VOI 1
        archetype SAI van du diem de kich hoat fast-path -- luc do
        is_customer_facing_feature khong bao gio duoc hoi, vi fast-path gan
        cung no la True vo dieu kien. Ket qua: ty le evade thuc te qua
        pipeline that la 8/11 (72.7%), gan nhu nguyen ven muc 100% ban dau,
        chu khong phai 0% nhu ket qua kiem chung truoc (kiem chung do goi
        thang _llm_full_parse, khong di qua fast-path, nen khong bao gio
        phoi bay lo hong nay). Da chon go bo fast-path hoan toan thay vi
        vas-va them 1 lop kiem tra rieng cho no -- an toan hon, danh doi
        mat loi ich chi phi/do tre da do trong RQ3 (xem paper).
        """
        rb_request_type  = classify_request(requirement, self.call_chains,
                                            self._priority_order,
                                            self.default_request_type)
        rb_similarity    = _compute_similarity(requirement, rb_request_type,
                                               self.call_chains)
        rb_template_info = self.call_chains.get(rb_request_type, {})
        print(f"  [Parser] Rule-based goi y: {rb_request_type} | similarity={rb_similarity:.2f}")

        request_type, core_svcs, adjustment, llm_reasoning, llm_conf, raw_inj_svc, llm_is_customer_facing, cf_confidence = \
            self._llm_full_parse(requirement, rb_request_type)

        # Re-lookup template sau khi LLM xac dinh request_type
        template_info  = self.call_chains.get(request_type, rb_template_info)
        template_delta = template_info.get('expected_delta_pct', 20.0)
        affected       = template_info.get('services', ['front-end'])
        # G6 can DUNG similarity cua chinh archetype LLM da chon (khong phai
        # chi rb_similarity ban dau) — LLM luon bi ep chon 1 loai "gan nhat",
        # nhung neu loai do CUNG khong chia se tu khoa nao voi requirement,
        # do la tin hieu that su khong co archetype dang tin cay.
        llm_pick_similarity = _compute_similarity(requirement, request_type,
                                                  self.call_chains)
        similarity_score = max(rb_similarity, llm_pick_similarity)
        llm_called     = True
        print(f"  [Parser] LLM full parse: {request_type} | conf: {llm_conf}")

        # --- GUARDS ---
        # G4: validate core_services
        # Fallback = gateway cua CHINH he thong nay (luon thuoc V), khong phai
        # 'front-end' viet cung. Xem docstring _guard_core_services.
        _fallback_svc = (self.default_gateway if self.default_gateway in self.known_services
                         else (sorted(self._gateways)[0] if self._gateways else None))
        core_svcs = _guard_core_services(core_svcs, self.known_services, _fallback_svc)

        # G5: low similarity -> siet adjustment ve 0
        if similarity_score < SIMILARITY_THRESHOLD:
            adjustment = 0.0
            llm_conf   = "LOW"
            llm_reasoning += " [LOW SIMILARITY: siet ve anchor, khong dieu chinh]"

        # G2 + G3: clamp delta
        raw_delta = template_delta + adjustment
        delta_clamped, adj_clamped, was_clamped = _guard_delta(raw_delta, template_delta)
        if was_clamped:
            llm_conf      = "LOW"
            llm_reasoning += (
                f" [CLAMPED: adj={adjustment:+.1f}% vuot nguong "
                f"{MAX_ADJUSTMENT_PCT}%, fallback ve anchor {template_delta}%]"
            )

        # G1: validate injection_service (gateway). `raw_inj_svc` is now a genuine
        # proposal (LLM-path) or the deterministic fast-path default -- NOT a
        # hardcoded literal fed to every call regardless of branch (previous
        # version always passed 'front-end' here, so G1 could never observe a
        # real violation even in principle; see docstring on _llm_full_parse).
        inj_svc, svc_ok = _guard_injection_service(
            raw_inj_svc, self._gateways,
            archetype_services=affected,
            default_gateway=self.default_gateway)
        if not svc_ok:
            llm_conf      = "LOW"
            llm_reasoning += f" [GATEWAY FALLBACK: LLM de xuat '{raw_inj_svc}' khong phai gateway hop le, dung {inj_svc}]"

        # G6 / Scope Gate: out-of-taxonomy refusal, decided by a single
        # conformal-calibrated score against two statistically-calibrated
        # thresholds (see _scope_gate_conformal_score and the two _TAU_
        # constants' docstrings above for exactly what each guarantees and
        # on what data). This replaced an earlier two-signal OR-rule (keyword
        # overlap OR the LLM's own self-declared is_customer_facing_feature,
        # with a separate hand-picked HITL escalation rule) after that design
        # was found to evade 100% of an 11-prompt polysemous-keyword
        # adversarial set in its keyword-only precursor, then to over-refuse
        # 12/50 legitimate RQ3 prompts even after the self-declaration fix
        # (docs/paper_draft.tex Section "Behavior at the Edge of the
        # Declared Scope" has the full history). cf_confidence remains
        # available from the same LLM extraction call but no longer drives
        # this decision -- the two calibrated thresholds below subsume its
        # role.
        #   score <  TAU_PASS   -> confidently in-scope, auto-pass.
        #   score >= TAU_REFUSE -> confidently out-of-scope, auto-refuse.
        #   TAU_PASS <= score < TAU_REFUSE -> uncertain by both statistical
        #     guarantees at once -> NEEDS_HUMAN_REVIEW.
        scope_gate_score = _scope_gate_conformal_score(
            keyword_similarity_score=similarity_score,
            llm_pick_similarity=llm_pick_similarity,
            rb_similarity=rb_similarity,
            llm_is_customer_facing=llm_is_customer_facing,
        )
        scope_gate_raw_features = {
            'rb_similarity': rb_similarity,
            'llm_pick_similarity': llm_pick_similarity,
            'keyword_similarity_score': similarity_score,
            'llm_is_customer_facing': llm_is_customer_facing,
        }

        hard_refuse = scope_gate_score >= _SCOPE_GATE_CONFORMAL_TAU_REFUSE
        needs_human_review = (not hard_refuse) and (scope_gate_score >= _SCOPE_GATE_CONFORMAL_TAU_PASS)
        is_out_of_scope = hard_refuse
        if hard_refuse:
            llm_conf      = "REFUSED"
            llm_reasoning += (
                f" [SCOPE GATE / conformal: score={scope_gate_score:.3f} >= "
                f"tau_refuse={_SCOPE_GATE_CONFORMAL_TAU_REFUSE:.3f} (split-conformal, "
                "beta=0.05, calibrated on 60 labeled legitimate examples).]"
            )
            llm_reasoning += (
                " KHONG du du lieu hieu chinh de dua ra con so dang tin cay — day chi la gia "
                "tri fallback, khuyen nghi load-test thu cong truoc khi trien khai thay vi "
                "dung so lieu du bao nay lam can cu quyet dinh."
            )
        elif needs_human_review:
            llm_conf = "NEEDS_HUMAN_REVIEW"
            llm_reasoning += (
                f" [SCOPE GATE / conformal HITL: score={scope_gate_score:.3f} nam giua "
                f"tau_pass={_SCOPE_GATE_CONFORMAL_TAU_PASS:.3f} va "
                f"tau_refuse={_SCOPE_GATE_CONFORMAL_TAU_REFUSE:.3f} -- khong du tin cay theo "
                "CA HAI bao dam thong ke de tu dong cho qua HOAC tu dong refuse, can nguoi "
                "xem lai truoc khi dua ra phan quyet dinh luong."
            )

        result = ParsedRequirement(
            request_type        = request_type,
            injection_service   = inj_svc,
            injection_delta_pct = delta_clamped,
            core_services       = core_svcs,
            affected_services   = affected,
            reasoning           = llm_reasoning,
            confidence          = llm_conf,
            matched_template    = request_type,
            template_delta      = template_delta,
            adjustment          = adj_clamped,
            similarity_score    = similarity_score,
            llm_was_called      = llm_called,
            is_out_of_scope     = is_out_of_scope,
            needs_human_review  = needs_human_review,
            scope_gate_score    = scope_gate_score,
            scope_gate_raw_features = scope_gate_raw_features,
        )

        print(f"  [Parser] -> delta={delta_clamped}% | core={core_svcs} | conf={llm_conf}")
        return result

    # ----------------------------------------------------------
    # INTERNAL: LLM lay core_services + adjustment (khi da biet request_type)
    # ----------------------------------------------------------
    def _llm_get_core_and_delta(
        self,
        requirement: str,
        request_type: str,
        template_delta: float
    ) -> tuple:
        """
        Goi LLM de xac dinh:
          - core_services: services can sua code (khac affected_services)
          - adjustment: dieu chinh delta so voi anchor trong [-10%, +10%]
        Tra ve: (core_services, adjustment, reasoning, confidence)
        """
        prompt = f"""Ban la Systems Analyst chuyen gia ve microservices {self.system_name}.

[CAC DICH VU TRONG HE THONG]
{self._services_ctx}

[BANG HIEU CHINH (Calibration Table — tu du lieu thuc nghiem)]
{self._calibration_ctx}

[TINH NANG MOI]
"{requirement}"

[BAN DA BIET]
  - Loai yeu cau: {request_type}
  - Delta anchor (tu bang hieu chinh): {template_delta}%
  - Khoang dieu chinh cho phep: [-10%, +10%] so voi anchor

[NHIEM VU]
1. Xac dinh "core_services": cac service CAN SUA CODE (khong phai toan bo blast radius).
   Quy tac:
   - Thay doi UI -> phai co "front-end"
   - Them logic gio hang/discount -> phai co "carts"
   - Them buoc thanh toan/kiem tra ma -> phai co "payment" hoac "orders"
   - Service chi doc du lieu (catalogue GET) -> KHONG can sua code

2. Xac dinh "adjustment": dieu chinh % so voi anchor.
   Chi tinh theo:
   - Them buoc DB write -> +3% den +5%
   - Them buoc validation phuc tap -> +2% den +4%
   - Tinh nang chi doc (GET only) -> -3% den -5%
   - Tinh nang nhe, khong them DB -> -2%
   KHONG duoc dat adjustment ngoai [-10%, +10%].

3. Xac dinh "confidence": HIGH (match ro), MEDIUM (match kha), LOW (khong chac)

Tra ve DUY NHAT JSON sau, KHONG them text khac:
{{
  "core_services": ["..."],
  "adjustment": <so thuc trong [-10, 10]>,
  "reasoning": "...",
  "confidence": "HIGH|MEDIUM|LOW"
}}"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            core  = data.get("core_services", ["front-end"])
            adj   = float(data.get("adjustment", 0.0))
            rsn   = str(data.get("reasoning", ""))
            conf  = data.get("confidence", "MEDIUM")
            if conf not in ("HIGH", "MEDIUM", "LOW"):
                conf = "MEDIUM"
            return core, adj, rsn, conf

        except Exception as e:
            print(f"  [Parser] LLM error: {e}")
            return ["front-end"], 0.0, f"[LLM ERROR: {e}]", "LOW"

    # ----------------------------------------------------------
    # INTERNAL: LLM xac dinh ca request_type + core_services + adjustment
    # ----------------------------------------------------------
    def _llm_full_parse(
        self,
        requirement: str,
        rb_suggestion: str
    ) -> tuple:
        """
        Duoc goi khi rule-based khong match du (similarity < threshold).
        LLM xac dinh ca request_type, core_services, adjustment, injection_service,
        is_customer_facing_feature, is_customer_facing_confidence.
        Tra ve: (request_type, core_services, adjustment, reasoning, confidence,
                 injection_service, is_customer_facing_feature,
                 is_customer_facing_confidence)

        Ve injection_service: truoc ban sua nay, ham nay khong hoi LLM ve diem
        injection tai tat ca -- parse() luon truyen cung 'front-end' vao G1
        (_guard_injection_service), nen G1 khong bao gio co co hoi bat loi that
        (chi la "hardcode dung", khong phai guard dang hoat dong). Gio LLM phai
        tu de xuat, de G1 co viec thuc su phai lam: kiem tra de xuat co phai la
        1 node in-degree=0 (gateway) khong, fallback neu sai.

        Ve is_customer_facing_feature: bo sung sau khi phat hien Scope Gate cu
        (chi dua vao similarity=0 sau khi da ep LLM chon 1 archetype) bi evade
        100% (10/10) boi cac yeu cau ngoai pham vi that su nhung tinh co chua 1
        tu khoa trung voi archetype khac (vd "refactor Java code" -> trung tu
        "code" cua APPLY_PROMO_CODE). Van de goc: prompt CU EP LLM luon phai
        chon 1 loai ("neu khong khop ro, chon loai TUONG TU NHAT"), khong cho
        no duong thoat de tu noi "khong cai nao phu hop" -- nen viec kiem tra
        pham vi phai lam RIENG, tach roi khoi hieu biet thuc su cua LLM. Gio
        LLM duoc hoi truc tiep, ngay trong luc trich xuat co cau truc, day co
        phai mot HANH DONG CUA KHACH HANG tren SockShop hay khong -- neu LLM
        da hieu day la viec noi bo/ky thuat (thuong no VAN hieu dung, chi la
        khong co cho de noi ra), no co the tu bao false thay vi bi ep chon
        archetype roi bi bat loi sau do boi mot co che tach biet, khong lien
        quan gi den chinh no. Day KHONG thay the backstop keyword-overlap doc
        lap (van giu nguyen) -- ca hai phai dong y moi duoc coi la trong pham
        vi, theo dung nguyen tac LLM-Modulo: khong tin 1 nguon duy nhat.
        """
        known_types = "\n".join(f"  - {k}: {v['description']}" for k, v in self.call_chains.items())

        prompt = f"""Ban la Systems Analyst chuyen gia ve microservices {self.system_name} ({self.domain_description}).

[CAC LOAI YEU CAU DA BIET]
{known_types}

[CAC DICH VU TRONG HE THONG]
{self._services_ctx}

[BANG HIEU CHINH (Calibration Table — tu du lieu thuc nghiem)]
{self._calibration_ctx}

[TINH NANG MOI CAN PHAN TICH]
"{requirement}"

[GUY Y TU PHAN LOAI TU DONG (co the sai)]
Rule-based goi y: {rb_suggestion}

[NHIEM VU]
0. Xac dinh "is_customer_facing_feature": day co phai mot HANH DONG CUA KHACH
   HANG thuc hien tren giao dien/ung dung {self.system_name} hay khong (vd: xem
   san pham, them gio hang, thanh toan, dang ky, theo doi don hang...)? Neu day
   la mot viec NOI BO/van hanh/ky thuat (vd: giam sat ha tang, refactor code,
   cap nhat thu vien, quy trinh nhan su, bao mat mang...) -- DU CO THE LIEN
   QUAN DEN CUNG HE THONG {self.system_name} -- dat gia tri nay la false. Day
   KHONG phai cau hoi "co lien quan {self.system_name} khong" (hau het moi thu
   gui vao day deu it nhieu lien quan) ma la "day co phai mot request TAO RA
   TAI TU KHACH HANG THAT hay khong".
   Ngoai ra xac dinh them "is_customer_facing_confidence": HIGH neu ban RAT
   chac chan ve cau tra loi true/false o tren; MEDIUM hoac LOW neu day la mot
   truong hop BIEN -- vi du noi dung THU DONG/TINH (trang FAQ, dieu khoan dich
   vu, nhan giao dien da ngon ngu, footer/phien ban) hoac mot qua trinh tu
   dong chay ngam (vd fraud detection tu dong quet) ma ban khong hoan toan
   chac day co tinh la "hanh dong chu dong cua khach hang" hay khong. KHONG
   duoc luon dat HIGH mac dinh -- day la truong rieng de danh dau su khong
   chac chan, se duoc dung de quyet dinh co can nguoi xem lai hay khong.
1. Chon "request_type" phu hop nhat tu danh sach da biet phia tren -- ke ca
   khi is_customer_facing_feature=false, van chon tam 1 loai gan nhat ve mat
   cau truc de he thong khong bi vo pipeline, nhung PHAI danh dau false o
   buoc 0, KHONG duoc ngam an chon dai roi bao la khop ro.
2. Xac dinh "core_services" (services can sua code).
3. Xac dinh "injection_service": dich vu nao la DIEM VAO (entry point) cua tai
   tang them do yeu cau nay gay ra -- tuc dich vu nhan request TRUC TIEP tu
   client/end-user, KHONG phai dich vu noi bo chi duoc goi giup boi service
   khac. Neu khong ro, chon dich vu co ve "gan client nhat" trong danh sach.
4. Xac dinh "adjustment": muc dieu chinh % (so voi anchor cua request_type da
   chon) ma theo BAN, phan anh dung nhat muc do bat thuong/quy mo cua yeu cau
   nay so voi truong hop dien hinh trong CALL_CHAINS. Tu do quyet dinh con so,
   khong bi gioi han truoc boi bat ky khoang nao — neu ban thay yeu cau nay
   thuc su bat thuong, cu de xuat con so lon; neu binh thuong, de xuat con so
   nho hoac 0.
5. Dat "confidence" = LOW neu khong chac chan, hoac neu is_customer_facing_feature=false.

QUAN TRONG: Neu is_customer_facing_feature=false hoac khong khop bat ky loai
nao, dat adjustment=0 va confidence=LOW.

Tra ve DUY NHAT JSON sau:
{{
  "is_customer_facing_feature": true/false,
  "is_customer_facing_confidence": "HIGH|MEDIUM|LOW",
  "request_type": "...",
  "core_services": ["..."],
  "injection_service": "...",
  "adjustment": <so thuc, don vi %>,
  "reasoning": "...",
  "confidence": "HIGH|MEDIUM|LOW"
}}"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            rt   = data.get("request_type", rb_suggestion)
            if rt not in self.call_chains:
                rt = rb_suggestion
            core = data.get("core_services", ["front-end"])
            inj_svc_raw = str(data.get("injection_service", "front-end"))
            adj  = float(data.get("adjustment", 0.0))
            rsn  = str(data.get("reasoning", ""))
            conf = data.get("confidence", "LOW")
            if conf not in ("HIGH", "MEDIUM", "LOW"):
                conf = "LOW"
            # Default True on a missing/malformed field rather than False: the
            # independent keyword-overlap backstop still applies regardless, so
            # this field failing open does not remove the other line of defense
            # (see docstring above -- this is Layer A, never trusted alone).
            is_customer_facing = bool(data.get("is_customer_facing_feature", True))
            cf_conf = data.get("is_customer_facing_confidence", "HIGH")
            if cf_conf not in ("HIGH", "MEDIUM", "LOW"):
                cf_conf = "HIGH"
            return rt, core, adj, rsn, conf, inj_svc_raw, is_customer_facing, cf_conf

        except Exception as e:
            print(f"  [Parser] LLM full parse error: {e}")
            return rb_suggestion, ["front-end"], 0.0, f"[LLM ERROR: {e}]", "LOW", "front-end", True, "HIGH"
