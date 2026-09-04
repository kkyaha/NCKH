# -*- coding: utf-8 -*-
"""
ParserAgent — Requirement Interpreter
======================================
Phan tich yeu cau tinh nang moi -> injection_delta_pct + core_services.

Chien luoc 2 tang (Plan v4):
  Tang 1 (Rule-based): classify_request() kiem tra keyword match
                       Neu match ro (similarity >= threshold) -> dung ket qua ngay
  Tang 2 (LLM):        Neu khong match -> LLM xac dinh request_type
                       LLM luon xac dinh: core_services + adjustment

Guards (bat buoc, kiem tra TRUOC khi tra ket qua):
  G1: injection_service phai la gateway (in-degree=0 tu do thi)
  G2: injection_delta_pct phai trong [MIN_DELTA, MAX_DELTA]
  G3: adjustment phai trong [-MAX_ADJ, +MAX_ADJ], vuot -> fallback ve anchor
  G4: core_services chi chua service ton tai trong graph
  G5: low similarity -> siet adjustment = 0

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

from request_router import CALL_CHAINS, classify_request, remove_accents

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
    Output chuan cua ParserAgent. Tuong thich voi ca 2 path:
      - PerformanceAgent.get_metrics_for_service(srv, injection_delta_pct)
      - SimulationAgent.simulate_intervention(injection_service, injection_delta_pct)
    """
    # Core fields — dung cho ca 2 path
    request_type:         str         # "APPLY_PROMO_CODE" | "UNKNOWN"
    injection_service:    str         # gateway: node in-degree=0 tu do thi
    injection_delta_pct:  float       # da qua guard, san sang cho SCM
    core_services:        list        # services can sua code (LLM xac dinh)
    affected_services:    list        # blast radius (tu CALL_CHAINS)

    # Explainability — cho paper
    reasoning:            str
    confidence:           str         # "HIGH" | "MEDIUM" | "LOW"
    matched_template:     str         # template duoc chon lam anchor
    template_delta:       float       # anchor goc (chua dieu chinh)
    adjustment:           float       # delta_actual - template_delta (sau clamp)
    similarity_score:     float       # keyword match score [0, 1]
    llm_was_called:       bool        # True neu goi LLM (cho efficiency metric)


# ============================================================
# HELPER: similarity scoring
# ============================================================
def _compute_similarity(text: str, request_type: str) -> float:
    """Ti le keyword match trong [0, 1] voi request_type."""
    if request_type not in CALL_CHAINS:
        return 0.0
    keywords = CALL_CHAINS[request_type].get('keywords', [])
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
def _guard_injection_service(service: str, gateways: set) -> tuple:
    """G1: dam bao injection_service la gateway."""
    if service in gateways:
        return service, True
    # Fallback: chon gateway dau tien (sorted de deterministic)
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


def _guard_core_services(services: list) -> list:
    """G4: chi giu service ton tai trong KNOWN_SERVICES."""
    valid = [s for s in services if s in KNOWN_SERVICES]
    return valid if valid else ['front-end']


# ============================================================
# PARSER AGENT
# ============================================================
class ParserAgent:
    """
    Phan tich requirement -> ParsedRequirement.

    Chien luoc:
      1. Rule-based (classify_request): nhanh, deterministic
         -> Neu similarity >= SIMILARITY_THRESHOLD: dung ket qua ngay (khong goi LLM)
      2. LLM fallback: khi khong match ro
         -> LLM xac dinh request_type (neu rule-based fail) + core_services + adjustment
         -> request_type tu rule-based duoc truyen vao LLM nhu goi y

    LLM KHONG duoc:
      - Tu xac dinh delta cho tung service
      - Tao gia tri ngoai calibration table +-10%
    """

    def __init__(self, llm, arch_agent):
        self.llm        = llm
        self.arch_agent = arch_agent
        self._gateways  = _get_gateways(arch_agent.graph)

        # Build services context cho prompt
        self._services_ctx = ""
        for node_id in arch_agent.graph.nodes:
            node_type = arch_agent.graph.nodes[node_id].get('type', '')
            if node_type in ('database', 'message_queue', 'worker'):
                continue
            desc = arch_agent.graph.nodes[node_id].get('description', '')
            self._services_ctx += f"  - {node_id}: {desc}\n"

        # Build calibration table string cho prompt
        self._calibration_ctx = ""
        for rt, info in CALL_CHAINS.items():
            self._calibration_ctx += (
                f"  {rt:<25} | anchor={info['expected_delta_pct']}% "
                f"| {len(info['services'])} services "
                f"| {info['resource_profile']}\n"
            )

        print(f"[ParserAgent] Khoi tao xong. Gateways: {self._gateways}")

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------
    def parse(self, requirement: str) -> ParsedRequirement:
        """
        Phan tich requirement -> ParsedRequirement day du guard.
        Day la method duy nhat can goi tu orchestrator.
        """
        # --- TANG 1: Rule-based classify ---
        rb_request_type  = classify_request(requirement)
        rb_similarity    = _compute_similarity(requirement, rb_request_type)
        rb_template_info = CALL_CHAINS.get(rb_request_type, {})

        print(f"  [Parser] Rule-based: {rb_request_type} | similarity={rb_similarity:.2f}")

        if rb_similarity >= SIMILARITY_THRESHOLD:
            # Match ro: dung rule-based, chi goi LLM de lay core_services + adjustment
            request_type    = rb_request_type
            template_delta  = rb_template_info.get('expected_delta_pct', 20.0)
            affected        = rb_template_info.get('services', ['front-end'])
            similarity_score = rb_similarity

            # LLM chi xac dinh core_services + adjustment (khong xac dinh request_type)
            core_svcs, adjustment, llm_reasoning, llm_conf = self._llm_get_core_and_delta(
                requirement, request_type, template_delta
            )
            llm_called = True
            print(f"  [Parser] LLM adjustment: {adjustment:+.1f}% | confidence: {llm_conf}")

        else:
            # Khong match: LLM xac dinh ca request_type
            request_type, core_svcs, adjustment, llm_reasoning, llm_conf = \
                self._llm_full_parse(requirement, rb_request_type)

            # Re-lookup template sau khi LLM xac dinh request_type
            template_info  = CALL_CHAINS.get(request_type, rb_template_info)
            template_delta = template_info.get('expected_delta_pct', 20.0)
            affected       = template_info.get('services', ['front-end'])
            similarity_score = rb_similarity
            llm_called     = True
            print(f"  [Parser] LLM full parse: {request_type} | conf: {llm_conf}")

        # --- GUARDS ---
        # G4: validate core_services
        core_svcs = _guard_core_services(core_svcs)

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

        # G1: validate injection_service (gateway)
        inj_svc, svc_ok = _guard_injection_service('front-end', self._gateways)
        if not svc_ok:
            llm_conf      = "LOW"
            llm_reasoning += f" [GATEWAY FALLBACK: dung {inj_svc}]"

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
        prompt = f"""Ban la Systems Analyst chuyen gia ve microservices SockShop.

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
        LLM xac dinh ca request_type, core_services, adjustment.
        Tra ve: (request_type, core_services, adjustment, reasoning, confidence)
        """
        known_types = "\n".join(f"  - {k}: {v['description']}" for k, v in CALL_CHAINS.items())

        prompt = f"""Ban la Systems Analyst chuyen gia ve microservices SockShop.

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
1. Chon "request_type" phu hop nhat tu danh sach da biet phia tren.
   Neu khong khop ro, chon loai CO CAU TRUC CALL CHAIN TUONG TU NHAT.
2. Xac dinh "core_services" (services can sua code).
3. Xac dinh "adjustment" trong [-10%, +10%] so voi anchor cua request_type da chon.
4. Dat "confidence" = LOW neu khong chac chan.

QUAN TRONG: Neu khong khop bat ky loai nao, dat adjustment=0 va confidence=LOW.
KHONG duoc dat adjustment ngoai [-10, 10].

Tra ve DUY NHAT JSON sau:
{{
  "request_type": "...",
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

            rt   = data.get("request_type", rb_suggestion)
            if rt not in CALL_CHAINS:
                rt = rb_suggestion
            core = data.get("core_services", ["front-end"])
            adj  = float(data.get("adjustment", 0.0))
            rsn  = str(data.get("reasoning", ""))
            conf = data.get("confidence", "LOW")
            if conf not in ("HIGH", "MEDIUM", "LOW"):
                conf = "LOW"
            return rt, core, adj, rsn, conf

        except Exception as e:
            print(f"  [Parser] LLM full parse error: {e}")
            return rb_suggestion, ["front-end"], 0.0, f"[LLM ERROR: {e}]", "LOW"
