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
  G6: similarity == 0 HOAC LLM tu bao is_customer_facing_feature=false voi
      do tin cay HIGH -> REFUSED (danh dau is_out_of_scope=True); do tin cay
      thap hon hoac 2 tin hieu bat dong -> needs_human_review=True thay vi
      tu dong quyet dinh. Day la co che tu choi/escalate tuong minh cho yeu
      cau nam ngoai taxonomy da hieu chinh (xem muc "Scope" trong paper).

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


def _guard_core_services(services: list, known_services: set = None) -> list:
    """G4: chi giu service ton tai trong known_services.

    `known_services` mac dinh la module-level KNOWN_SERVICES (SockShop, giu de
    tuong thich nguoc cho cac script goi ham nay truc tiep khong qua mot
    ParserAgent instance) -- nhung parse() luon truyen self.known_services
    (tu chinh graph cua instance), de guard nay tong quat hoa theo he thong
    dang chay thay vi luon gia dinh SockShop."""
    ks = known_services if known_services is not None else KNOWN_SERVICES
    valid = [s for s in services if s in ks]
    return valid if valid else ['front-end']


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
                 domain_description: str = "an online sock e-commerce store"):
        """
        system_name / domain_description: tham so hoa de Layer 1 (Scope Gate)
        khong con viet cung "SockShop" trong prompt -- xuat phat tu cau hoi
        "khung nay da tong quat hoa cho cac he thong khac chua" (chua kiem
        chung thuc nghiem tren Train Ticket, nhung co che gio da tham so hoa
        thay vi hardcode). Mac dinh giu nguyen gia tri SockShop de tuong thich
        nguoc voi moi noi goi ParserAgent() khong truyen 2 tham so nay.
        """
        self.llm        = llm
        self.arch_agent = arch_agent
        self.system_name = system_name
        self.domain_description = domain_description
        self._gateways  = _get_gateways(arch_agent.graph)

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
        for rt, info in CALL_CHAINS.items():
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
        rb_request_type  = classify_request(requirement)
        rb_similarity    = _compute_similarity(requirement, rb_request_type)
        rb_template_info = CALL_CHAINS.get(rb_request_type, {})
        print(f"  [Parser] Rule-based goi y: {rb_request_type} | similarity={rb_similarity:.2f}")

        request_type, core_svcs, adjustment, llm_reasoning, llm_conf, raw_inj_svc, llm_is_customer_facing, cf_confidence = \
            self._llm_full_parse(requirement, rb_request_type)

        # Re-lookup template sau khi LLM xac dinh request_type
        template_info  = CALL_CHAINS.get(request_type, rb_template_info)
        template_delta = template_info.get('expected_delta_pct', 20.0)
        affected       = template_info.get('services', ['front-end'])
        # G6 can DUNG similarity cua chinh archetype LLM da chon (khong phai
        # chi rb_similarity ban dau) — LLM luon bi ep chon 1 loai "gan nhat",
        # nhung neu loai do CUNG khong chia se tu khoa nao voi requirement,
        # do la tin hieu that su khong co archetype dang tin cay.
        llm_pick_similarity = _compute_similarity(requirement, request_type)
        similarity_score = max(rb_similarity, llm_pick_similarity)
        llm_called     = True
        print(f"  [Parser] LLM full parse: {request_type} | conf: {llm_conf}")

        # --- GUARDS ---
        # G4: validate core_services
        core_svcs = _guard_core_services(core_svcs, self.known_services)

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
        inj_svc, svc_ok = _guard_injection_service(raw_inj_svc, self._gateways)
        if not svc_ok:
            llm_conf      = "LOW"
            llm_reasoning += f" [GATEWAY FALLBACK: LLM de xuat '{raw_inj_svc}' khong phai gateway hop le, dung {inj_svc}]"

        # G6 / Scope Gate: Out-of-taxonomy refusal, now TWO independent signals
        # combined with OR -- neither trusted alone (LLM-Modulo: no single source
        # of truth for a safety property). An 11-prompt adversarial set found the
        # OLD single-signal design (keyword-overlap only) evaded 100% of the time
        # by requirements containing one incidental keyword from an unrelated
        # archetype (data/processed/scm_results/g6_scope_gate_adversarial.csv);
        # a semantic-embedding replacement was tried and rejected (tested 2 models
        # x 2 description styles, all gave F1~0.31 with ~48-50/50 false refusals
        # on legitimate RQ3 prompts -- see scope_gate_embedding_calibration.py).
        #   Signal 1 (keyword-overlap, independent of the LLM's own report):
        #     similarity_score == 0.0 -- no archetype, including the LLM's own
        #     pick, shares a single keyword with the requirement.
        #   Signal 2 (structural self-declaration, NEW): the LLM is now asked,
        #     as part of the SAME structured extraction, whether this describes
        #     a genuine customer-facing SockShop action at all (is_customer_facing
        #     _feature) -- rather than being forced to always pick an archetype
        #     with no way to express "none of these fit". This does not replace
        #     Signal 1; it is checked in addition to it.
        keyword_signal  = (similarity_score == 0.0)
        scope_signal    = (llm_is_customer_facing == False)

        # Human-in-the-loop (HITL) escalation: added after finding that neither
        # signal alone, nor their raw disagreement, cleanly separates genuine
        # out-of-scope requests from legitimate-but-passive ones (5/50 RQ3
        # prompts -- static FAQ/ToS pages, a bilingual UI label, a footer, an
        # automatic fraud-detection scan -- were wrongly auto-refused; see
        # docs/paper_draft.tex Section "Behavior at the Edge of the Declared
        # Scope"). Rather than force a binary decision on cases the extraction
        # step itself is not confident about, we use the SAME self-declared
        # confidence field (is_customer_facing_confidence) to route uncertain
        # cases to a human reviewer instead of guessing either direction --
        # Layer 2 and Layer 3 remain fully automatic (their checks are
        # deterministic set-membership / numeric-interval tests with no
        # ambiguity to resolve); this HITL path applies to Layer 1 only.
        #   Auto-refuse (confident, no human needed): both signals agree, OR
        #     the LLM confidently (HIGH) self-declares non-customer-facing.
        #   Needs human review: the keyword backstop and the LLM's confident
        #     self-declaration DISAGREE (keyword says refuse, LLM confidently
        #     says it is a real customer action), OR the self-declaration
        #     itself (whichever way it leans) is not HIGH confidence.
        #   Auto-allow (confident, no human needed): neither signal fires and
        #     confidence is HIGH.
        hard_refuse = (keyword_signal and scope_signal) or (scope_signal and cf_confidence == "HIGH")
        needs_human_review = (not hard_refuse) and (
            (keyword_signal and not scope_signal)
            or (cf_confidence != "HIGH")
        )
        is_out_of_scope = hard_refuse
        if hard_refuse:
            llm_conf      = "REFUSED"
            if keyword_signal:
                llm_reasoning += (
                    " [SCOPE GATE / keyword-overlap: khong tim thay tu khoa trung khop voi "
                    "bat ky archetype da hieu chinh nao trong CALL_CHAINS, ke ca lua chon gan "
                    "nhat cua LLM.]"
                )
            if scope_signal:
                llm_reasoning += (
                    " [SCOPE GATE / self-declared: LLM tu bao day khong phai mot hanh dong "
                    f"cua khach hang tren {self.system_name} (is_customer_facing_feature=false, "
                    f"do tin cay={cf_confidence}).]"
                )
            llm_reasoning += (
                " KHONG du du lieu hieu chinh de dua ra con so dang tin cay — day chi la gia "
                "tri fallback, khuyen nghi load-test thu cong truoc khi trien khai thay vi "
                "dung so lieu du bao nay lam can cu quyet dinh."
            )
        elif needs_human_review:
            llm_conf = "NEEDS_HUMAN_REVIEW"
            llm_reasoning += (
                f" [SCOPE GATE / HITL: is_customer_facing={llm_is_customer_facing} nhung do tin "
                f"cay chi la {cf_confidence}, hoac keyword-overlap va tu-khai-bao cua LLM bat "
                "dong voi nhau -- day la truong hop BIEN, he thong KHONG tu quyet dinh refuse "
                "hay cho qua, can nguoi xem lai truoc khi dua ra phan quyet dinh luong."
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
        known_types = "\n".join(f"  - {k}: {v['description']}" for k, v in CALL_CHAINS.items())

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
            if rt not in CALL_CHAINS:
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
