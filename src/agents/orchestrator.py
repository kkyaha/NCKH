# -*- coding: utf-8 -*-
"""
Orchestrator — MAS Feasibility Analyzer
=========================================
Luan do 4 buoc (Plan v4 — Dual-Path Architecture):

  [parse_requirement] -> [map_impact] -> [simulate] -> [generate_report]
         |                    |               |               |
   ParserAgent          ArchAgent       Dual-path:       LLM report
   (LLM+rule)           (Graph)       FAST (bivariate)  (ca 2 nguon)
   core_services                      ACCURATE (DAG)
   injection_delta

Dual-Path:
  Fast Path   = PerformanceAgent (Bivariate, 21 model) — nhanh, per-service doc lap
  Accurate Path = SimulationAgent (Global DAG 28-node) — cascade attenuation thuc te
"""

import os
import sys
import json
from dataclasses import asdict
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Path setup — resolve project root tu vi tri file nay
_AGENT_DIR   = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR     = os.path.dirname(_AGENT_DIR)
PROJECT_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from agents.architecture_agent import ArchitectureAgent
from agents.capacity_agent     import CapacityAgent
from agents.parser_agent       import ParserAgent

# ==========================================
# 1. CAU HINH API KEY VA KHOI TAO LLM
# ==========================================
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
else:
    print("[WARNING] GOOGLE_API_KEY chua duoc thiet lap trong .env. Su dung Offline Synthesizer Fallback.")
    class OfflineLLM:
        def invoke(self, messages):
            class Resp:
                content = (
                    "### BAO CAO DANH GIA KHA THI (OFFLINE FALLBACK)\n\n"
                    "**1. TAC DONG KIEN TRUC:**\n"
                    "Tinh nang moi yeu cau cap nhat tai cac core services va cac API lien doi theo so do blast radius.\n\n"
                    "**2. PHAN TICH NANG LUC & RUI RO TAI NGUYEN (CapacityAgent ReAct):**\n"
                    "SCM do-calculus xac nhan tai lan truyen qua cac hop va suy giam cascade. "
                    "Cac dich vu trong call chain van nam trong nguong an toan, tuy nhien can chu y cac nut bao hoa.\n\n"
                    "**3. KET LUAN & KHUYEN NGHI:**\n"
                    "CO THE TRIEN KHAI voi dieu kien ap dung day du cac khuyen nghi ky thuat tu CapacityAgent."
                )
            return Resp()
    llm = OfflineLLM()

# ==========================================
# 2. KHOI TAO CAC AGENT
# ==========================================
_GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')
_DATA_DIR   = os.path.join(PROJECT_ROOT, 'data', 'raw')

arch_agent = ArchitectureAgent(_GRAPH_PATH)

# CapacityAgent hop nhat ca 2 che do SCM va tich hop buoc ReAct Reasoning
capacity_agent = CapacityAgent(
    llm        = llm,
    data_dir   = _DATA_DIR,
    graph_path = _GRAPH_PATH,
    auto_train = True
)

parser_agent = ParserAgent(llm=llm, arch_agent=arch_agent)

# Backward-compatibility aliases
perf_agent = capacity_agent
sim_agent  = capacity_agent

# ==========================================
# 2b. FEASIBILITY AGENT (tinh nang CHUA TUNG CO, khac muc dich voi CapacityAgent o tren)
# ==========================================
# CapacityAgent.train() hoc tu du lieu RCAEval fault-injection, phuc vu RQ1-RQ12 da cong bo -- KHONG
# dung chung pipeline voi duong nay de khong dung cham vao no. NewFeatureFeasibilityAgent tra loi mot
# cau hoi khac: "them mot tinh nang CHUA TUNG CO thi he thong Sock Shop dang chay con dap ung SLO o tai
# dinh L khong" (docs/DATA_FRAMEWORK.md), hoc tu SS-TRAIN/SS-LIMITS (khong phai RCAEval). Tao lazy (chi
# khi goi assess_new_feature_requirement) de import nay khong lam hong moi truong thieu file dong bang.
_feasibility_agent = None


def _get_feasibility_agent():
    global _feasibility_agent
    if _feasibility_agent is None:
        from agents.feasibility_agent import NewFeatureFeasibilityAgent
        _feasibility_agent = NewFeatureFeasibilityAgent()
    return _feasibility_agent


def assess_new_feature_requirement(text: str, L_peak: float, k: dict = None, bootstrap: bool = True) -> dict:
    """Duong rieng NL -> archetype -> kha thi, dung ParserAgent (da co) + NewFeatureFeasibilityAgent
    (P2/P3, docs/DATA_FRAMEWORK.md). KHONG dung StateGraph `feasibility_analyzer` o duoi (duong do dung
    CapacityAgent cho muc dich khac); goi ham nay TRUC TIEP, khong qua workflow.invoke().

    k: boi so goi do duoc (vd tu experiments/probe_feature_chain.py) neu tinh nang DA duoc cai va do --
    truyen vao thi dung P3, khong truyen thi P2 (gia dinh k=1, it lac quan hon, xem docs muc 5g-5h).
    """
    parsed = parser_agent.parse(text)
    agent = _get_feasibility_agent()
    from feasibility_predictor import SOCKSHOP_CALL_CHAINS, FEATURE_ARCHETYPE  # noqa: E402 (da tren sys.path qua feasibility_agent)
    archetype = parsed.request_type
    if archetype not in SOCKSHOP_CALL_CHAINS:
        return {'parsed_requirement': asdict(parsed), 'feasibility': None,
                'error': f"archetype '{archetype}' khong co trong taxonomy -- ngoai pham vi (xem Scope Gate)"}
    # scale sao cho delta THAT su dung = injection_delta_pct cua CHINH yeu cau nay (khong phai
    # anchor mac dinh cua archetype) -- ton trong uoc luong rieng cua ParserAgent cho tung yeu cau.
    anchor = SOCKSHOP_CALL_CHAINS[archetype]['expected_delta_pct']
    scale = (parsed.injection_delta_pct / anchor) if anchor else 1.0
    verdict = agent.assess(archetype, scale=scale, L_peak=L_peak, k=k, bootstrap=bootstrap)
    return {'parsed_requirement': asdict(parsed), 'feasibility': verdict}

# ==========================================
# 3. DINH NGHIA STATE
# ==========================================
class RequirementState(TypedDict):
    input_requirement:   str    # Yeu cau tu nguoi dung
    parsed_requirement:  dict   # Output cua ParserAgent (ParsedRequirement)
    core_services:       list   # Services can sua code
    impact_graph:        dict   # So do tac dong API
    performance_metrics: dict   # Fast Path: bivariate predictions
    simulation_result:   dict   # Accurate Path: multi-hop cascade predictions
    capacity_assessment: dict   # ReAct Output: expert analysis & Devil's Advocate
    feasibility_report:  str    # Bao cao kha thi cuoi cung

# ==========================================
# 4. CAC NODES
# ==========================================

def parse_requirement_node(state: RequirementState) -> dict:
    """
    Node 1: ParserAgent phan tich yeu cau.
    Chien luoc: Rule-based truoc (classify_request), LLM fallback.
    LLM xac dinh: core_services + adjustment (khong tu tinh delta cho tung service).
    """
    print("\n[Phase 1] Phan tich yeu cau (Rule-based + LLM)...")
    parsed = parser_agent.parse(state['input_requirement'])
    print(f"  -> request_type={parsed.request_type} | "
          f"delta={parsed.injection_delta_pct}% | "
          f"confidence={parsed.confidence} | "
          f"llm_called={parsed.llm_was_called}")
    return {
        "parsed_requirement": asdict(parsed),
        "core_services":      parsed.core_services,
    }


def map_impact_node(state: RequirementState) -> dict:
    """
    Node 2: ArchitectureAgent quet do thi tim dich vu lien doi.
    Su dung parsed_requirement.affected_services lam blast radius chinh.
    """
    print("\n[Phase 2] Phan tich tac dong day chuyen tren do thi...")
    impact_mapping = {}

    affected = state['parsed_requirement'].get('affected_services', [])
    base_services = set(state['core_services']) | set(affected)

    for service in base_services:
        if service not in arch_agent.graph:
            continue
        impact_mapping[service] = {
            "api_consumers_to_notify":     arch_agent.get_upstream_dependencies(service),
            "downstream_services_to_check": arch_agent.get_downstream_dependencies(service),
        }

    print(f"  -> Da quet xong {len(impact_mapping)} dich vu.")
    return {"impact_graph": impact_mapping}


def simulate_node(state: RequirementState) -> dict:
    """
    Node 3: CapacityAgent thuc thi chu trinh ReAct (Act -> Observe -> Reason).
      - ACT: Chay ca Fast Path (Bivariate) va Accurate Path (Global DAG).
      - OBSERVE: Sang loc cac service vuot nguong an toan (>70%).
      - REASON: Phan tich diem nghen va tu phan bien Devil's Advocate.
    """
    print("\n[Phase 3] CapacityAgent thuc thi chu trinh ReAct (Act -> Observe -> Reason)...")

    assessment = capacity_agent.assess_capacity(
        parsed_requirement = state['parsed_requirement'],
        impact_graph       = state['impact_graph']
    )

    print(f"  -> Trang thai Capacity: {assessment.status}")
    print(f"  -> Diem nghen bao hoa: {assessment.saturated_services}")
    print(f"  -> Khuyen nghi ky thuat: {len(assessment.recommendations)} muc")

    return {
        "performance_metrics": assessment.fast_metrics,
        "simulation_result":   assessment.simulation_result,
        "capacity_assessment": asdict(assessment),
    }


def generate_report_node(state: RequirementState) -> dict:
    """
    Node 4: LLM tong hop Bao cao Kha thi cuoi cung.
    Duoc tiep suc boi phan tich chuyen gia tu CapacityAgent (Reasoning & Devil's Advocate).
    """
    print("\n[Phase 4] Tong hop Bao cao Kha thi (Lead Architect LLM)...")

    parsed = state['parsed_requirement']

    # G6 short-circuit: neu ParserAgent da danh dau is_out_of_scope=True (khong
    # archetype nao trung khop du 1 tu khoa), TU CHOI tong hop mot bao cao kha
    # thi dinh luong. Day la buoc quyet dinh (khong phai chi mot ghi chu trong
    # prompt) de tranh truong hop LLM tong hop "lam min" canh bao refusal thanh
    # mot ket luan nghe co ve chac chan. Xem Section "Scope" / muc "Behavior at
    # the Edge of the Declared Scope" trong paper.
    if parsed.get('is_out_of_scope'):
        print("  [Phase 4] G6 TRIGGERED: tu choi tong hop bao cao dinh luong.")
        refusal_report = f"""### BAO CAO: KHONG DU DU LIEU HIEU CHINH (OUT-OF-TAXONOMY)

**Yeu cau**: "{state['input_requirement']}"

**Ket luan**: He thong KHONG tim thay archetype hieu chinh nao (trong
CALL_CHAINS) chia se du 1 tu khoa voi yeu cau nay — ke ca lua chon "gan nhat"
ma LLM da thu de xuat. Day la tin hieu ro rang requirement nam NGOAI pham vi
taxonomy da hieu chinh cua he thong (xem Scope trong tai lieu thiet ke).

**KHONG dua ra phan quyet kha thi dinh luong** (CO THE trien khai / CAN SCALE)
cho truong hop nay, vi bat ky con so workload delta nao duoc tao ra deu chi la
gia tri fallback khong co co so thuc nghiem, khong phai du bao dang tin cay.

**Khuyen nghi**:
1. Load-test thu cong tinh nang nay truoc khi trien khai san xuat.
2. Neu tinh nang thuc su gan voi mot nghiep vu da biet, hay dien dat lai yeu
   cau ro rang hon (nhac ten nghiep vu/service lien quan) de he thong co the
   tra ve mot uoc luong co can cu.
3. Bo sung archetype hieu chinh moi (CALL_CHAINS) neu day la mot loai tinh
   nang se lap lai trong tuong lai.

**Ly do chi tiet tu ParserAgent**: {parsed['reasoning']}
"""
        return {"feasibility_report": refusal_report}

    delta         = parsed['injection_delta_pct']
    cap_eval      = state.get('capacity_assessment', {})
    expert_assess = cap_eval.get('expert_assessment', 'Danh gia nang luc tai hoan tat.')
    risk_critique = cap_eval.get('risk_critique', 'Can kiem tra ky co che hang doi.')
    recs          = cap_eval.get('recommendations', [])
    status        = cap_eval.get('status', 'SAFE')

    # G7: OOD-confidence guard (magnitude of extrapolation, orthogonal to G6's
    # taxonomy-membership check). Unlike G6, a requirement that trips this is
    # still a legitimate archetype match -- it just projects one or more
    # downstream nodes outside their own training envelope (see
    # CapacityAgent._classify_ood_confidence, experiments/g7_ood_guard_test.py).
    # We do not refuse the numeric verdict; we deterministically attach the
    # caveat below regardless of what the synthesis LLM chooses to mention,
    # for the same reason G6 short-circuits at the code level rather than
    # relying on an instruction the LLM could smooth over.
    ood_confidence = cap_eval.get('ood_confidence', 'high')
    ood_flagged    = cap_eval.get('ood_flagged_nodes', [])

    system_prompt = """Ban la Ky su Truong (Principal Engineer) chuyen gia Microservices.
Nhiem vu: Tong hop Bao cao Kha thi Toan dien khi them tinh nang moi vao kien truc vi dich vu.
Dua vao ket qua phan tich chuyen gia tu CapacityAgent va so do tac dong kien truc tu ArchitectureAgent."""

    human_prompt = f"""
=== YEU CAU TINH NANG MOI ===
{state['input_requirement']}

=== KET QUA PHAN TICH YEU CAU (ParserAgent) ===
  Loai yeu cau: {parsed['request_type']}
  Dich vu can sua code: {state['core_services']}
  Blast radius: {parsed['affected_services']}
  Workload delta tai gateway: +{delta}%
  Confidence: {parsed['confidence']}
  Ly do: {parsed['reasoning']}

=== TAC DONG KIEN TRUC (Do thi API tu ArchitectureAgent) ===
{state['impact_graph']}

=== DANH GIA NANG LUC & PHAN BIEN CHUYEN GIA (CapacityAgent ReAct Output) ===
  Trang thai he thong: {status}
  Cac dich vu cham nguong bao hoa: {cap_eval.get('saturated_services', [])}

  [Phan tich chuyen gia Capacity]:
  {expert_assess}

  [Tu phan bien rui ro - Devil's Advocate Critique]:
  {risk_critique}

  [Cac khuyen nghi ky thuat]:
  {recs}

=== DU LIEU DINH LUONG SCM (Fast Path & Accurate Path) ===
Fast Path (Bivariate): {state['performance_metrics']}
Accurate Path (Global DAG): {state['simulation_result']}

=== G7: DO TIN CAY NGOAI SUY (OOD-confidence, doc lap voi confidence taxonomy) ===
  Muc do tin cay: {ood_confidence}
  Node vuot P95 phan phoi huan luyen: {ood_flagged if ood_flagged else 'Khong co'}

=== YEU CAU BAO CAO ===
1. TAC DONG KIEN TRUC: Liet ke API can them/sua, ai goi ai.
2. PHAN TICH RUI RO TAI NGUYEN (Tich hop phan bien Devil's Advocate):
   - Uu tien ket qua Accurate Path (cascade thuc te) va phan tich chuyen gia cua CapacityAgent.
   - Nhac lai cac rui ro ngoai sinh tiem an tu phan tu phan bien Devil's Advocate.
3. KET LUAN KHA THI & LO TRINH TRIEN KHAI:
   - Ket luan: CO THE trien khai ngay / CAN SCALE TRUOC.
   - Ke hoach hanh dong dua tren danh sach khuyen nghi cua CapacityAgent.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm.invoke(messages)
    report = response.content

    # G7 short-circuit: force the OOD caveat onto the report regardless of
    # whether the synthesis LLM chose to surface it -- a code-level
    # guarantee, not a prompt instruction (same rationale as G6, Section
    # "Behavior at the Edge of the Declared Scope" in the paper).
    if ood_confidence in ("low", "very_low"):
        report += f"""

---
**[G7 -- CANH BAO NGOAI SUY (OOD), tu dong chen boi guard, khong phai LLM]**
Muc do tin cay ngoai suy: **{ood_confidence.upper()}**. Cac node sau co gia tri
du phong VUOT NGOAI khoang P95 cua phan phoi du lieu huan luyen (SCM van tra
ve mot con so, nhung con so nay it duoc kiem chung boi du lieu thuc te hon
cac truong hop thong thuong):
{chr(10).join(f'  - {n}' for n in ood_flagged) if ood_flagged else '  (khong xac dinh duoc node cu the)'}

Day KHONG phai loi do requirement sai taxonomy (xem G6) -- day la mot yeu cau
hop le nhung day muc do tac dong toi ria (hoac vuot ria) vung du lieu da
tung quan sat. Khuyen nghi: doi chieu voi load-test thuc te truoc khi xem
ket luan phia tren la bao chung cuoi cung.
"""

    return {"feasibility_report": report}


# ==========================================
# 5. LAP RAP WORKFLOW LANGGRAPH
# ==========================================
workflow = StateGraph(RequirementState)

workflow.add_node("parse_requirement", parse_requirement_node)
workflow.add_node("map_impact",        map_impact_node)
workflow.add_node("simulate",          simulate_node)
workflow.add_node("generate_report",   generate_report_node)

workflow.set_entry_point("parse_requirement")
workflow.add_edge("parse_requirement", "map_impact")
workflow.add_edge("map_impact",        "simulate")
workflow.add_edge("simulate",          "generate_report")
workflow.add_edge("generate_report",   END)

feasibility_analyzer = workflow.compile()


# ==========================================
# 6. THUC THI KIEM THU
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MAS FEASIBILITY ANALYZER — Dual-Path SCM (v4)")
    print("  Fast Path: Bivariate | Accurate Path: Global DAG 28-node")
    print("=" * 70)

    initial_state = {
        "input_requirement": (
            "Customers want a new feature to apply a discount promo code (Voucher) "
            "on the checkout screen before confirming the order."
        )
    }

    result = feasibility_analyzer.invoke(initial_state)

    print("\n" + "=" * 70)
    print("BAO CAO DANH GIA TAC DONG & KHA THI TAI NGUYEN:")
    print("=" * 70)
    print(result['feasibility_report'])
