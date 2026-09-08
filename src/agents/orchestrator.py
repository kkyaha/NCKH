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

    parsed        = state['parsed_requirement']
    delta         = parsed['injection_delta_pct']
    cap_eval      = state.get('capacity_assessment', {})
    expert_assess = cap_eval.get('expert_assessment', 'Danh gia nang luc tai hoan tat.')
    risk_critique = cap_eval.get('risk_critique', 'Can kiem tra ky co che hang doi.')
    recs          = cap_eval.get('recommendations', [])
    status        = cap_eval.get('status', 'SAFE')

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
    return {"feasibility_report": response.content}


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
