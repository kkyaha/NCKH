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
from agents.performance_agent  import PerformanceAgent
from agents.parser_agent       import ParserAgent
from agents.simulation_agent   import SimulationAgent

# ==========================================
# 1. CAU HINH API KEY VA KHOI TAO LLM
# ==========================================
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY not set. Vui long them GOOGLE_API_KEY vao file .env hoac export GOOGLE_API_KEY='your-key'."
    )

llm = ChatGoogleGenerativeAI(model="gemini-3.0-flash", temperature=0.2)

# ==========================================
# 2. KHOI TAO CAC AGENT
# ==========================================
_GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')
_DATA_DIR   = os.path.join(PROJECT_ROOT, 'data', 'raw')

arch_agent = ArchitectureAgent(_GRAPH_PATH)

perf_agent = PerformanceAgent(
    data_dir   = _DATA_DIR,
    auto_train = True
)

sim_agent = SimulationAgent(
    data_dir   = _DATA_DIR,
    graph_path = _GRAPH_PATH
)
sim_agent.train()

parser_agent = ParserAgent(llm=llm, arch_agent=arch_agent)

# ==========================================
# 3. DINH NGHIA STATE
# ==========================================
class RequirementState(TypedDict):
    input_requirement:  str    # Yeu cau tu nguoi dung
    parsed_requirement: dict   # Output cua ParserAgent (ParsedRequirement)
    core_services:      list   # Services can sua code
    impact_graph:       dict   # So do tac dong API
    performance_metrics: dict  # Fast Path: bivariate predictions
    simulation_result:  dict   # Accurate Path: multi-hop cascade predictions
    feasibility_report: str    # Bao cao kha thi cuoi cung

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

    # Uu tien affected_services tu ParsedRequirement (chinh xac hon vi da match CALL_CHAIN)
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
    Node 3: Chay song song ca 2 path.

    Fast Path (Bivariate — PerformanceAgent):
      - Ap cung delta doc lap vao 21 bivariate model
      - Nhanh, khong co cascade effect
      - Dung cho: quick feasibility check, cross-check

    Accurate Path (Multi-hop Global DAG — SimulationAgent):
      - do(injection_service_workload = baseline x (1 + delta/100))
      - SCM tu lan truyen qua call chain, phan anh cascade attenuation
      - Dung cho: quyet dinh chinh, blast radius chinh xac
    """
    print("\n[Phase 3] Chay Dual-Path SCM Simulation...")

    parsed = state['parsed_requirement']
    delta  = parsed['injection_delta_pct']
    inj    = parsed['injection_service']

    # Gom service can kiem tra (Core + Blast Radius + Upstream/Downstream)
    services_to_check = set(parsed.get('affected_services', []))
    services_to_check.update(state['core_services'])
    for srv, deps in state['impact_graph'].items():
        services_to_check.update(deps.get('api_consumers_to_notify', []))
        services_to_check.update(deps.get('downstream_services_to_check', []))

    # --- FAST PATH ---
    print(f"  [Fast Path] Ap delta={delta}% vao Bivariate model ({len(services_to_check)} dich vu)...")
    fast_metrics = {}
    for srv in services_to_check:
        m = perf_agent.get_metrics_for_service(srv, workload_delta_pct=delta)
        if m:
            fast_metrics[srv] = m
    print(f"  [Fast Path] Xong: {len(fast_metrics)} dich vu co du lieu.")

    # --- ACCURATE PATH ---
    print(f"  [Accurate Path] do({inj}_workload +{delta}%) qua Global DAG 28-node...")
    sim_result = sim_agent.simulate_intervention(inj, delta)
    print(f"  [Accurate Path] Xong: {len(sim_result)} dich vu trong DAG.")

    return {
        "performance_metrics": fast_metrics,
        "simulation_result":   sim_result,
    }


def generate_report_node(state: RequirementState) -> dict:
    """
    Node 4: LLM tong hop Bao cao Kha thi tu ca 2 nguon du lieu.
    Accurate Path (multi-hop) duoc uu tien cho quyet dinh.
    Fast Path dung de cross-check.
    """
    print("\n[Phase 4] Tong hop Bao cao Kha thi (LLM)...")

    parsed        = state['parsed_requirement']
    delta         = parsed['injection_delta_pct']
    scm_accuracy  = perf_agent.get_accuracy_summary()
    dag_summary   = sim_agent.get_dag_summary()

    system_prompt = """Ban la Ky su Truong (Principal Engineer) chuyen gia Microservices.
Nhiem vu: danh gia tinh kha thi khi them tinh nang moi vao kien truc vi dich vu.
Phan tich tren 3 phuong dien:
  1. Tac dong kien truc (code & API changes).
  2. Rui ro tai nguyen — dua tren du lieu dinh luong tu 2 mo hinh SCM.
  3. Ket luan kha thi: CO THE / CAN SCALE TRUOC."""

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

=== TAC DONG KIEN TRUC (Do thi API) ===
{state['impact_graph']}

=== DU DOAN TAI NGUYEN — FAST PATH (Bivariate, delta ap deu {delta}%) ===
{state['performance_metrics']}
[Luu y: Fast Path KHONG co cascade suy giam — delta ap deu cho moi service]

=== DU DOAN TAI NGUYEN — ACCURATE PATH (Global DAG {dag_summary.get('n_nodes',28)}-node, do-calculus) ===
{state['simulation_result']}
[Luu y: Accurate Path phan anh cascade attenuation thuc te — service cang nhieu hop cang nhan it hon]

=== DO CHINH XAC MO HINH (tu benchmark Q1) ===
{scm_accuracy}
[Tieu chi: MAPE <10% = EXCELLENT, 10-25% = FAIR, >25% = POOR]

=== YEU CAU BAO CAO ===
1. TAC DONG KIEN TRUC: Liet ke API can them/sua, ai goi ai.
2. PHAN TICH RUI RO TAI NGUYEN:
   - Uu tien ket qua Accurate Path (cascade thuc te) cho nhan dinh chinh.
   - Neu Fast Path va Accurate Path chenh nhau lon (>10%) -> giai thich cascade effect.
   - Chi ra service nao co nguy co qua tai cao nhat.
3. KET LUAN KHA THI:
   - CO THE trien khai ngay: neu tat ca service trong blast radius con buffer tai nguyen.
   - CAN SCALE TRUOC: neu bat ky service nao co du bao CPU/Mem qua 70% baseline.
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
