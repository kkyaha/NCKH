# -*- coding: utf-8 -*-
"""
MAS Full Pipeline Test — End-to-End with Detailed Logging
==========================================================
This script executes the COMPLETE system pipeline from start to finish:

  1. Agent Initialization (Architecture, Performance, Simulation, Parser)
  2. Phase 1: Parse Requirement (Rule-based + LLM)
  3. Phase 2: Map Impact (Graph BFS blast radius)
  4. Phase 3: Dual-Path SCM Simulation (Fast + Accurate)
  5. Phase 4: Generate Feasibility Report (LLM synthesis)

All output is logged to both console and a timestamped log file.

Usage:
    .venv/bin/python run_full_pipeline_test.py
"""

import os
import sys
import time
import logging
from datetime import datetime

# ============================================================
# LOGGING SETUP — Console + File
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE  = os.path.join(LOG_DIR, f'full_pipeline_{timestamp}.log')

# Create logger
logger = logging.getLogger('MAS_Pipeline')
logger.setLevel(logging.DEBUG)

# File handler — full detail
fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Console handler — key info
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(fh)
logger.addHandler(ch)


def log_banner(title, char='=', width=80):
    logger.info('')
    logger.info(char * width)
    pad = (width - len(title) - 2) // 2
    logger.info(f"{' ' * pad} {title}")
    logger.info(char * width)


def log_separator(char='-', width=80):
    logger.info(char * width)


def log_phase(n, title):
    logger.info('')
    logger.info(f"{'='*80}")
    logger.info(f"  PHASE {n}: {title}")
    logger.info(f"{'='*80}")


def log_metric(label, value, indent=4):
    logger.info(f"{' ' * indent}{label:<40}: {value}")


# ============================================================
# TEST SCENARIOS
# ============================================================
TEST_SCENARIOS = [
    {
        "name": "Scenario 1 — Apply Promo Code (Medium Load, 4 services)",
        "requirement": (
            "Customers want a new feature to apply a discount promo code (Voucher) "
            "on the checkout screen before confirming the order."
        ),
    },
    {
        "name": "Scenario 2 — Product Recommendations (Heavy Load, 4 services)",
        "requirement": (
            "Add an AI-powered product recommendation engine that suggests items "
            "based on the customer's purchase history and browsing patterns."
        ),
    },
    {
        "name": "Scenario 3 — Package Tracking (Light Load, 3 services)",
        "requirement": (
            "Customers want to track their package delivery status in real-time "
            "with live updates on the shipping progress."
        ),
    },
]


# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================
def run_single_scenario(scenario, arch_agent, perf_agent, sim_agent, parser_agent, llm, feasibility_analyzer):
    """Run one complete scenario through the full pipeline with detailed logging."""

    log_banner(scenario['name'])
    logger.info(f"  Requirement: \"{scenario['requirement']}\"")
    t_total = time.time()

    # ========================================
    # PHASE 1: PARSE REQUIREMENT
    # ========================================
    log_phase(1, "PARSE REQUIREMENT (ParserAgent)")
    t0 = time.time()

    from dataclasses import asdict
    parsed = parser_agent.parse(scenario['requirement'])
    p_dict = asdict(parsed)

    t1 = time.time()
    logger.info('')
    log_metric("Request Type",        parsed.request_type)
    log_metric("Matched Template",    parsed.matched_template)
    log_metric("Template Delta",      f"{parsed.template_delta}%")
    log_metric("LLM Adjustment",      f"{parsed.adjustment:+.1f}%")
    log_metric("Final Delta (post-guard)", f"{parsed.injection_delta_pct}%")
    log_metric("Injection Service",   parsed.injection_service)
    log_metric("Core Services",       str(parsed.core_services))
    log_metric("Affected Services",   str(parsed.affected_services))
    log_metric("Similarity Score",    f"{parsed.similarity_score:.2f}")
    log_metric("Confidence",          parsed.confidence)
    log_metric("LLM Was Called",      str(parsed.llm_was_called))
    log_metric("Reasoning",           parsed.reasoning[:120] + '...' if len(parsed.reasoning) > 120 else parsed.reasoning)
    logger.info('')
    log_metric("Phase 1 Duration",    f"{t1 - t0:.2f}s")

    # ========================================
    # PHASE 2: MAP IMPACT (Architecture Agent)
    # ========================================
    log_phase(2, "MAP IMPACT (ArchitectureAgent — Graph BFS)")
    t0 = time.time()

    impact_mapping = {}
    affected = parsed.affected_services or []
    base_services = set(parsed.core_services) | set(affected)

    for service in base_services:
        if service not in arch_agent.graph:
            logger.debug(f"  [SKIP] {service} not in graph")
            continue
        upstream   = arch_agent.get_upstream_dependencies(service)
        downstream = arch_agent.get_downstream_dependencies(service)
        impact_mapping[service] = {
            "api_consumers_to_notify":      upstream,
            "downstream_services_to_check": downstream,
        }
        logger.info(f"    {service:<14} | upstream={upstream} | downstream={downstream}")

    t1 = time.time()
    logger.info('')
    log_metric("Services Scanned",    len(impact_mapping))
    log_metric("Phase 2 Duration",    f"{t1 - t0:.4f}s")

    # ========================================
    # PHASE 3: DUAL-PATH SCM SIMULATION
    # ========================================
    log_phase(3, "DUAL-PATH SCM SIMULATION")
    delta = parsed.injection_delta_pct
    inj   = parsed.injection_service

    # Collect all services to check
    services_to_check = set(affected)
    services_to_check.update(parsed.core_services)
    for srv, deps in impact_mapping.items():
        services_to_check.update(deps.get('api_consumers_to_notify', []))
        services_to_check.update(deps.get('downstream_services_to_check', []))

    # --- FAST PATH (Bivariate) ---
    logger.info('')
    logger.info(f"  [FAST PATH] Bivariate SCM — delta={delta}% applied independently to {len(services_to_check)} services")
    log_separator()
    t0 = time.time()

    fast_metrics = {}
    for srv in sorted(services_to_check):
        m = perf_agent.get_metrics_for_service(srv, workload_delta_pct=delta)
        if m:
            fast_metrics[srv] = m
            cpu_chg = m.get('CPU_change_pct', 'N/A')
            mem_chg = m.get('Memory_change_pct', 'N/A')
            logger.info(f"    {srv:<14} | CPU: {cpu_chg:>+8}% | Mem: {mem_chg:>+8}%")
        else:
            logger.info(f"    {srv:<14} | [NO DATA]")

    t1 = time.time()
    logger.info('')
    log_metric("Fast Path Services",  len(fast_metrics))
    log_metric("Fast Path Duration",  f"{t1 - t0:.2f}s")
    logger.info('')
    logger.info(f"  [NOTE] Fast Path applies SAME delta ({delta}%) to ALL services — no cascade attenuation")

    # --- ACCURATE PATH (Global DAG) ---
    logger.info('')
    logger.info(f"  [ACCURATE PATH] Global DAG 28-node — do({inj}_workload += {delta}%)")
    log_separator()
    t0 = time.time()

    sim_result = {}
    if sim_agent._is_trained:
        sim_result = sim_agent.simulate_intervention(inj, delta)
        logger.info('')
        logger.info(f"    {'Service':<14} | {'Hops':>4} | {'CPU Δ':>10} | {'Mem Δ':>10} | {'Lat Δ':>10}")
        logger.info(f"    {'-'*60}")
        for svc in ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']:
            if svc not in sim_result:
                continue
            r = sim_result[svc]
            hops = r.get('n_hops', -1)
            cpu  = r.get('cpu_change_pct')
            mem  = r.get('mem_change_pct')
            lat  = r.get('latency_change_pct')
            logger.info(
                f"    {svc:<14} | {hops:>4} | "
                f"{(str(round(cpu,1))+'%') if cpu is not None else 'N/A':>10} | "
                f"{(str(round(mem,1))+'%') if mem is not None else 'N/A':>10} | "
                f"{(str(round(lat,1))+'%') if lat is not None else 'N/A':>10}"
            )
    else:
        logger.warning("  [SKIP] SimulationAgent not trained.")

    t1 = time.time()
    logger.info('')
    log_metric("Accurate Path Services", len(sim_result))
    log_metric("Accurate Path Duration", f"{t1 - t0:.2f}s")

    # --- DIFF ANALYSIS ---
    if fast_metrics and sim_result:
        logger.info('')
        logger.info(f"  [DIFF ANALYSIS] Fast Path vs Accurate Path (CPU change %)")
        log_separator()
        logger.info(f"    {'Service':<14} | {'Fast (Biv.)':>12} | {'Accurate (DAG)':>14} | {'Diff':>8} | {'Hops':>4}")
        logger.info(f"    {'-'*65}")
        for svc in sorted(services_to_check):
            fast_cpu = fast_metrics.get(svc, {}).get('CPU_change_pct')
            dag_r    = sim_result.get(svc, {})
            dag_cpu  = dag_r.get('cpu_change_pct')
            hops     = dag_r.get('n_hops', '?')
            if fast_cpu is not None and dag_cpu is not None:
                diff = abs(fast_cpu - dag_cpu)
                flag = ' <<<' if diff > 10 else ''
                logger.info(
                    f"    {svc:<14} | {fast_cpu:>+11.1f}% | {dag_cpu:>+13.1f}% | {diff:>7.1f}% | {hops:>4}{flag}"
                )

    # ========================================
    # PHASE 4: GENERATE FEASIBILITY REPORT (LLM)
    # ========================================
    log_phase(4, "GENERATE FEASIBILITY REPORT (LLM Synthesis)")
    t0 = time.time()

    from langchain_core.messages import SystemMessage, HumanMessage

    dag_summary = sim_agent.get_dag_summary()
    scm_accuracy = perf_agent.get_accuracy_summary()

    system_prompt = """You are a Principal Microservices Engineer.
Task: Evaluate the feasibility of adding a new feature to the microservices architecture.
Analyze 3 aspects:
  1. Architectural impact (code & API changes required).
  2. Resource risk — based on quantitative data from 2 SCM models.
  3. Final verdict: FEASIBLE / NEEDS SCALING FIRST."""

    human_prompt = f"""
=== NEW FEATURE REQUIREMENT ===
{scenario['requirement']}

=== PARSER ANALYSIS ===
  Request type: {parsed.request_type}
  Services needing code changes: {parsed.core_services}
  Blast radius: {parsed.affected_services}
  Workload delta at gateway: +{delta}%
  Confidence: {parsed.confidence}

=== ARCHITECTURE IMPACT (API Graph) ===
{impact_mapping}

=== RESOURCE PREDICTION — FAST PATH (Bivariate, delta={delta}% applied equally) ===
{fast_metrics}
[Note: Fast Path does NOT model cascade attenuation — delta applied equally to all services]

=== RESOURCE PREDICTION — ACCURATE PATH (Global DAG {dag_summary.get('n_nodes',28)}-node, do-calculus) ===
{sim_result}
[Note: Accurate Path reflects real cascade attenuation — services further from gateway receive less impact]

=== MODEL ACCURACY (from Q1 benchmark) ===
{scm_accuracy}
[Criteria: MAPE <10% = EXCELLENT, 10-25% = FAIR, >25% = POOR]

=== REPORT REQUIREMENTS ===
1. ARCHITECTURAL IMPACT: List APIs to add/modify, who calls whom.
2. RESOURCE RISK ANALYSIS:
   - Prioritize Accurate Path (real cascade) for main conclusions.
   - If Fast Path and Accurate Path differ by >10% -> explain cascade effect.
   - Identify the service with highest overload risk.
3. FINAL VERDICT:
   - FEASIBLE: if all services in blast radius have sufficient resource buffer.
   - NEEDS SCALING FIRST: if any service is predicted to exceed 70% baseline CPU/Mem.
"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    logger.info("  Calling LLM for final report synthesis...")
    response = llm.invoke(messages)
    report = response.content

    t1 = time.time()
    logger.info('')
    log_metric("Phase 4 Duration", f"{t1 - t0:.2f}s")

    # ========================================
    # FINAL REPORT OUTPUT
    # ========================================
    logger.info('')
    log_banner("FEASIBILITY REPORT", '▓')
    for line in report.split('\n'):
        logger.info(f"  {line}")

    t_end = time.time()
    logger.info('')
    log_separator('=')
    log_metric("TOTAL PIPELINE TIME", f"{t_end - t_total:.2f}s")
    log_separator('=')

    return {
        'scenario': scenario['name'],
        'parsed': p_dict,
        'impact': impact_mapping,
        'fast_path': fast_metrics,
        'accurate_path': sim_result,
        'report': report,
        'total_time': round(t_end - t_total, 2),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    log_banner("MAS FULL PIPELINE TEST — End-to-End Execution", '█')
    logger.info(f"  Log file: {LOG_FILE}")
    logger.info(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Scenarios: {len(TEST_SCENARIOS)}")

    # ========================================
    # AGENT INITIALIZATION
    # ========================================
    log_phase(0, "AGENT INITIALIZATION")
    t_init = time.time()

    # -- Setup paths --
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))

    GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')
    DATA_DIR   = os.path.join(PROJECT_ROOT, 'data', 'raw')

    # -- LLM --
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not set. Vui long them GOOGLE_API_KEY vao file .env hoac export GOOGLE_API_KEY='your-key'")
        sys.exit(1)
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
    logger.info("  [OK] LLM initialized (Gemini 3.6 Flash)")

    # -- Architecture Agent --
    from agents.architecture_agent import ArchitectureAgent
    arch_agent = ArchitectureAgent(GRAPH_PATH)
    n_nodes = arch_agent.graph.number_of_nodes()
    n_edges = arch_agent.graph.number_of_edges()
    logger.info(f"  [OK] ArchitectureAgent loaded ({n_nodes} nodes, {n_edges} edges)")

    # -- Performance Agent (Bivariate SCM) --
    from agents.performance_agent import PerformanceAgent
    perf_agent = PerformanceAgent(data_dir=DATA_DIR, auto_train=True)
    n_models = len(perf_agent.trained_models)
    logger.info(f"  [OK] PerformanceAgent trained ({n_models} bivariate models)")

    # -- Simulation Agent (Global DAG) --
    from agents.simulation_agent import SimulationAgent
    sim_agent = SimulationAgent(data_dir=DATA_DIR, graph_path=GRAPH_PATH)
    sim_agent.train()
    if sim_agent._is_trained:
        dag = sim_agent.dag
        logger.info(f"  [OK] SimulationAgent trained (DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges)")
    else:
        logger.warning("  [WARN] SimulationAgent could not train — missing data")

    # -- Parser Agent --
    from agents.parser_agent import ParserAgent
    parser_agent = ParserAgent(llm=llm, arch_agent=arch_agent)
    logger.info(f"  [OK] ParserAgent initialized")

    t_init_end = time.time()
    logger.info('')
    log_metric("Total Init Duration", f"{t_init_end - t_init:.2f}s")

    # ========================================
    # RUN ALL SCENARIOS
    # ========================================
    all_results = []
    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        logger.info(f"\n\n{'#' * 80}")
        logger.info(f"  RUNNING SCENARIO {i}/{len(TEST_SCENARIOS)}")
        logger.info(f"{'#' * 80}")

        result = run_single_scenario(
            scenario, arch_agent, perf_agent, sim_agent, parser_agent, llm,
            feasibility_analyzer=None  # Not using LangGraph compiled graph, running manually
        )
        all_results.append(result)

    # ========================================
    # SUMMARY
    # ========================================
    log_banner("EXECUTION SUMMARY", '█')
    logger.info(f"  {'Scenario':<55} | {'Time':>8} | {'Delta':>6} | {'Type':<20}")
    log_separator()
    for r in all_results:
        p = r['parsed']
        logger.info(
            f"  {r['scenario']:<55} | {r['total_time']:>6.1f}s | "
            f"{p['injection_delta_pct']:>5.0f}% | {p['request_type']:<20}"
        )

    logger.info('')
    logger.info(f"  Log saved to: {LOG_FILE}")
    logger.info(f"  Finished:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_separator('=')


if __name__ == '__main__':
    main()
