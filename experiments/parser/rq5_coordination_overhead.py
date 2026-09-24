# -*- coding: utf-8 -*-
"""
RQ5 — Multi-Agent Coordination Overhead vs. Single-LLM-Call Baseline
======================================================================
RQ5: "Co the thay the kien truc da tac tu (Parser Agent -> Architecture Agent
      -> Capacity Agent voi dual-path SCM) bang MOT lan goi LLM duy nhat
      (single-shot, tu nhien ngon ngu -> phan quyet kha thi) hay khong, va
      neu khong, chi phi dieu phoi da tac tu (LLM calls, latency) dem lai
      gia tri gi de doi lay?"

Thiet ke (tuong tu parser_benchmark_suite.py cho RQ3, dung lai cung 50 prompt
benchmark de so sanh cong bang):

  A. Single_LLM_Call  : MOT prompt duy nhat, duoc cho CUNG mot ngu canh (danh
                         sach service + bang hieu chinh, giong B2-FewShot cua
                         RQ3), phai tu suy ra CA: injection_service, delta,
                         core_services, VA tu "doan" luon trang thai nang luc
                         (status/saturated_services/verdict) MA KHONG CO SCM
                         nao chay ca — hoan toan dua vao "linh cam" cua LLM.
  B. Guarded_MAS      : ParserAgent that (co guard) -> Architecture Agent BFS
                         (deterministic, khong LLM) -> CapacityAgent that
                         (dual-path SCM that su duoc huan luyen tren du lieu
                         RCAEval, + 1 LLM call de reasoning/critique). Trang
                         thai (SAFE/WARNING/CRITICAL) o day la san pham cua
                         mo hinh nhan qua that, khong phai LLM doan.

QUAN TRONG VE DIEN GIAI: khong co "ground truth" thuc te cho mot tinh nang
CHUA duoc xay dung — vi vay chi so "agreement" o day KHONG phai "accuracy so
voi su that", ma la "ty le Single_LLM_Call trung khop voi phan quyet duoc
SCM-do(x) hau thuan". Bat dong khong tu dong co nghia Single_LLM_Call sai,
nhung mot he thong khong co co che nao de kiem tra lai chinh no (SCM, guard)
thi khong co cach nao phan biet duoc 2 truong hop do — day chinh la luan diem
RQ5 muon do luong: chi phi dieu phoi da tac tu doi lay duoc kha nang GIAI
THICH VA KIEM CHUNG DUOC, khong chi la mot con so cuoi cung.

Metric bao cao:
  - PBVR / SHR / GMR / Anchor MAE  : giong het RQ3, ap dung cho phan
    injection_service/delta/core_services ma Single_LLM_Call tu doan (de so
    sanh truc tiep voi Bang RQ3 — day la mot cau hinh "kho hon" B1/B2 vi con
    phai lam luon phan capacity trong CUNG mot lan goi).
  - Status Agreement (%)           : ty le Single_LLM_Call.status ==
    Guarded_MAS.status (SAFE/WARNING/CRITICAL), tren cac prompt KHONG bi G6
    tu choi (is_out_of_scope) o nhanh Guarded_MAS.
  - LLM calls / prompt             : chi phi dieu phoi (cost proxy) —
    Single_LLM_Call = 1; Guarded_MAS = 0..2 (Parser co the fast-path,
    Capacity luon goi 1 lan reasoning; KHONG tinh node 4 Report Synthesis vi
    node do la text-synthesis khong duoc danh gia dinh luong trong paper —
    xem Section III-A).
  - Latency (ms) / prompt          : tong thoi gian goi LLM (khong tinh thoi
    gian train SCM 1 lan dau, vi do la chi phi co dinh dung chung cho toan
    bo benchmark, khong phai chi phi per-request).
"""

import os
import sys
import csv
import time
import json
import warnings
from dataclasses import asdict

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # de import parser_benchmark_suite

from request_router import CALL_CHAINS
from architecture_agent import ArchitectureAgent
from capacity_agent import CapacityAgent
from parser_agent import ParserAgent, KNOWN_SERVICES

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Dung lai dung 1 ham get_llm() da kiem chung o RQ3 (cung logic: live API that,
# fallback offline emulator co canh bao ro rang — tranh code trung lap/lech pha).
from parser_benchmark_suite import get_llm

GRAPH_PATH   = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROMPTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'parser_benchmark_prompts.json')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
DOCS_DIR     = os.path.join(PROJECT_ROOT, 'docs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


# ============================================================
# 1. LLM CALL COUNTER / LATENCY WRAPPER
# ============================================================
class CountingLLM:
    """
    Boc mot LLM that/gia lap de dem so lan goi va tong latency — dung chung
    cho ca 2 nhanh (Single_LLM_Call va Guarded_MAS) de phep so sanh "chi phi
    dieu phoi" cong bang (cung 1 backend, cung 1 co che rate-limit).
    """
    def __init__(self, inner_llm, is_synthetic: bool, sleep_between_calls: float = 4.5):
        self.inner = inner_llm
        self.is_synthetic = is_synthetic
        self.sleep_between_calls = sleep_between_calls
        self.call_count = 0
        self.total_latency_ms = 0.0

    def reset(self):
        self.call_count = 0
        self.total_latency_ms = 0.0

    def invoke(self, messages):
        t0 = time.time()
        resp = self.inner.invoke(messages)
        elapsed = (time.time() - t0) * 1000
        self.call_count += 1
        self.total_latency_ms += elapsed
        # Rate-limit Free Tier (giong het RQ3): ~4.5s giua cac lan goi that.
        if not self.is_synthetic:
            time.sleep(self.sleep_between_calls)
        return resp


def _clean_json(raw: str) -> dict:
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ============================================================
# 2. BASELINE A: SINGLE-LLM-CALL (khong SCM, khong guard, khong da tac tu)
# ============================================================
class SingleLLMCallBaseline:
    """
    Mot prompt DUY NHAT thay the toan bo pipeline 3 buoc dau (Parser ->
    Architecture -> Capacity): duoc cho CUNG ngu canh (danh sach service +
    bang hieu chinh) nhu B2-FewShot cua RQ3, nhung phai TU DOAN LUON trang
    thai nang luc (status/saturated_services) thay vi de mot SCM that tinh.
    """

    def __init__(self, llm: CountingLLM, services_ctx: str, calibration_ctx: str):
        self.llm = llm
        self.services_ctx = services_ctx
        self.calibration_ctx = calibration_ctx

    def run(self, requirement: str) -> dict:
        prompt = f"""Ban la mot Kien truc su Truong (Principal Engineer) chuyen gia microservices,
lam TOAN BO cong viec danh gia kha thi mot minh trong MOT LAN TRA LOI DUY NHAT
(khong co cong cu do luong nao khac ho tro ban — khong SCM, khong do thi phu
thuoc duoc tinh toan san, chi co kien thuc va uoc luong cua chinh ban).

[CAC DICH VU TRONG HE THONG]
{self.services_ctx}

[BANG HIEU CHINH (Calibration Table — tu du lieu thuc nghiem)]
{self.calibration_ctx}

[YEU CAU TINH NANG MOI]
"{requirement}"

[NHIEM VU — TRA LOI TAT CA TRONG MOT LAN]
1. "injection_service": dich vu nao nhan tai tang them dau tien (thuong la
   gateway/front-end).
2. "injection_delta_pct": % tang tai uoc tinh tai injection_service.
3. "core_services": cac service CAN SUA CODE.
4. "affected_services": toan bo blast radius (cac service bi anh huong day
   chuyen), uoc luong theo hieu biet cua ban ve kien truc microservices dien
   hinh (khong co do thi phu thuoc that de tra cuu).
5. "status": uoc luong tinh trang nang luc he thong sau khi trien khai —
   "SAFE" | "WARNING" | "CRITICAL".
6. "saturated_services": danh sach service ban DU DOAN se cham nguong bao
   hoa (neu co).
7. "verdict": "CAN_DEPLOY" (co the trien khai ngay) hoac "NEEDS_SCALING"
   (can scale truoc khi trien khai).
8. "confidence": "HIGH" | "MEDIUM" | "LOW".

Tra ve DUY NHAT JSON sau, KHONG them text khac:
{{
  "injection_service": "...",
  "injection_delta_pct": <float>,
  "core_services": ["..."],
  "affected_services": ["..."],
  "status": "SAFE|WARNING|CRITICAL",
  "saturated_services": ["..."],
  "verdict": "CAN_DEPLOY|NEEDS_SCALING",
  "confidence": "HIGH|MEDIUM|LOW"
}}"""
        from langchain_core.messages import HumanMessage
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            data = _clean_json(resp.content)
            return {
                'injection_service':   str(data.get('injection_service', 'front-end')),
                'injection_delta_pct': float(data.get('injection_delta_pct', 20.0)),
                'core_services':       list(data.get('core_services', ['front-end'])),
                'affected_services':   list(data.get('affected_services', ['front-end'])),
                'status':              str(data.get('status', 'SAFE')).upper(),
                'saturated_services':  list(data.get('saturated_services', [])),
                'verdict':             str(data.get('verdict', 'CAN_DEPLOY')).upper(),
                'confidence':          str(data.get('confidence', 'MEDIUM')).upper(),
                'parse_error':         False,
            }
        except Exception as e:
            return {
                'injection_service': 'front-end', 'injection_delta_pct': 20.0,
                'core_services': ['front-end'], 'affected_services': ['front-end'],
                'status': 'SAFE', 'saturated_services': [], 'verdict': 'CAN_DEPLOY',
                'confidence': 'LOW', 'parse_error': True, 'error_msg': str(e),
            }


# ============================================================
# 3. BASELINE B: GUARDED MULTI-AGENT PIPELINE (proposed)
# ============================================================
class GuardedMASRunner:
    """
    Tai hien 3 node dau cua orchestrator.py (Parser -> Architecture(BFS) ->
    Capacity(dual-path SCM that + 1 LLM reasoning call)), KHONG bao gom node
    4 (Report Synthesis) vi node do la text-synthesis khong duoc danh gia
    dinh luong trong paper (Section III-A) — bao gom no se lam chi phi "cost"
    cua ca 2 nhanh lech nhau khong can thiet (single-call van co the them 1
    cau LLM synthesis van ban tuong duong ma khong thay doi status/verdict
    da tinh o buoc truoc).
    """

    def __init__(self, parser_agent: ParserAgent, arch_agent: ArchitectureAgent,
                 capacity_agent: CapacityAgent):
        self.parser = parser_agent
        self.arch = arch_agent
        self.capacity = capacity_agent

    def run(self, requirement: str) -> dict:
        parsed = self.parser.parse(requirement)
        parsed_d = asdict(parsed)

        if parsed.is_out_of_scope:
            # Giong het G6 short-circuit trong orchestrator.generate_report_node:
            # tu choi dua ra phan quyet dinh luong, KHONG goi CapacityAgent.
            return {
                'injection_service':   parsed.injection_service,
                'injection_delta_pct': parsed.injection_delta_pct,
                'core_services':       parsed.core_services,
                'affected_services':   parsed.affected_services,
                'status':              'REFUSED',
                'saturated_services':  [],
                'verdict':             'REFUSED',
                'confidence':          'REFUSED',
                'parse_error':         False,
                'llm_was_called_parser': parsed.llm_was_called,
                'is_out_of_scope':     True,
            }

        # Architecture Agent: BFS thuan tuy, KHONG LLM.
        impact_graph = {}
        base_services = set(parsed.core_services) | set(parsed.affected_services)
        for s in base_services:
            if s not in self.arch.graph:
                continue
            impact_graph[s] = {
                'api_consumers_to_notify': self.arch.get_upstream_dependencies(s),
                'downstream_services_to_check': self.arch.get_downstream_dependencies(s),
            }

        assessment = self.capacity.assess_capacity(parsed_d, impact_graph)
        verdict = 'CAN_DEPLOY' if assessment.status == 'SAFE' else 'NEEDS_SCALING'

        return {
            'injection_service':   parsed.injection_service,
            'injection_delta_pct': parsed.injection_delta_pct,
            'core_services':       parsed.core_services,
            'affected_services':   parsed.affected_services,
            'status':              assessment.status,
            'saturated_services':  assessment.saturated_services,
            'verdict':             verdict,
            'confidence':          assessment.confidence,
            'parse_error':         False,
            'llm_was_called_parser': parsed.llm_was_called,
            'is_out_of_scope':     False,
        }


# ============================================================
# 4. BENCHMARK EXECUTION ENGINE
# ============================================================
def run_rq5_benchmark(use_live_api=True, configs_to_run=None):
    """
    configs_to_run: neu chi dinh (vd ['Guarded_MAS_Pipeline']), CHI chay lai
    cac config nay bang LLM that; config con lai duoc TAI SU DUNG tu CSV cu
    (khong tra API) -- dung khi mot ban sua code CHI anh huong 1 nhanh (vd
    sua ParserAgent/Scope Gate chi anh huong Guarded_MAS_Pipeline, khong
    anh huong Single_LLM_Call vi nhanh do khong dung ParserAgent), giong
    pattern --only= cua parser_benchmark_suite.py. Tranh ton quota re-run
    lai nhung gi khong the doi.
    """
    if '--live-llm' in sys.argv:
        use_live_api = True
    if '--offline' in sys.argv:
        use_live_api = False

    n_repeats = int(os.environ.get('RQ5_REPEATS', 3))
    for arg in sys.argv:
        if arg.startswith('--repeats='):
            n_repeats = int(arg.split('=', 1)[1])
        if arg.startswith('--only='):
            configs_to_run = arg.split('=', 1)[1].split(',')

    print("=" * 75)
    print("   RUNNING RQ5: MULTI-AGENT COORDINATION OVERHEAD vs SINGLE-LLM-CALL")
    print(f"   Repeats per configuration: {n_repeats} | Live API requested: {use_live_api}")
    print("=" * 75)

    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} requirement test cases.")

    # 1 LLM backend dung chung cho CA 2 nhanh, de so sanh cong bang.
    raw_llm, llm_backend = get_llm(use_live_api=use_live_api)
    is_synthetic = llm_backend.startswith("SYNTHETIC")
    counting_llm = CountingLLM(raw_llm, is_synthetic=is_synthetic)
    if is_synthetic:
        print("\n" + "!" * 75)
        print("  CANH BAO: Dang chay voi BO GIA LAP OFFLINE, khong phai LLM that.")
        print("  Ket qua nay CHI de kiem tra code, KHONG dua vao bai bao khoa hoc.")
        print("!" * 75 + "\n")

    # Cac thanh phan dung chung, huan luyen 1 LAN DUY NHAT roi tai su dung cho
    # tat ca prompt/repeat (giong pattern cua parser_benchmark_suite.py) —
    # tranh train lai SCM 50 x n_repeats lan mot cach lang phi.
    print("\n[Setup] Huan luyen CapacityAgent (Dual-Path SCM that su)...")
    arch_agent = ArchitectureAgent(GRAPH_PATH)
    capacity_agent = CapacityAgent(llm=counting_llm, data_dir=DATA_DIR,
                                    graph_path=GRAPH_PATH, auto_train=True)
    parser_agent = ParserAgent(llm=counting_llm, arch_agent=arch_agent)

    single_call = SingleLLMCallBaseline(
        llm=counting_llm,
        services_ctx=parser_agent._services_ctx,
        calibration_ctx=parser_agent._calibration_ctx,
    )
    mas_runner = GuardedMASRunner(parser_agent, arch_agent, capacity_agent)

    all_model_configs = ['Single_LLM_Call', 'Guarded_MAS_Pipeline']
    run_selected = configs_to_run if configs_to_run else all_model_configs
    all_results = []
    run_timestamp = pd.Timestamp.now().isoformat()

    # Ghi CSV NGAY sau moi prompt (khong doi den cuoi) — Free Tier API co the het
    # quota giua chung mot lan chay (da tung xay ra), va truoc day toan bo ket
    # qua da tinh xong bi mat trang vi df.to_csv() chi goi mot lan o cuoi. Gio
    # neu process bi kill/loi giua chung, phan da chay van duoc giu lai tren dia.
    fieldnames = [
        'repeat_id', 'llm_backend', 'run_timestamp', 'config', 'prompt_id',
        'category', 'requirement', 'expected_anchor', 'pred_delta', 'abs_error',
        'injection_service', 'core_services', 'physical_boundary_violation',
        'service_hallucination', 'gateway_misdirection', 'status',
        'saturated_services', 'verdict', 'confidence', 'is_out_of_scope',
        'llm_calls', 'llm_latency_ms',
    ]
    csv_path = os.path.join(OUTPUT_DIR, 'rq5_coordination_overhead.csv')

    reused_rows = []
    if configs_to_run and os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        skipped = [c for c in all_model_configs if c not in run_selected]
        reused_rows = old_df[old_df['config'].isin(skipped)].to_dict('records')
        print(f"[REUSE] Keeping {len(reused_rows)} existing rows for unaffected "
              f"configuration(s) {skipped} (not re-run -- code path unchanged for these).")

    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    for row in reused_rows:
        row = {k: row.get(k) for k in fieldnames}
        all_results.append(row)
        csv_writer.writerow(row)
    csv_file.flush()

    print(f"\nEvaluating across {len(prompts)} scenarios x {n_repeats} repeats "
          f"x configuration(s) {run_selected} (backend={llm_backend})...")
    try:
        for repeat_id in range(1, n_repeats + 1):
            for p in prompts:
                p_id, cat, req = p['id'], p['category'], p['requirement']
                exp_anchor, exp_gw = p['expected_anchor'], p['expected_gateway']

                for config_name, runner in [
                    ('Single_LLM_Call', single_call),
                    ('Guarded_MAS_Pipeline', mas_runner),
                ]:
                    if config_name not in run_selected:
                        continue
                    print(f"  [repeat {repeat_id}/{n_repeats}] [{config_name}] {p_id}...")
                    counting_llm.reset()
                    out = runner.run(req)

                    delta = out['injection_delta_pct']
                    core = out['core_services']
                    pbv = (delta < 5.0) or (delta > 50.0)
                    hallucinated = [s for s in core if s not in KNOWN_SERVICES]
                    sh = len(hallucinated) > 0
                    gm = (out['injection_service'] != exp_gw)
                    ae = abs(delta - exp_anchor)

                    row = {
                        'repeat_id': repeat_id,
                        'llm_backend': llm_backend,
                        'run_timestamp': run_timestamp,
                        'config': config_name,
                        'prompt_id': p_id,
                        'category': cat,
                        'requirement': req,
                        'expected_anchor': exp_anchor,
                        'pred_delta': delta,
                        'abs_error': ae,
                        'injection_service': out['injection_service'],
                        'core_services': str(core),
                        'physical_boundary_violation': pbv,
                        'service_hallucination': sh,
                        'gateway_misdirection': gm,
                        'status': out['status'],
                        'saturated_services': str(out.get('saturated_services', [])),
                        'verdict': out['verdict'],
                        'confidence': out['confidence'],
                        'is_out_of_scope': out.get('is_out_of_scope', False),
                        'llm_calls': counting_llm.call_count,
                        'llm_latency_ms': counting_llm.total_latency_ms,
                    }
                    all_results.append(row)
                    csv_writer.writerow(row)
                    csv_file.flush()
    finally:
        csv_file.close()

    df = pd.DataFrame(all_results)
    expected_total = len(reused_rows) + len(prompts) * n_repeats * len(run_selected)
    print(f"\n[OK] Raw RQ5 results saved incrementally to: {csv_path} "
          f"({len(df)}/{expected_total} rows: {len(reused_rows)} reused + "
          f"{len(prompts) * n_repeats * len(run_selected)} freshly run)")

    # ============================================================
    # 5. STATUS AGREEMENT (Single_LLM_Call vs Guarded_MAS, khop theo prompt+repeat)
    #    Chi tinh tren cac prompt Guarded_MAS KHONG tu choi (is_out_of_scope=False),
    #    vi khi da tu choi thi khong co "status" nang luc nao de doi chieu.
    # ============================================================
    wide = df.pivot_table(
        index=['repeat_id', 'prompt_id', 'category'],
        columns='config',
        values='status',
        aggfunc='first'
    ).reset_index()

    scope_mask = df[(df['config'] == 'Guarded_MAS_Pipeline')][['repeat_id', 'prompt_id', 'is_out_of_scope']]
    wide = wide.merge(scope_mask, on=['repeat_id', 'prompt_id'], how='left')
    in_scope = wide[wide['is_out_of_scope'] == False].copy()
    in_scope['agree'] = in_scope['Single_LLM_Call'] == in_scope['Guarded_MAS_Pipeline']
    overall_agreement_pct = 100.0 * in_scope['agree'].mean() if len(in_scope) else float('nan')

    agreement_by_cat = (
        in_scope.groupby('category')['agree'].mean().mul(100).round(2)
        if len(in_scope) else pd.Series(dtype=float)
    )

    # ============================================================
    # 6. SUMMARY TABLE PER CONFIG (mean +/- std qua N_REPEATS)
    # ============================================================
    def _per_repeat_rate(sub_df, col):
        return [sub_df[sub_df['repeat_id'] == r][col].mean() * 100.0
                for r in sorted(sub_df['repeat_id'].unique())]

    def _per_repeat_mean(sub_df, col):
        return [sub_df[sub_df['repeat_id'] == r][col].mean()
                for r in sorted(sub_df['repeat_id'].unique())]

    def fmt(vals, suffix=''):
        arr = np.array(vals, dtype=float)
        if len(arr) <= 1 or np.isnan(arr).all():
            return f"{np.nanmean(arr):.2f}{suffix}"
        return f"{arr.mean():.2f}±{arr.std(ddof=1):.2f}{suffix}"

    summary_rows = []
    for config_name in ['Single_LLM_Call', 'Guarded_MAS_Pipeline']:
        c_df = df[df['config'] == config_name]
        summary_rows.append({
            'Configuration': config_name,
            'N Repeats': c_df['repeat_id'].nunique(),
            'PBVR (%)': fmt(_per_repeat_rate(c_df, 'physical_boundary_violation'), '%'),
            'SHR (%)': fmt(_per_repeat_rate(c_df, 'service_hallucination'), '%'),
            'GMR (%)': fmt(_per_repeat_rate(c_df, 'gateway_misdirection'), '%'),
            'Anchor MAE (%)': fmt(_per_repeat_mean(c_df, 'abs_error'), '%'),
            'LLM calls/prompt': fmt(_per_repeat_mean(c_df, 'llm_calls')),
            'LLM latency (ms)/prompt': fmt(_per_repeat_mean(c_df, 'llm_latency_ms'), ' ms'),
        })
    sum_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 85)
    print("                    RQ5 SUMMARY TABLE")
    print("=" * 85)
    print(sum_df.to_string(index=False))
    print(f"\nStatus agreement (Single_LLM_Call == Guarded_MAS_Pipeline), "
          f"in-scope prompts only: {overall_agreement_pct:.2f}% (n={len(in_scope)})")
    print("By category:")
    print(agreement_by_cat.to_string() if len(agreement_by_cat) else "(no in-scope rows)")
    print("=" * 85)

    agreement_csv = os.path.join(OUTPUT_DIR, 'rq5_status_agreement.csv')
    in_scope.to_csv(agreement_csv, index=False, encoding='utf-8')
    print(f"[OK] Per-prompt status agreement saved to: {agreement_csv}")

    # ============================================================
    # 7. GENERATE SCIENTIFIC MARKDOWN REPORT
    # ============================================================
    backend_warning = (
        f"\n> ⚠️ **CẢNH BÁO**: Chạy với bộ giả lập offline (`{llm_backend}`), "
        "KHÔNG phải LLM thật. Không dùng số liệu này làm bằng chứng khoa học.\n"
        if is_synthetic else
        f"\n> ✅ Backend: **LLM thật** (`{llm_backend}`), {n_repeats} lần lặp độc lập.\n"
    )

    sum_table_md = "| Configuration | PBVR (%) | SHR (%) | GMR (%) | Anchor MAE (%) | LLM calls/prompt | LLM latency (ms)/prompt |\n"
    sum_table_md += "|---|---|---|---|---|---|---|\n"
    for r in summary_rows:
        sum_table_md += (f"| `{r['Configuration']}` | {r['PBVR (%)']} | {r['SHR (%)']} | "
                          f"{r['GMR (%)']} | {r['Anchor MAE (%)']} | {r['LLM calls/prompt']} | "
                          f"{r['LLM latency (ms)/prompt']} |\n")

    agree_table_md = "| Category | Status Agreement (%) |\n|---|---|\n"
    for cat, val in agreement_by_cat.items():
        agree_table_md += f"| {cat} | {val} |\n"

    md_report = f"""# 📊 BÁO CÁO RQ5 — CHI PHÍ ĐIỀU PHỐI ĐA TÁC TỬ vs. SINGLE-LLM-CALL

Tài liệu này được **sinh tự động** từ `rq5_coordination_overhead.csv` để trả lời RQ5:
> **RQ5**: *"Kiến trúc đa tác tử (Parser Agent + Architecture Agent + Capacity Agent
> với dual-path SCM) có vượt trội một lời gọi LLM đơn lẻ (single-LLM-call) không,
> và chi phí điều phối đa tác tử (số lần gọi LLM, độ trễ) đem lại giá trị gì?"*
{backend_warning}
Thời điểm chạy: `{run_timestamp}` | Số lần lặp: `{n_repeats}` | Backend: `{llm_backend}`

---

## ⚠️ Giới hạn diễn giải cần đọc trước bảng số liệu

Không tồn tại "ground truth" thực tế cho một tính năng **chưa được xây dựng** — vì vậy
"Status Agreement" dưới đây **không phải** độ chính xác so với sự thật, mà là tỔc độ
`Single_LLM_Call` trùng khớp với phán quyết mà mô phỏng SCM $do(x)$ thật sự đưa ra
(`Guarded_MAS_Pipeline`, được tính từ các mô hình nhân quả đã huấn luyện trên dữ liệu
RCAEval thật, không phải LLM đoán). Bất đồng không tự động nghĩa là `Single_LLM_Call`
sai — nhưng nó **không có cơ chế nào để tự kiểm chứng lại chính nó** (không SCM, không
guard), nên không có cách nào phân biệt được hai trường hợp đó chỉ từ đầu ra của nó.
Đây chính là luận điểm RQ5 muốn đo: chi phí điều phối đa tác tử đổi lấy khả năng
**giải thích và kiểm chứng được**, không chỉ là một con số cuối cùng.

---

## 📊 1. BẢNG TỔNG HỢP (PBVR/SHR/GMR/MAE giống hệt phương pháp RQ3, cộng thêm chi phí)

{sum_table_md}
*Ghi chú*: `Single_LLM_Call` phải tự đoán CẢ phần capacity assessment trong CÙNG một
lần gọi — đây là cấu hình khó hơn B1/B2 của RQ3 (vốn chỉ phải làm mỗi việc parse).

---

## 🤝 2. STATUS AGREEMENT (Single_LLM_Call so với phán quyết SCM $do(x)$ thật)

Tổng thể: **{overall_agreement_pct:.2f}%** (n={len(in_scope)} prompt trong phạm vi taxonomy,
loại các prompt bị G6 từ chối ở nhánh Guarded_MAS vì không có status để đối chiếu).

Theo nhóm:

{agree_table_md}

---

## 📁 3. DỮ LIỆU THÔ
* Toàn bộ {len(df)} bản ghi ({len(prompts)} prompts × 2 cấu hình × {n_repeats} lần lặp):
  `data/processed/scm_results/rq5_coordination_overhead.csv`
* Chi tiết status agreement theo từng prompt: `data/processed/scm_results/rq5_status_agreement.csv`
"""
    md_path = os.path.join(DOCS_DIR, 'RQ5_COORDINATION_OVERHEAD_REPORT.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"[OK] Scientific markdown report generated: {md_path}")
    print("=" * 75)
    return df, sum_df, overall_agreement_pct


if __name__ == '__main__':
    run_rq5_benchmark()
