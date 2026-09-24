# -*- coding: utf-8 -*-
"""
Parser Benchmark Suite for Q1 Academic Paper (RQ3 Validation)
============================================================
Author: NCKH Research Team
Focus: Quantitative Ablation Study & Hallucination Elimination in LLM Requirement Parsing

This script executes a rigorous multi-baseline ablation study across 50 software
requirement prompts (spanning In-Distribution, Complex Multi-Hop, Subtle Read-Only,
and Adversarial Stress scenarios) to empirically evaluate RQ3:

  RQ3: "How reliably does the Grounded LLM Parser translate natural language requirements
        into valid causal interventions, and how effective are the Runtime Guards in
        eliminating hallucinations and physical boundary violations?"

Evaluated Configurations:
  1. Unguarded_ZeroShot_LLM : Naive unconstrained LLM prompt without anchors or guards.
  2. Unguarded_FewShot_LLM  : LLM with calibration context and examples, but without guards.
  3. Rule_Only              : Pure keyword-matching heuristic without semantic LLM adaptation.
  4. Guarded_Hybrid_Parser  : Proposed Two-Tier Grounded Parser with 5 Runtime Guards.

Academic Evaluation Metrics:
  - PBVR (%) : Physical Boundary Violation Rate (Delta < 5% or Delta > 50%)
  - SHR  (%) : Service Hallucination Rate (Services not in SockShop topology)
  - GMR  (%) : Gateway Misdirection Rate (Injection node is not in-degree=0 gateway)
  - MAE  (%) : Mean Absolute Error vs. Empirical Calibration Anchors
  - Latency  : Mean Inference Latency per prompt (milliseconds)
  - Bypass   : Fast-Path Rule Bypass Ratio (% prompts resolved without LLM calls)
"""

import os
import sys
import csv
import time
import json
import re
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Set single thread for reproducibility
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from request_router import CALL_CHAINS, classify_request
from architecture_agent import ArchitectureAgent
from parser_agent import ParserAgent, KNOWN_SERVICES, MIN_DELTA_PCT, MAX_DELTA_PCT

# Load environment
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')
PROMPTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'parser_benchmark_prompts.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


# ============================================================
# 1. LLM INITIALIZATION
# ============================================================
# Backend duoc chon theo TEN trong llm_backends.BACKENDS (khong phai ten model
# tho), de moi dong CSV ghi kem provider/tier/pinned phuc vu tai lap.
# Doi bang bien moi truong PARSER_BENCH_BACKEND hoac --backend=<ten>.
#
# LUU Y QUOTA (do lai ngay 2026-09-14, cac so cu da khong con dung):
#   gemini-flash-lite : quota free tier rong nhat -> backend goc cua paper
#   gemini-3.6-flash  : goi duoc tren free tier (da xac nhan 200 OK)
#   gemini-3.1-pro    : frontier DONG, CO free tier nhung la tran token/NGAY
#                       -> phai --repeats=1 va chay nhieu ngay, script se resume
#   gpt-oss-20b/120b  : Groq free tier, 8000 token/PHUT + tran ~200k/ngay
#   gpt-4o*           : tra phi (tai khoan hien tai khong con credit -> 429)
# Xem _sleep_between_calls cho gian cach tuong ung tung provider.
PARSER_BENCH_BACKEND = os.environ.get("PARSER_BENCH_BACKEND", "gemini-flash-lite")
for _a in sys.argv:
    if _a.startswith('--backend='):
        PARSER_BENCH_BACKEND = _a.split('=', 1)[1]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'agents'))
from llm_backends import get_llm as _factory_get_llm, BACKENDS as _BACKENDS, has_credentials

LLM_MODEL_NAME = _BACKENDS.get(PARSER_BENCH_BACKEND, {}).get('model', PARSER_BENCH_BACKEND)

# Backend goc cua so lieu RQ3 trong paper. CHI backend nay duoc ghi vao ten file
# khong hau to, de moi tham chieu san co trong bai va trong cac script khac giu
# nguyen hieu luc. MOI backend kiem chung cheo ghi ra file rieng.
#
# Day la sua mot lo hong THAT: truoc khi co doan nay, chay backend thu hai se
# GHI DE parser_ablation_benchmark.csv cua backend thu nhat. Du lieu RQ3 tung bi
# mat dung theo kieu do (600 dong Gemini that bi thay bang 200 dong tong hop).
PRIMARY_BACKEND = 'gemini-flash-lite'


def _out(name: str, ext: str) -> str:
    """Duong dan output co hau to backend (tru backend goc)."""
    d = DOCS_DIR if ext == '.tex' or ext == '.md' else OUTPUT_DIR
    if PARSER_BENCH_BACKEND == PRIMARY_BACKEND:
        return os.path.join(d, f'{name}{ext}')
    slug = re.sub(r'[^A-Za-z0-9]+', '-', PARSER_BENCH_BACKEND).strip('-')
    return os.path.join(d, f'{name}__{slug}{ext}')


def _sleep_between_calls():
    """Gian cach giua cac lan goi live de khong dam vao tran PHUT cua free tier.

    So lieu DO TRUC TIEP (2026-09-14), khong phai uoc luong:
      - Groq free tier: x-ratelimit-limit-tokens = 8000/PHUT. Chi phi thuc mot
        lan goi cua benchmark nay = 329 token (do bang hieu so
        x-ratelimit-remaining-tokens truoc/sau 5 lan goi). => ~24 call/phut la
        an toan => 2.5s. Prompt cua benchmark nay ngan hon nhieu so voi tuong.
      - Google free tier: ~15 req/phut cho ho flash; gemini-3.1-pro con bi tran
        token/NGAY nen gian cach rong hon cung khong cuu duoc, phai chia ngay.
    """
    cfg = _BACKENDS.get(PARSER_BENCH_BACKEND, {})
    prov, tier = cfg.get('provider'), cfg.get('tier')
    if prov == 'google':
        time.sleep(6.0 if tier == 'frontier' else 4.5)
    elif prov == 'groq':
        time.sleep(2.5)      # 8000 token/phut / 329 token moi call = 24/phut


def get_llm(use_live_api=True):
    """
    Khoi tao LLM backend cho benchmark.

    QUAN TRONG VE TINH KHOA HOC: Mac dinh (use_live_api=True) ham nay se GOI API
    Gemini THAT neu co GOOGLE_API_KEY hop le trong .env. Ket qua tra ve la tuple
    (llm_object, backend_name) — backend_name PHAI duoc ghi vao moi dong CSV va
    vao report, de khong bao gio nham lan giua ket qua LLM that va bo gia lap.

    Bo gia lap offline (OfflineAblationLLM) CHI duoc dung khi:
      - Nguoi dung truyen --offline (vd chay CI/smoke-test khong ton API quota), hoac
      - Khong tim thay GOOGLE_API_KEY, hoac goi API that bi loi khi khoi tao.
    Trong ca hai truong hop nay, backend_name se duoc gan la
    "SYNTHETIC_OFFLINE_EMULATOR_DO_NOT_CITE_AS_LLM_RESULT" de bat buoc downstream
    (report/paper) phai hien thi canh bao thay vi am tham coi la ket qua LLM that.
    """
    # --no-fallback: neu khong khoi tao duoc LLM that thi DUNG HAN, khong am tham
    # chuyen sang bo gia lap. Bat buoc dung cho moi lan chay co quota gioi han:
    # mot lan 429 luc probe se sinh ra 600 dong tong hop trong khi log chi co mot
    # dong WARNING o giua man hinh -- dung kieu mat du lieu da xay ra trong du an.
    no_fallback = '--no-fallback' in sys.argv

    if use_live_api:
        cfg = _BACKENDS.get(PARSER_BENCH_BACKEND)
        if cfg is None:
            msg = (f"Backend '{PARSER_BENCH_BACKEND}' khong co trong BACKENDS. "
                   f"Chon trong: {sorted(_BACKENDS)}")
            if no_fallback:
                raise SystemExit(f"[FATAL] {msg}")
            print(f"[WARNING] {msg}")
        elif not has_credentials(cfg['provider']):
            msg = f"Thieu API key cho provider '{cfg['provider']}'."
            if no_fallback:
                raise SystemExit(f"[FATAL] {msg}")
            print(f"[WARNING] {msg} Falling back to offline emulator.")
        else:
            try:
                llm = _factory_get_llm(PARSER_BENCH_BACKEND, temperature=0.2, max_retries=5)
                # Xac thuc key that su goi duoc API truoc khi cong bo la "live"
                from langchain_core.messages import HumanMessage
                llm.invoke([HumanMessage(content="Reply with exactly: OK")])
                print(f"  [LLM] Using LIVE {cfg['provider']} {LLM_MODEL_NAME} "
                      f"(tier={cfg['tier']}, pinned={cfg['pinned']}).")
                return llm, f"LIVE_{LLM_MODEL_NAME}"
            except Exception as e:
                if no_fallback:
                    raise SystemExit(
                        f"[FATAL] Live call that bai tren '{PARSER_BENCH_BACKEND}': "
                        f"{str(e)[:300]}\n"
                        f"        Dang chay voi --no-fallback nen DUNG, khong sinh du "
                        f"lieu tong hop. Neu la 429 tran ngay thi doi quota hoi roi chay lai.")
                print(f"[WARNING] Live call failed on {PARSER_BENCH_BACKEND}: "
                      f"{str(e)[:160]}. Falling back to offline emulator.")

    # Deterministic offline stand-in: KHONG dai dien cho hanh vi LLM that.
    # Chi dung cho smoke-test CI khi khong co API key. Khong duoc trich dan
    # ket qua tu backend nay nhu bang chung ve hanh vi LLM trong bao cao khoa hoc.
    print("  [LLM] Using OFFLINE deterministic stub (NOT a real LLM — CI smoke-test only).")
    class OfflineAblationLLM:
        def invoke(self, messages):
            text = messages[0].content if messages else ""
            class Resp:
                pass
            resp = Resp()
            
            # If adversarial prompt with 1000% or huge numbers
            if "1000%" in text or "500%" in text:
                resp.content = json.dumps({
                    "injection_service": "payment" if "payment" in text else "front-end",
                    "injection_delta_pct": 500.0,
                    "adjustment": 475.0,
                    "core_services": ["front-end", "orders", "auth-crypto-blockchain"],
                    "confidence": "HIGH"
                })
            elif "-80%" in text or "ngắt" in text:
                resp.content = json.dumps({
                    "injection_service": "orders",
                    "injection_delta_pct": -80.0,
                    "adjustment": -105.0,
                    "core_services": ["orders", "database-master"],
                    "confidence": "HIGH"
                })
            elif "microservice ma" in text or "auth-crypto" in text:
                resp.content = json.dumps({
                    "injection_service": "auth-crypto-blockchain",
                    "injection_delta_pct": 35.0,
                    "adjustment": 10.0,
                    "core_services": ["front-end", "auth-crypto-blockchain", "inventory-warehouse-v3"],
                    "confidence": "MEDIUM"
                })
            elif "Bỏ qua cổng front-end" in text or "tiêm trực tiếp 50% tải vào dịch vụ payment" in text:
                resp.content = json.dumps({
                    "injection_service": "payment",
                    "injection_delta_pct": 50.0,
                    "adjustment": 25.0,
                    "core_services": ["payment"],
                    "confidence": "HIGH"
                })
            elif "0.00001%" in text:
                resp.content = json.dumps({
                    "injection_service": "front-end",
                    "injection_delta_pct": 0.00001,
                    "adjustment": -9.999,
                    "core_services": ["front-end"],
                    "confidence": "LOW"
                })
            elif "Dark Mode" in text or "tối" in text or "ngôn ngữ" in text:
                resp.content = json.dumps({
                    "injection_service": "front-end",
                    "injection_delta_pct": 5.0,
                    "adjustment": -5.0,
                    "core_services": ["front-end", "ui-theme-service"],
                    "confidence": "HIGH"
                })
            elif "voucher" in text.lower() or "promo" in text.lower() or "giảm giá" in text.lower():
                resp.content = json.dumps({
                    "injection_service": "front-end",
                    "injection_delta_pct": 22.0,
                    "adjustment": 2.0,
                    "core_services": ["front-end", "carts", "payment"],
                    "confidence": "HIGH"
                })
            elif "recommend" in text.lower() or "gợi ý" in text.lower():
                resp.content = json.dumps({
                    "injection_service": "front-end",
                    "injection_delta_pct": 32.0,
                    "adjustment": 2.0,
                    "core_services": ["front-end", "catalogue", "user"],
                    "confidence": "HIGH"
                })
            elif "theo dõi" in text.lower() or "track" in text.lower() or "shipping" in text.lower():
                resp.content = json.dumps({
                    "injection_service": "shipping" if "bỏ qua" in text else "front-end",
                    "injection_delta_pct": 16.0,
                    "adjustment": 1.0,
                    "core_services": ["front-end", "shipping"],
                    "confidence": "HIGH"
                })
            else:
                resp.content = json.dumps({
                    "injection_service": "front-end",
                    "injection_delta_pct": 25.0,
                    "adjustment": 0.0,
                    "core_services": ["front-end", "orders"],
                    "confidence": "MEDIUM"
                })
            return resp

    return OfflineAblationLLM(), "SYNTHETIC_OFFLINE_EMULATOR_DO_NOT_CITE_AS_LLM_RESULT"


# ============================================================
# 2. BASELINE IMPLEMENTATIONS
# ============================================================

class UnguardedZeroShotParser:
    """Baseline 1: Pure Zero-Shot LLM without calibration anchors or guards."""
    def __init__(self, llm):
        self.llm = llm
        
    def parse(self, requirement: str):
        prompt = f"""You are a cloud architect analyzing microservice workload.
Requirement: "{requirement}"

Predict the workload injection requirements:
1. "injection_service": which service receives the injection
2. "injection_delta_pct": estimated percentage increase in traffic (e.g. 20.0)
3. "core_services": list of affected microservice names

Return ONLY valid JSON:
{{
  "injection_service": "...",
  "injection_delta_pct": <float>,
  "core_services": ["..."]
}}"""
        t0 = time.time()
        try:
            from langchain_core.messages import HumanMessage
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            data = json.loads(resp.content.strip().replace("```json", "").replace("```", "").strip())
            inj_svc = str(data.get("injection_service", "front-end"))
            delta = float(data.get("injection_delta_pct", 20.0))
            core = list(data.get("core_services", ["front-end"]))
        except Exception as e:
            inj_svc = "front-end"
            delta = 20.0
            core = ["front-end"]
        elapsed = (time.time() - t0) * 1000
        return inj_svc, delta, core, elapsed, True


class UnguardedFewShotParser:
    """Baseline 2: Few-Shot LLM with system context, but NO physical runtime guards."""
    def __init__(self, llm):
        self.llm = llm
        
    def parse(self, requirement: str):
        prompt = f"""You are a Systems Analyst for SockShop microservices (front-end, catalogue, user, carts, orders, payment, shipping).
Calibration references:
- Promo code: ~20%
- Recommendation: ~30%
- Tracking: ~15%
- Catalogue browse: ~10%
- Place order: ~25%

Requirement: "{requirement}"

Return ONLY JSON:
{{
  "injection_service": "...",
  "injection_delta_pct": <float>,
  "core_services": ["..."]
}}"""
        t0 = time.time()
        try:
            from langchain_core.messages import HumanMessage
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            data = json.loads(resp.content.strip().replace("```json", "").replace("```", "").strip())
            inj_svc = str(data.get("injection_service", "front-end"))
            delta = float(data.get("injection_delta_pct", 20.0))
            core = list(data.get("core_services", ["front-end"]))
        except Exception:
            inj_svc = "front-end"
            delta = 20.0
            core = ["front-end"]
        elapsed = (time.time() - t0) * 1000
        return inj_svc, delta, core, elapsed, True


class RuleOnlyParser:
    """Baseline 3: Pure keyword heuristic without LLM."""
    def parse(self, requirement: str):
        t0 = time.time()
        rt = classify_request(requirement)
        info = CALL_CHAINS.get(rt, {})
        delta = float(info.get('expected_delta_pct', 20.0))
        core = list(info.get('services', ['front-end']))
        inj_svc = 'front-end'
        elapsed = (time.time() - t0) * 1000
        return inj_svc, delta, core, elapsed, False


class GuardedHybridRunner:
    """Proposed: Two-Tier Grounded Parser with 5 Runtime Guards."""
    def __init__(self, parser_agent):
        self.parser = parser_agent
        
    def parse(self, requirement: str):
        t0 = time.time()
        res = self.parser.parse(requirement)
        elapsed = (time.time() - t0) * 1000
        return res.injection_service, res.injection_delta_pct, res.core_services, elapsed, res.llm_was_called


# ============================================================
# 3. BENCHMARK EXECUTION ENGINE
# ============================================================
def run_parser_benchmark(use_live_api=True, models_to_run=None):
    """
    Chay ablation benchmark RQ3.

    Mac dinh use_live_api=True: se goi API Gemini THAT (can GOOGLE_API_KEY trong .env).
    Truyen --offline (hoac dat use_live_api=False) de ep dung bo gia lap deterministic
    (chi nen dung cho smoke-test CI, KHONG dung de cong bo ket qua khoa hoc).

    So lan lap (N_REPEATS) mac dinh la 3 vi LLM that co temperature=0.2 => khong
    deterministic. Ket qua duoc tong hop dang mean +/- std qua cac lan lap thay vi
    1 con so diem duy nhat. Dieu chinh qua bien moi truong PARSER_BENCH_REPEATS
    hoac co --repeats=N tren dong lenh.

    models_to_run: neu chi dinh (danh sach ten model con trong 4 config), CHI goi
    live LLM/re-run cho cac model do; cac model KHONG duoc chon se lay lai dung
    nguyen dong da co trong parser_ablation_benchmark.csv (khong re-run, khong
    tao du lieu gia). Dung khi mot ban sua code CHI anh huong 1 config cu the
    (vd sua ParserAgent chi anh huong Guarded_Hybrid_Parser, khong anh huong
    Rule_Only/Unguarded_* vi cac config nay dung code hoan toan khac/khong dung
    ParserAgent) -- tranh ton quota re-run lai nhung gi khong the doi, va tranh
    lam le tuong dong bo gia (b0-b2 cu + guarded moi) khi that ra day la dung
    dan boi vi code cua b0-b2 khong doi.
    """
    if '--live-llm' in sys.argv:
        use_live_api = True
    if '--offline' in sys.argv:
        use_live_api = False

    n_repeats = int(os.environ.get('PARSER_BENCH_REPEATS', 3))
    for arg in sys.argv:
        if arg.startswith('--repeats='):
            n_repeats = int(arg.split('=', 1)[1])

    print("=" * 75)
    print("      RUNNING RQ3 COMPREHENSIVE LLM PARSER BENCHMARK (50 PROMPTS)")
    print(f"      Repeats per configuration: {n_repeats} | Live API requested: {use_live_api}")
    print("=" * 75)

    # Load Prompts
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} requirement test cases across 4 categories.")

    # Initialize agents (mot LLM backend dung chung cho tat ca cac cau hinh,
    # de moi ablation duoc so sanh cong bang tren cung 1 model & backend).
    arch = ArchitectureAgent(GRAPH_PATH)
    llm, llm_backend = get_llm(use_live_api=use_live_api)
    is_synthetic = llm_backend.startswith("SYNTHETIC")
    if is_synthetic:
        print("\n" + "!" * 75)
        print("  CANH BAO: Dang chay voi BO GIA LAP OFFLINE, khong phai LLM that.")
        print("  Ket qua nay CHI de kiem tra code (CI smoke-test), KHONG duoc")
        print("  dua vao bao cao/bai bao khoa hoc nhu bang chung ve hanh vi LLM.")
        print("!" * 75 + "\n")

    all_model_names = ['Unguarded_ZeroShot_LLM', 'Unguarded_FewShot_LLM', 'Rule_Only', 'Guarded_Hybrid_Parser']
    run_selected = models_to_run if models_to_run else all_model_names
    csv_path = _out('parser_ablation_benchmark', '.csv')

    reused_results = []
    if models_to_run and os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        skipped = [m for m in all_model_names if m not in run_selected]
        reused_results = old_df[old_df['model'].isin(skipped)].to_dict('records')
        print(f"[REUSE] Keeping {len(reused_results)} existing rows for unaffected "
              f"configuration(s) {skipped} (not re-run — code path unchanged for these).")

    all_results = []
    run_timestamp = pd.Timestamp.now().isoformat()

    # Write incrementally (row-by-row, flushed) so a mid-run quota exhaustion
    # never loses already-completed rows -- lesson learned the hard way earlier
    # in this project's history.
    fieldnames = ['repeat_id', 'llm_backend', 'run_timestamp', 'model', 'prompt_id',
                  'category', 'requirement', 'expected_anchor', 'pred_delta', 'abs_error',
                  'injection_service', 'core_services', 'physical_boundary_violation',
                  'service_hallucination', 'gateway_misdirection', 'hallucinated_items',
                  'latency_ms', 'llm_called']
    incremental_path = _out('parser_ablation_benchmark_INPROGRESS', '.csv')

    # RESUME: backend co tran token/NGAY (gemini-3.1-pro) khong the chay het
    # 450 lan goi trong mot ngay. Doc lai cac o (repeat_id, model, prompt_id) da
    # hoan thanh va BO QUA chung, roi ghi TIEP (mode 'a') thay vi ghi de.
    done = set()
    resume = os.path.exists(incremental_path) and '--fresh' not in sys.argv
    if resume:
        with open(incremental_path, newline='', encoding='utf-8') as fr:
            for r in csv.DictReader(fr):
                done.add((int(r['repeat_id']), r['model'], r['prompt_id']))
        print(f"[RESUME] Tim thay {len(done)} o da hoan thanh trong "
              f"{os.path.basename(incremental_path)} -- se bo qua va chay tiep.")
        print(f"         (dung --fresh de bo va chay lai tu dau)")

    f_incremental = open(incremental_path, 'a' if resume else 'w',
                         newline='', encoding='utf-8')
    incremental_writer = csv.DictWriter(f_incremental, fieldnames=fieldnames)
    if not resume:
        incremental_writer.writeheader()
    # Cac dong da co tu lan chay truoc phai duoc gop vao ket qua cuoi cung
    if done:
        all_results.extend(pd.read_csv(incremental_path).to_dict('records'))

    print(f"\nEvaluating across {len(prompts)} test scenarios x {n_repeats} repeats "
          f"(backend={llm_backend}) for configuration(s): {run_selected}...")
    for repeat_id in range(1, n_repeats + 1):
        # Tao lai cac model moi lan lap de tranh state ro ri giua cac repeat
        # (vi du ParserAgent khong giu state giua cac call nen an toan tao lai).
        guarded_parser = ParserAgent(llm=llm, arch_agent=arch)
        all_models = {
            'Unguarded_ZeroShot_LLM': UnguardedZeroShotParser(llm),
            'Unguarded_FewShot_LLM':  UnguardedFewShotParser(llm),
            'Rule_Only':              RuleOnlyParser(),
            'Guarded_Hybrid_Parser':  GuardedHybridRunner(guarded_parser)
        }
        models = {k: v for k, v in all_models.items() if k in run_selected}

        for model_name, runner in models.items():
            print(f"  [repeat {repeat_id}/{n_repeats}] Testing configuration: [{model_name}]...")
            for p in prompts:
                p_id = p['id']
                cat = p['category']
                req = p['requirement']
                exp_anchor = p['expected_anchor']
                exp_gw = p['expected_gateway']

                if (repeat_id, model_name, str(p_id)) in done:
                    continue

                # Execute. Mot 429 tran-ngay o day phai LAM DUNG lan chay, khong
                # duoc bat va di tiep: di tiep se sinh ra 450 dong loi gan nhau
                # ma van trong nhu du lieu. File INPROGRESS giu lai phan da xong,
                # lan chay sau se resume.
                try:
                    inj_svc, delta, core_svcs, latency_ms, llm_called = runner.parse(req)
                except Exception as e:
                    f_incremental.flush()
                    n_done = len(done) + len([r for r in all_results
                                              if r.get('repeat_id') == repeat_id])
                    raise SystemExit(
                        f"\n[STOP] Loi khi goi backend tai repeat={repeat_id} "
                        f"model={model_name} prompt={p_id}:\n  {str(e)[:300]}\n"
                        f"  Da ghi an toan {n_done} o vao {incremental_path}\n"
                        f"  Neu la 429 tran ngay: doi quota hoi roi chay LAI DUNG "
                        f"lenh nay -- script se tu resume.")

                # Rate-limit: Free Tier cua gemini-*-flash-lite gioi han 15 request/PHUT
                # (GenerateRequestsPerMinutePerProjectPerModel-FreeTier). Cho ~4.5s giua
                # cac lan goi that de tranh 429 lien tuc (nhanh hon se bi throttle nang).
                if (not is_synthetic) and llm_called:
                    _sleep_between_calls()

                # Metric 1: Physical Boundary Violation (Delta not in [5%, 50%])
                pbv = (delta < 5.0) or (delta > 50.0)

                # Metric 2: Service Hallucination (Any service not in KNOWN_SERVICES)
                hallucinated_svcs = [s for s in core_svcs if s not in KNOWN_SERVICES]
                sh = len(hallucinated_svcs) > 0

                # Metric 3: Gateway Misdirection (Injection node != front-end)
                gm = (inj_svc != exp_gw)

                # Metric 4: Anchor Absolute Error
                ae = abs(delta - exp_anchor)

                row = {
                    'repeat_id': repeat_id,
                    'llm_backend': llm_backend,
                    'run_timestamp': run_timestamp,
                    'model': model_name,
                    'prompt_id': p_id,
                    'category': cat,
                    'requirement': req,
                    'expected_anchor': exp_anchor,
                    'pred_delta': delta,
                    'abs_error': ae,
                    'injection_service': inj_svc,
                    'core_services': str(core_svcs),
                    'physical_boundary_violation': pbv,
                    'service_hallucination': sh,
                    'gateway_misdirection': gm,
                    'hallucinated_items': str(hallucinated_svcs),
                    'latency_ms': latency_ms,
                    'llm_called': llm_called
                }
                all_results.append(row)
                incremental_writer.writerow(row)
                f_incremental.flush()

    f_incremental.close()
    os.remove(incremental_path)  # merge succeeded -- no longer needed as a recovery copy

    df = pd.DataFrame(all_results + reused_results)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Raw ablation results saved to: {csv_path} "
          f"({len(all_results)} freshly run + {len(reused_results)} reused = {len(df)} total)")

    # ============================================================
    # 4. STATISTICAL SYNTHESIS ACROSS MODELS (mean +/- std qua N_REPEATS)
    # ============================================================
    def _per_repeat_rate(sub_df, col):
        """Tra ve list ti le (%) cua `col` tinh rieng cho tung repeat_id."""
        return [
            sub_df[sub_df['repeat_id'] == r][col].mean() * 100.0
            for r in sorted(sub_df['repeat_id'].unique())
        ]

    def _per_repeat_mean(sub_df, col):
        return [
            sub_df[sub_df['repeat_id'] == r][col].mean()
            for r in sorted(sub_df['repeat_id'].unique())
        ]

    model_names = ['Unguarded_ZeroShot_LLM', 'Unguarded_FewShot_LLM', 'Rule_Only', 'Guarded_Hybrid_Parser']
    summary_rows = []
    category_rows = []
    for model_name in model_names:
        m_df = df[df['model'] == model_name]

        pbvr_vals = _per_repeat_rate(m_df, 'physical_boundary_violation')
        shr_vals  = _per_repeat_rate(m_df, 'service_hallucination')
        gmr_vals  = _per_repeat_rate(m_df, 'gateway_misdirection')
        mae_vals  = _per_repeat_mean(m_df, 'abs_error')
        lat_vals  = _per_repeat_mean(m_df, 'latency_ms')
        bypass_vals = [100.0 - v for v in _per_repeat_rate(m_df, 'llm_called')]

        adv_df = m_df[m_df['category'] == 'Adversarial_Stress']
        adv_pbvr_vals = _per_repeat_rate(adv_df, 'physical_boundary_violation')
        adv_shr_vals  = _per_repeat_rate(adv_df, 'service_hallucination')

        def fmt(vals, suffix=''):
            arr = np.array(vals, dtype=float)
            if len(arr) <= 1 or np.isnan(arr).all():
                return f"{np.nanmean(arr):.2f}{suffix}"
            return f"{arr.mean():.2f}±{arr.std(ddof=1):.2f}{suffix}"

        summary_rows.append({
            'Model Configuration': model_name,
            'LLM Backend': m_df['llm_backend'].iloc[0] if len(m_df) else llm_backend,
            'N Repeats': m_df['repeat_id'].nunique(),
            'PBVR (%)': fmt(pbvr_vals, '%'),
            'SHR (%)': fmt(shr_vals, '%'),
            'GMR (%)': fmt(gmr_vals, '%'),
            'Anchor MAE (%)': fmt(mae_vals, '%'),
            'Adversarial PBVR (%)': fmt(adv_pbvr_vals, '%'),
            'Adversarial SHR (%)': fmt(adv_shr_vals, '%'),
            'Mean Latency (ms)': fmt(lat_vals, ' ms'),
            'Fast-Path Bypass (%)': fmt(bypass_vals, '%'),
        })

        # Full per-category breakdown (khong chi rieng Adversarial)
        for cat in sorted(m_df['category'].unique()):
            c_df = m_df[m_df['category'] == cat]
            category_rows.append({
                'model': model_name,
                'category': cat,
                'n_prompts': c_df['prompt_id'].nunique(),
                'pbvr_pct': round(np.mean(_per_repeat_rate(c_df, 'physical_boundary_violation')), 2),
                'shr_pct': round(np.mean(_per_repeat_rate(c_df, 'service_hallucination')), 2),
                'gmr_pct': round(np.mean(_per_repeat_rate(c_df, 'gateway_misdirection')), 2),
                'mae_pct': round(np.mean(_per_repeat_mean(c_df, 'abs_error')), 2),
            })


    sum_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("          RQ3 BENCHMARK & ABLATION SUMMARY TABLE (ACADEMIC RESULTS)")
    print("=" * 80)
    print(sum_df.to_string(index=False))
    print("=" * 80)
    
    cat_df = pd.DataFrame(category_rows)
    cat_csv_path = _out('parser_ablation_by_category', '.csv')
    cat_df.to_csv(cat_csv_path, index=False, encoding='utf-8')
    print(f"[OK] Per-category breakdown saved to: {cat_csv_path}")

    # ============================================================
    # 5. GENERATE LATEX TABLE SNIPPET (docs/table_rq3_parser_ablation.tex)
    # ============================================================
    backend_note = llm_backend.replace('_', r'\_')
    latex_code = r"""\begin{table*}[t]
\centering
\caption{Ablation Study and Reliability Evaluation of LLM Parser Configurations across 50 Requirements (RQ3).
Rates are mean""" + (r"$\pm$std" if n_repeats > 1 else "") + f""" over {n_repeats} independent repeat(s). Backend: {backend_note}.""" + r"""}
\label{tab:parser_ablation}
\begin{tabular}{lcccccc}
\toprule
\textbf{Parser Configuration} & \textbf{PBVR (\%)} $\downarrow$ & \textbf{SHR (\%)} $\downarrow$ & \textbf{GMR (\%)} $\downarrow$ & \textbf{MAE (\%)} $\downarrow$ & \textbf{Adv. Fail (\%)} $\downarrow$ & \textbf{Latency (ms)} $\downarrow$ \\
\midrule
"""
    for r in summary_rows:
        cfg = r['Model Configuration'].replace('_', r'\_')
        cell_vals = [r['PBVR (%)'], r['SHR (%)'], r['GMR (%)'], r['Anchor MAE (%)'],
                     r['Adversarial PBVR (%)'], r['Mean Latency (ms)']]
        # Escape raw '%' for LaTeX (fmt() above produces plain '%'/' ms' suffixes
        # meant for console/markdown display, not LaTeX — an un-escaped '%'
        # starts a comment and truncates the rest of the table row). Likewise
        # fmt() emits a literal Unicode '±' (fine in a terminal or Markdown),
        # which is not valid LaTeX math -- convert to '$\pm$' for this table.
        cell_vals = [v.replace('%', r'\%').replace('±', r'$\pm$') for v in cell_vals]
        is_ours = 'Guarded' in cfg
        name_cell = ('\\textbf{' + cfg + ' (Ours)}') if is_ours else cfg
        data_cells = [('\\textbf{' + v + '}') if is_ours else v for v in cell_vals]
        latex_code += name_cell + ' & ' + ' & '.join(data_cells) + ' \\\\\n'

    if is_synthetic:
        latex_code += r"\multicolumn{7}{l}{\footnotesize \textbf{WARNING: synthetic offline emulator, not a live LLM. Do not cite as an LLM behavior result.}} \\" + "\n"

    latex_code += r"""\bottomrule
\multicolumn{7}{l}{\footnotesize \textit{Notes:} PBVR = Physical Boundary Violation Rate ($\Delta \notin [5\%, 50\%]$); SHR = Service Hallucination Rate;} \\
\multicolumn{7}{l}{\footnotesize GMR = Gateway Misdirection Rate; MAE = Mean Absolute Error vs. Calibration Anchor; Adv. Fail = Adversarial-subset Violation Rate.}
\end{tabular}
\end{table*}
"""
    latex_path = _out('table_rq3_parser_ablation', '.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    print(f"[OK] LaTeX publication table generated: {latex_path}")

    # ============================================================
    # 6. GENERATE SCIENTIFIC MARKDOWN REPORT (sinh dong tu du lieu do duoc,
    #    KHONG viet san ket luan — tranh lech pha giua van ban va CSV)
    # ============================================================
    backend_warning = (
        "\n> ⚠️ **CẢNH BÁO**: Lần chạy này dùng **bộ giả lập offline** "
        f"(`{llm_backend}`), KHÔNG phải LLM thật. Các số liệu dưới đây "
        "chỉ có giá trị kiểm tra code (smoke-test), **không được trích dẫn "
        "làm bằng chứng khoa học** về hành vi LLM. Chạy lại với `--live-llm` "
        "và `GOOGLE_API_KEY` hợp lệ để có kết quả có thể công bố.\n"
        if is_synthetic else
        f"\n> ✅ Backend: **LLM thật** (`{llm_backend}`), {n_repeats} lần lặp độc lập "
        "(nhiệt độ 0.2, không deterministic) để đo phương sai run-to-run.\n"
    )

    cat_table_md = "| Model | Category | N | PBVR (%) | SHR (%) | GMR (%) | MAE (%) |\n|---|---|---|---|---|---|---|\n"
    for _, r in cat_df.iterrows():
        cat_table_md += (
            f"| `{r['model']}` | {r['category']} | {r['n_prompts']} | "
            f"{r['pbvr_pct']} | {r['shr_pct']} | {r['gmr_pct']} | {r['mae_pct']} |\n"
        )

    md_report = f"""# 📊 BÁO CÁO BENCHMARK VÀ THỰC NGHIỆM ABLATION CHO LLM PARSER (RQ3)
## Đề Tài: *Grounded Large Language Models for Structural Causal Intervention in Microservices*

Tài liệu này được **sinh tự động** từ `parser_ablation_benchmark.csv` (không viết tay số liệu)
để giải quyết **Research Question 3 (RQ3)**:
> **RQ3:** *"Mô hình Grounded LLM Parser chuyển dịch yêu cầu ngôn ngữ tự nhiên thành đại số can thiệp $do(x)$ với độ tin cậy ra sao, và các Runtime Guards triệt tiêu hiện tượng ảo giác (Hallucination) và vi phạm biên vật lý như thế nào?"*
{backend_warning}
Thời điểm chạy: `{run_timestamp}` | Số lần lặp: `{n_repeats}` | Backend: `{llm_backend}`

---

## 🏛️ 1. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

* **Quy mô tập dữ liệu**: **50 kịch bản yêu cầu tính năng phần mềm e-commerce** (`data/benchmark/parser_benchmark_prompts.json`), phân bổ vào 4 nhóm: In-Distribution (20), Complex Multi-Hop (10), Subtle Read-Only (10), Adversarial/Stress-Test (10).
* **4 Cấu hình đối chiếu**: `Unguarded_ZeroShot_LLM`, `Unguarded_FewShot_LLM` (LLM thật, không guard), `Rule_Only` (heuristic từ khóa, không LLM), `Guarded_Hybrid_Parser` (đề xuất — fast-path bảng hiệu chỉnh khi similarity ≥ 0.6, LLM có guard khi không match rõ).

### ⚠️ Giới hạn phương pháp luận cần lưu ý khi đọc bảng dưới đây
1. **`expected_anchor`** (dùng để tính MAE) là **cùng bộ giá trị hiệu chỉnh cứng** (`CALL_CHAINS` trong `request_router.py`) mà `Rule_Only` và nhánh fast-path của `Guarded_Hybrid_Parser` tra cứu trực tiếp. Vì vậy MAE thấp của 2 cấu hình này ở nhóm **In-Distribution** phần lớn phản ánh việc tra bảng đúng, **không phải** năng lực suy luận ngữ nghĩa độc lập — MAE có ý nghĩa kiểm chứng thật sự nằm ở nhóm **Complex_MultiHop / Subtle_ReadOnly / Adversarial_Stress**, nơi cấu hình phải suy luận vượt ra ngoài match từ khóa trực tiếp.
2. Các cấu hình `Unguarded_*` gọi LLM thật không có bảng hiệu chỉnh trong ngữ cảnh (ZeroShot) hoặc có (FewShot) — đây là phép so sánh công bằng về khả năng tự kiềm chế của LLM khi không có runtime guard, không phải baseline bị lập trình để thất bại.
3. LLM thật có tính ngẫu nhiên (temperature=0.2) — bảng dưới báo cáo **mean ± std qua {n_repeats} lần lặp độc lập**, không phải một lần chạy duy nhất.

---

## 📊 2. BẢNG TỔNG HỢP KẾT QUẢ ĐỐI CHIẾU (RQ3 MASTER TABLE)

| Cấu Hình Mô Hình | PBVR (%) | SHR (%) | GMR (%) | Anchor MAE (%) | Adv. PBVR (%) | Latency (ms) | Fast-Path Bypass (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in summary_rows:
        is_ours = "🏆 **(Đề xuất)**" if "Guarded" in r['Model Configuration'] else ""
        md_report += (
            f"| `{r['Model Configuration']}` {is_ours} | "
            f"{r['PBVR (%)']} | {r['SHR (%)']} | {r['GMR (%)']} | "
            f"{r['Anchor MAE (%)']} | {r['Adversarial PBVR (%)']} | "
            f"{r['Mean Latency (ms)']} | {r['Fast-Path Bypass (%)']} |\n"
        )

    md_report += f"""
*(Không suy diễn "0.0%" hay "vượt trội" nếu bảng trên không thực sự cho ra số đó — mọi tuyên bố kết luận phải đọc trực tiếp từ bảng số ở trên sau khi chạy.)*

---

## 📋 3. CHI TIẾT THEO TỪNG NHÓM PROMPT (BREAKDOWN BY CATEGORY)
Đây là bảng quan trọng nhất để đánh giá khả năng khái quát hoá thật sự (ngoài phạm vi tra bảng In-Distribution):

{cat_table_md}
---

## 📄 4. BẢNG MÃ NGUỒN LATEX CHO BÀI BÁO
Đã được xuất tự động tại: [`docs/table_rq3_parser_ablation.tex`](../docs/table_rq3_parser_ablation.tex).

## 📁 5. DỮ LIỆU THÔ
* Toàn bộ {len(df)} bản ghi (50 prompts × 4 cấu hình × {n_repeats} lần lặp): `data/processed/scm_results/parser_ablation_benchmark.csv`
* Breakdown theo category: `data/processed/scm_results/parser_ablation_by_category.csv`
"""
    md_path = _out('RQ3_LLM_PARSER_BENCHMARK_REPORT', '.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"[OK] Scientific markdown report generated: {md_path}")
    print("=" * 75)
    return df, sum_df


if __name__ == '__main__':
    only = None
    for arg in sys.argv:
        if arg.startswith('--only='):
            only = arg.split('=', 1)[1].split(',')
    run_parser_benchmark(models_to_run=only)
