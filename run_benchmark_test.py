"""
run_benchmark_test.py
=====================
Kiem thu hieu nang truoc/sau khi chinh sua:

  Test 1: Parser delta calibration — MAE vs CALL_CHAINS[expected_delta_pct]
  Test 2: Guard integrity — delta luon trong [5%, 50%] sau clamp
  Test 3: Bivariate accuracy — so sanh MAPE/F1 voi baseline Q1 report
  Test 4: Multi-hop cascade — payment nhan it hon front-end (cascade suy giam)
  Test 5: Parser efficiency — phuong an A, rule-based truoc
"""

import os, sys, time, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

import numpy as np
import pandas as pd

GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')
DATA_DIR   = os.path.join(PROJECT_ROOT, 'data', 'raw')

# Baseline Q1 report
Q1_BASELINE = {
    ('front-end', 'CPU'):    1.3,
    ('front-end', 'Memory'): 3.1,
    ('user',      'CPU'):    1.4,
    ('user',      'Memory'): 1.0,
    ('carts',     'CPU'):    4.7,
    ('carts',     'Memory'): 0.4,
    ('orders',    'CPU'):    12.9,
    ('orders',    'Memory'): 0.8,
    ('payment',   'CPU'):    17.0,
    ('payment',   'Memory'): 2.6,
    ('catalogue', 'CPU'):    15.5,
    ('catalogue', 'Memory'): 1.9,
    ('shipping',  'CPU'):    8.6,
    ('shipping',  'Memory'): 0.1,
}
TOLERANCE_PCT = 5.0
SEP = "=" * 70

def print_sep(title=""):
    if title:
        pad = max(0, (70 - len(title) - 4) // 2)
        print(f"\n{SEP}\n{' '*pad}  {title}\n{SEP}")
    else:
        print("-" * 70)


# ===========================================================
def init_agents():
    print_sep("KHOI TAO AGENTS")
    from architecture_agent import ArchitectureAgent
    from performance_agent  import PerformanceAgent
    from simulation_agent   import SimulationAgent

    arch = ArchitectureAgent(GRAPH_PATH)
    print("  [OK] ArchitectureAgent")

    t0 = time.time()
    perf = PerformanceAgent(data_dir=DATA_DIR, auto_train=True)
    print(f"  [OK] PerformanceAgent ({time.time()-t0:.1f}s)")

    t0 = time.time()
    sim = SimulationAgent(data_dir=DATA_DIR, graph_path=GRAPH_PATH)
    sim.train()
    print(f"  [OK] SimulationAgent  ({time.time()-t0:.1f}s)")

    return arch, perf, sim


# ===========================================================
def test1_parser_calibration(arch):
    print_sep("TEST 1: Parser Delta Calibration")
    print("  Muc tieu: MAE(delta_pred, expected_delta) <= 5%")
    print("  Dung DummyLLM: chi test rule-based path + guard logic\n")

    from request_router import CALL_CHAINS
    from parser_agent import ParserAgent

    class DummyLLM:
        def invoke(self, messages):
            class R:
                content = '{"core_services": ["front-end"], "adjustment": 0.0, "reasoning": "dummy", "confidence": "MEDIUM"}'
            return R()

    parser = ParserAgent(llm=DummyLLM(), arch_agent=arch)

    rows = []
    for rt, info in CALL_CHAINS.items():
        desc     = info['description']
        expected = info['expected_delta_pct']
        parsed   = parser.parse(desc)
        error    = abs(parsed.injection_delta_pct - expected)
        rows.append({'rt': rt, 'expected': expected,
                     'predicted': parsed.injection_delta_pct,
                     'error': error, 'sim': parsed.similarity_score})
        status = "OK  " if error <= 5.0 else "FAIL"
        print(f"  [{status}] {rt:<25} | exp={expected:>3}% | "
              f"pred={parsed.injection_delta_pct:>5.1f}% | "
              f"err={error:>4.1f}% | sim={parsed.similarity_score:.2f}")

    df   = pd.DataFrame(rows)
    mae  = df['error'].mean()
    n_ok = (df['error'] <= 5.0).sum()
    print_sep()
    print(f"  MAE = {mae:.2f}% | Pass {n_ok}/{len(df)}")
    verdict = "PASS" if mae <= 5.0 else "FAIL"
    print(f"  KET LUAN TEST 1: {verdict}")
    return verdict, mae


# ===========================================================
def test2_guard_integrity(arch):
    print_sep("TEST 2: Guard Integrity (fuzz test)")
    print("  Muc tieu: delta luon [5%,50%], adjustment <= 10%, core_services hop le\n")

    from parser_agent import ParserAgent, MIN_DELTA_PCT, MAX_DELTA_PCT, MAX_ADJUSTMENT_PCT

    adversarial = [
        ("Adjustment +100%",  '{"core_services":["front-end"],"adjustment":100.0,"reasoning":"x","confidence":"HIGH"}'),
        ("Adjustment -80%",   '{"core_services":["payment"],"adjustment":-80.0,"reasoning":"x","confidence":"HIGH"}'),
        ("Service khong ton tai", '{"core_services":["fake","unknown"],"adjustment":5.0,"reasoning":"x","confidence":"MEDIUM"}'),
        ("JSON sai format",   'KHONG PHAI JSON GI CA'),
        ("Confidence sai",    '{"core_services":["carts"],"adjustment":3.0,"reasoning":"ok","confidence":"SUPER_HIGH"}'),
    ]

    all_pass = True
    for desc, resp in adversarial:
        class MockLLM:
            def __init__(self, r):
                self._r = r
            def invoke(self, _):
                class R: pass
                r = R(); r.content = self._r; return r

        parser = ParserAgent(llm=MockLLM(resp), arch_agent=arch)
        parsed = parser.parse("place order buy product checkout")

        ok1 = MIN_DELTA_PCT <= parsed.injection_delta_pct <= MAX_DELTA_PCT
        ok2 = abs(parsed.adjustment) <= MAX_ADJUSTMENT_PCT
        ok3 = all(s in {'front-end','catalogue','user','carts','orders','payment','shipping'}
                  for s in parsed.core_services)
        ok4 = parsed.confidence in ('HIGH','MEDIUM','LOW')
        ok  = ok1 and ok2 and ok3 and ok4
        if not ok: all_pass = False

        print(f"  [{'OK  ' if ok else 'FAIL'}] {desc:<30} | "
              f"delta={parsed.injection_delta_pct:.1f}% "
              f"adj={parsed.adjustment:.1f}% "
              f"core={parsed.core_services} "
              f"conf={parsed.confidence}")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n  KET LUAN TEST 2: {verdict}")
    return verdict


# ===========================================================
def test3_bivariate_accuracy(perf):
    print_sep("TEST 3: Bivariate MAPE vs Q1 Baseline")
    print("  Muc tieu: MAPE moi (service, metric) lech <= 5pp so voi Q1\n")

    df_acc = perf.get_accuracy_report()
    if df_acc.empty:
        print("  [SKIP] Khong co accuracy data.")
        return "SKIP"

    rows = []
    for _, row in df_acc.iterrows():
        svc, metric, mape_v, f1_v = row['service'], row['metric'], row['mape_pct'], row['f1_score']
        q1 = Q1_BASELINE.get((svc, metric))
        if q1 is not None:
            diff = abs(mape_v - q1)
            ok   = diff <= TOLERANCE_PCT
            rows.append({'ok': ok})
            print(f"  [{'OK  ' if ok else 'DIFF'}] {svc:<12} {metric:<8} | "
                  f"now={mape_v:>5.1f}% | Q1={q1:>5.1f}% | diff={diff:>4.1f}pp | F1={f1_v:.3f}")
        else:
            print(f"  [--- ] {svc:<12} {metric:<8} | now={mape_v:>5.1f}% | F1={f1_v:.3f} (no Q1 baseline)")

    df_r   = pd.DataFrame(rows)
    n_ok   = df_r['ok'].sum() if not df_r.empty else 0
    total  = len(df_r)
    mean_mape = df_acc['mape_pct'].mean()
    mean_f1   = df_acc['f1_score'].mean()

    print_sep()
    print(f"  Pass {n_ok}/{total} | Mean MAPE={mean_mape:.2f}% | Mean F1={mean_f1:.3f}")
    verdict = "PASS" if n_ok == total else f"WARN ({total-n_ok} lech > {TOLERANCE_PCT}pp)"
    print(f"  KET LUAN TEST 3: {verdict}")
    return verdict


# ===========================================================
def test4_cascade_attenuation(sim):
    print_sep("TEST 4: Multi-hop Cascade Attenuation (Global DAG)")
    print("  Muc tieu: CPU(front-end) >= CPU(payment) sau do(+25%)\n")

    if not sim._is_trained:
        print("  [SKIP] SimulationAgent chua train.")
        return "SKIP"

    result = sim.simulate_intervention('front-end', 25.0, n_samples=200)

    print(f"  {'Service':<14} | {'CPU Delta':>10} | {'Mem Delta':>10} | {'Hops':>5}")
    print("  " + "-" * 48)

    cpu_vals = {}
    for svc in ['front-end','catalogue','user','carts','orders','payment','shipping']:
        if svc not in result: continue
        r    = result[svc]
        cpu  = r.get('cpu_change_pct')
        mem  = r.get('mem_change_pct')
        hops = r.get('n_hops', -1)
        print(f"  {svc:<14} | {(str(round(cpu,1))+'%') if cpu is not None else 'N/A':>10} | "
              f"{(str(round(mem,1))+'%') if mem is not None else 'N/A':>10} | {hops:>5}")
        if cpu is not None:
            cpu_vals[svc] = cpu

    fe  = cpu_vals.get('front-end')
    pay = cpu_vals.get('payment')
    print_sep()
    if fe is not None and pay is not None:
        ok = fe >= pay
        print(f"  front-end CPU {fe:+.1f}% >= payment CPU {pay:+.1f}%: {'CO' if ok else 'KHONG'}")
        verdict = "PASS" if ok else "FAIL"
    else:
        verdict = "SKIP (thieu du lieu)"
    print(f"  KET LUAN TEST 4: {verdict}")
    return verdict


# ===========================================================
def test5_parser_efficiency(arch):
    print_sep("TEST 5: Parser Efficiency (Rule-based fast path)")
    print("  Muc tieu: keyword ro rang -> similarity >= 0.6 (khong goi LLM them lan thu 2)\n")

    from parser_agent import ParserAgent, SIMILARITY_THRESHOLD

    class DummyLLM:
        def invoke(self, _):
            class R:
                content = '{"core_services":["front-end"],"adjustment":0.0,"reasoning":"ok","confidence":"MEDIUM"}'
            return R()

    parser = ParserAgent(llm=DummyLLM(), arch_agent=arch)

    cases = [
        ("place order buy product checkout",  "PLACE_ORDER"),
        ("add product to shopping cart",      "ADD_TO_CART"),
        ("view my shopping cart",             "VIEW_CART"),
        ("register a new user account",       "REGISTER"),
        ("apply discount promo code voucher", "APPLY_PROMO_CODE"),
    ]

    all_ok = True
    for text, expected_type in cases:
        parsed = parser.parse(text)
        sim_ok = parsed.similarity_score >= SIMILARITY_THRESHOLD
        type_ok = parsed.request_type == expected_type
        ok = sim_ok
        if not ok: all_ok = False
        print(f"  [{'OK  ' if ok else 'WARN'}] sim={parsed.similarity_score:.2f} | "
              f"type={parsed.request_type:<25} | \"{text[:40]}\"")

    verdict = "PASS" if all_ok else "WARN"
    print(f"\n  KET LUAN TEST 5: {verdict}")
    return verdict


# ===========================================================
if __name__ == '__main__':
    print(SEP)
    print("  BENCHMARK TEST v4 — Dual-Path SCM Architecture")
    print(SEP)

    t_all = time.time()
    arch, perf, sim = init_agents()

    results = {}
    r1, mae = test1_parser_calibration(arch)
    results['Test1 Parser Calibration'] = r1

    r2 = test2_guard_integrity(arch)
    results['Test2 Guard Integrity']    = r2

    r3 = test3_bivariate_accuracy(perf)
    results['Test3 Bivariate Accuracy'] = r3

    r4 = test4_cascade_attenuation(sim)
    results['Test4 Cascade Attenuation'] = r4

    r5 = test5_parser_efficiency(arch)
    results['Test5 Parser Efficiency']   = r5

    print_sep("TONG KET")
    for name, v in results.items():
        icon = "OK " if "PASS" in v else ("---" if "SKIP" in v else "!!!")
        print(f"  [{icon}] {name:<35}: {v}")
    print(f"\n  Thoi gian tong: {time.time()-t_all:.1f}s")
    print(SEP)
