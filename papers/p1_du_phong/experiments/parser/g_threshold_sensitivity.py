# -*- coding: utf-8 -*-
"""
Threshold Sensitivity Analysis for G2/G3/G5 — Are the Specific Numbers Load-Bearing?
========================================================================================
G2 ([5,50]%), G3 (+/-10%), and G5 (similarity >= 0.6) are hand-set engineering judgments
(src/agents/parser_agent.py MIN_DELTA_PCT/MAX_DELTA_PCT/MAX_ADJUSTMENT_PCT/
SIMILARITY_THRESHOLD), not derived from a statistical or literature-grounded process.
This is a legitimate objection a reviewer would raise: are RQ3's guarded-configuration
results (PBVR/SHR/GMR = 0.0%) an artifact of one lucky choice of thresholds, or does the
*enforcement mechanism itself* -- not the exact numbers -- do the real work?

CAP NHAT KIEN TRUC (quan trong, doc truoc khi dung lai so lieu cu):
Ban dau cua script nay (du lieu bat ngay 2026-09-09, file g_threshold_raw_capture.csv/
g_threshold_sensitivity.csv cu) gia dinh ParserAgent.parse() co MOT NHANH FAST-PATH
(rb_similarity >= SIMILARITY_THRESHOLD -> bo qua LLM, dung thang anchor). Fast-path
DA BI GO BO HOAN TOAN khoi parser_agent.py (xem docstring ParserAgent.parse(): phat
hien 8/11 prompt doi khang dung du khop tu khoa VOI MOT archetype SAI van du diem kich
hoat fast-path, lam Scope Gate khong bao gio duoc hoi). Ban cu CUNG gia dinh
_llm_full_parse() tra ve 7 gia tri -- ham hien tai tra ve 8 (them cf_confidence, dung
cho Scope Gate G6). Chay ban cu tren code hien tai se raise ValueError khi unpack.

Ket qua THUC TE da doi hoan toan sau khi sua (xem lich su hoi thoai): voi prompt hien
tai cua _llm_full_parse() (noi ro "KHONG bi gioi han truoc boi bat ky khoang nao --
neu thay bat thuong, cu de xuat so lon"), capture lai tren dung 50 prompt RQ3 cho
PBVR=8.00% KHI BO G2/G3 (4/50, gom 2 ca cuc doan: REQ-41 adjustment=+1000 -> delta=1025%,
REQ-45 adjustment=+500 -> delta=520%, ca hai deu o nhom Adversarial_Stress) -- hoan toan
nguoc voi ket luan "0% o moi bien the" cua ban cu (vi ban cu bat tren mot phien ban
prompt cu, bao thu hon). KET LUAN DUNG: G2/G3 DANG chan that, khong duoc bo.

Design (khong doi): bat MOT LAN cho moi prompt dung _llm_full_parse() de lay de xuat
THO (request_type, raw adjustment, raw core_services, cac similarity) cho ca 50 prompt,
roi replay THU CONG qua nhieu cau hinh (G2, G3, G5) hoan toan ngoai tuyen (khong goi LLM
them) de tach bien ngau nhien cua LLM (da do rieng o RQ3) khoi bien ngung. Replay dung
LAI DUNG _guard_delta()/_guard_core_services() tu parser_agent.py (khong viet lai cong
thuc) de khong bao gio lech voi guard THAT dang chay.

Output:
  data/processed/scm_results/g_threshold_raw_capture.csv      (raw, one live LLM pass)
  data/processed/scm_results/g_threshold_sensitivity.csv      (offline replay results)
"""

import os
import sys
import json
import time

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while PROJECT_ROOT != os.path.dirname(PROJECT_ROOT) and not os.path.isdir(os.path.join(PROJECT_ROOT, 'src')):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
assert os.path.isdir(os.path.join(PROJECT_ROOT, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from architecture_agent import ArchitectureAgent
from parser_agent import (ParserAgent, _compute_similarity, _guard_delta,
                          _guard_core_services, KNOWN_SERVICES,
                          MIN_DELTA_PCT, MAX_DELTA_PCT, MAX_ADJUSTMENT_PCT,
                          SIMILARITY_THRESHOLD)
from request_router import classify_request, CALL_CHAINS
from llm_backends import get_llm as _factory_get_llm, has_credentials

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
PROMPTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'benchmark', 'parser_benchmark_prompts.json')
GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'sockshop_agent_graph.json')

# Baseline (current production) thresholds, for reference in the sweep table.
BASELINE = {'min_delta': MIN_DELTA_PCT, 'max_delta': MAX_DELTA_PCT,
            'max_adj': MAX_ADJUSTMENT_PCT, 'sim_thresh': SIMILARITY_THRESHOLD}

GATEWAY_FALLBACK = 'front-end'   # SockShop chi co 1 gateway -- dung cho G4 fallback khi loc rong


def get_live_llm():
    """Khong fallback am tham -- giong quy uoc cua cac script live-LLM khac trong du an."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    if not has_credentials('google'):
        raise SystemExit("[FATAL] Thieu GOOGLE_API_KEY -- script nay can mot lan chay LLM thuc.")
    llm = _factory_get_llm('gemini-flash-lite', temperature=0.2, max_retries=5)
    from langchain_core.messages import HumanMessage
    llm.invoke([HumanMessage(content="Reply with exactly: OK")])  # xac thuc goi duoc truoc khi dung "LIVE"
    return llm


def capture_raw(n_repeats=1):
    """Mot lan goi LLM thuc moi prompt moi lan lap -- goi _llm_full_parse() TRUC TIEP
    (KHONG qua parse()) de lay de xuat THO, chua qua bat ky guard nao (G2/G3/G4/G5/G6).

    KHONG con khai niem "bo qua fast-path" (da go fast-path khoi parse()) -- moi
    requirement deu di qua _llm_full_parse() trong ca pipeline that, nen o day cung
    goi dung ham do, voi dung tham so (rb_suggestion) nhu parse() dang truyen.
    """
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    arch = ArchitectureAgent(GRAPH_PATH)
    llm = get_live_llm()
    parser = ParserAgent(llm=llm, arch_agent=arch)

    rows = []
    for repeat_id in range(1, n_repeats + 1):
        for p in prompts:
            req = p['requirement']
            rb_request_type = classify_request(req)
            rb_similarity = _compute_similarity(req, rb_request_type)

            # _llm_full_parse() hien tai tra ve 8 gia tri (them cf_confidence cho G6,
            # chua dung o day nhung phai unpack du de khong loi).
            (request_type, core_svcs, adjustment, reasoning, llm_conf,
             raw_inj_svc, llm_is_customer_facing, cf_conf) = parser._llm_full_parse(req, rb_request_type)

            llm_pick_similarity = _compute_similarity(req, request_type)
            similarity_score = max(rb_similarity, llm_pick_similarity)

            template_info = CALL_CHAINS.get(request_type, {})
            template_delta = template_info.get('expected_delta_pct', 20.0)
            affected = template_info.get('services', ['front-end'])

            rows.append({
                'repeat_id': repeat_id, 'prompt_id': p['id'], 'category': p['category'],
                'requirement': req, 'expected_anchor': p['expected_anchor'],
                'expected_gateway': p['expected_gateway'],
                'rb_request_type': rb_request_type, 'rb_similarity': round(rb_similarity, 4),
                'llm_request_type': request_type, 'llm_raw_adjustment': adjustment,
                'llm_raw_core_services': str(core_svcs),
                'llm_pick_similarity': round(llm_pick_similarity, 4),
                'similarity_score': round(similarity_score, 4),
                'llm_is_customer_facing': llm_is_customer_facing,
                'template_delta': template_delta, 'affected_services': str(affected),
            })
            print(f"  [repeat {repeat_id}] {p['id']} ({p['category']}): "
                  f"sim={similarity_score:.2f} llm_type={request_type} raw_adj={adjustment:+.1f}")
            time.sleep(4.5)  # free-tier rate-limit, giong parser_benchmark_suite.py

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'g_threshold_raw_capture.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Raw capture saved: {csv_path} ({len(df)} rows)")
    return df


def replay(df: pd.DataFrame, min_delta, max_delta, max_adj, sim_thresh) -> dict:
    """Tai dung lai outcome cua pipeline Guarded cho moi dong da bat, voi MOT cau hinh
    nguong cho truoc -- dung LAI chinh _guard_delta()/_guard_core_services() cua
    parser_agent.py (khong viet lai cong thuc rieng) de khong bao gio lech voi guard
    THAT. KHONG con nhanh fast-path (da go khoi parse() -- xem CAP NHAT KIEN TRUC o
    dau file): moi dong deu di qua G4 -> G5 -> G2+G3, dung thu tu nhu parse().

    LOI DA SUA (quan trong): PBV PHAI cham theo hang so VAT LY CO DINH
    MIN_DELTA_PCT/MAX_DELTA_PCT (5%/50%, dung chinh xac cong thuc RQ3 that trong
    parser_benchmark_suite.py: `pbv = (delta < 5.0) or (delta > 50.0)`), KHONG
    duoc cham theo (min_delta, max_delta) DANG SWEEP. Ban truoc cham theo chinh
    cap dang sweep -- tu quy chieu: _guard_delta() LUON ep delta vao [min_delta,
    max_delta] o buoc cuoi, nen PBV do lai bang CHINH cap do se LUON la False bat
    ke sweep gia tri gi (ke ca "bo han" [-1e9,1e9] -- delta=1025% van nam "trong"
    [-1e9,1e9] mot cach vo nghia). Hau qua: moi dong trong bang sweep CU deu bao
    PBVR=0.0% mot cach gia tao, che mat phat hien thuc (REQ-41 delta=1025%,
    REQ-45 delta=520% khi thuc su bo G2/G3 -- xem lich su hoi thoai).
    """
    pbv_flags, sh_flags, errors = [], [], []
    for _, row in df.iterrows():
        core_svcs_raw = eval(row['llm_raw_core_services'])
        adjustment = row['llm_raw_adjustment']
        similarity_score = row['similarity_score']
        template_delta = row['template_delta']

        # G4: loc core_services theo known_services (KNOWN_SERVICES cua module = SockShop)
        core_svcs = _guard_core_services(core_svcs_raw, KNOWN_SERVICES, GATEWAY_FALLBACK)

        # G5: similarity thap -> siet adjustment ve 0
        if similarity_score < sim_thresh:
            adjustment = 0.0

        # G2 + G3: dung THANG ham guard that, voi nguong dang sweep
        raw_delta = template_delta + adjustment
        delta, _adj_clamped, _was_clamped = _guard_delta(raw_delta, template_delta,
                                                          min_delta, max_delta, max_adj)

        # Cham theo hang so VAT LY CO DINH (5%/50%), KHONG theo cap dang sweep --
        # xem docstring o tren.
        pbv_flags.append((delta < MIN_DELTA_PCT) or (delta > MAX_DELTA_PCT))
        hallucinated = [s for s in core_svcs_raw if s not in KNOWN_SERVICES]  # SH do tren DE XUAT THO (truoc G4)
        sh_flags.append(len(hallucinated) > 0)
        errors.append(abs(delta - row['expected_anchor']))

    n = len(df)
    return {
        'min_delta': min_delta, 'max_delta': max_delta, 'max_adj': max_adj,
        'sim_thresh': sim_thresh,
        'PBVR_pct': round(100.0 * sum(pbv_flags) / n, 2),
        'SHR_pct': round(100.0 * sum(sh_flags) / n, 2),
        'anchor_MAE': round(sum(errors) / n, 3),
        'n': n,
    }


def run_sweep(df: pd.DataFrame):
    variants = []
    b = BASELINE
    # Vary G2 alone (+ "bo han" = [-1e9, 1e9], tra loi dung cau hoi "neu bo di thi sao")
    for lo, hi in [(3, 60), (5, 50), (8, 40), (10, 30), (1, 100), (-1e9, 1e9)]:
        label = f'[{lo},{hi}]' if hi < 1e8 else 'KHONG_KEP (bo han)'
        variants.append(('G2', label, replay(df, lo, hi, b['max_adj'], b['sim_thresh'])))
    # Vary G3 alone (+ "bo han" = +/-1e9)
    for adj in [5.0, 10.0, 15.0, 20.0, 30.0, 1e9]:
        label = f'+/-{adj}' if adj < 1e8 else 'KHONG_KEP (bo han)'
        variants.append(('G3', label, replay(df, b['min_delta'], b['max_delta'], adj, b['sim_thresh'])))
    # Vary G5 alone
    for sim in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        variants.append(('G5', f'>={sim}', replay(df, b['min_delta'], b['max_delta'], b['max_adj'], sim)))
    # Bo CA G2 va G3 cung luc (dung trang thai "khong guard delta/adjustment nao ca")
    variants.append(('G2+G3', 'KHONG_KEP CA HAI',
                     replay(df, -1e9, 1e9, 1e9, b['sim_thresh'])))

    rows = []
    for guard, label, res in variants:
        res2 = dict(res)
        res2['guard_varied'] = guard
        res2['variant'] = label
        rows.append(res2)
    out = pd.DataFrame(rows)[['guard_varied', 'variant', 'min_delta', 'max_delta', 'max_adj',
                                'sim_thresh', 'PBVR_pct', 'SHR_pct', 'anchor_MAE', 'n']]
    csv_path = os.path.join(OUTPUT_DIR, 'g_threshold_sensitivity.csv')
    out.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Sensitivity sweep saved: {csv_path}")
    print(out.to_string(index=False))
    return out


def main():
    n_repeats = 1
    for arg in sys.argv:
        if arg.startswith('--repeats='):
            n_repeats = int(arg.split('=', 1)[1])
    raw_csv = os.path.join(OUTPUT_DIR, 'g_threshold_raw_capture.csv')
    if '--skip-capture' in sys.argv and os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
        print(f"[SKIP] Reusing existing raw capture: {raw_csv} ({len(df)} rows)")
    else:
        df = capture_raw(n_repeats=n_repeats)
    run_sweep(df)


if __name__ == '__main__':
    main()
