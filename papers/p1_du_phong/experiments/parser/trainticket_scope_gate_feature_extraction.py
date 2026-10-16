# -*- coding: utf-8 -*-
"""
Train Ticket Scope Gate -- feature extraction (live LLM), KHONG fit lai gi ca
================================================================================
Muc dich: thu thap 4 dac trung tho (rb_similarity, llm_pick_similarity,
keyword_similarity_score, llm_is_customer_facing) cho bo 28 prompt da duoc
NGUOI DUNG DUYET tai data/benchmark/trainticket_scope_gate_prompts_DRAFT.csv
(20 hop le x 2/archetype + 8 doi khang: 6 trung tu khoa kieu polysemous, 1
khong trung tu khoa, 1 thu prompt-injection), bang CACH GOI DUNG pipeline
cong khai ParserAgent.parse() -- GIONG HET phuong phap da dung de thu
data/processed/scm_results/g6_scope_gate_layerA_validation.csv (61 prompt
SockShop) va scope_gate_holdout_test.py (20 prompt holdout SockShop).

Script nay CHI thu dac trung + ghi CSV. KHONG fit logistic regression, KHONG
tinh nguong conformal moi -- buoc do o mot script rieng (xem TODO cuoi file)
dung dung phuong pháp cua scope_gate_conformal_calibration.py, de tach biet
"thu du lieu" khoi "hieu chinh mo hinh" (tranh vua ra de vua tu cham diem
trong CHINH 1 buoc).

Backend: gemini-flash-lite (= model gemini-flash-lite-latest qua
llm_backends.BACKENDS['gemini-flash-lite']) -- DUNG backend da dung de thu
ca 61-prompt validation set va 20-prompt holdout set cua SockShop
(g6_scope_gate_layerA_test.py, scope_gate_holdout_test.py), de ket qua Train
Ticket co the so sanh ngang hang, khong lech vi doi model giua 2 he thong.

KHONG fallback am tham: neu khong goi duoc LLM thuc (thieu key / 429 / loi
mang), script DUNG HAN (SystemExit) thay vi am tham sinh du lieu gia -- dung
bai hoc da ghi lai nhieu lan trong parser_benchmark_suite.py (mot lan quota
tran giua chung tung bien 48/50 dong thanh fallback 3-bo-an-toan ma khong ai
de y, lam sai lech ca mot bang da cong bo).

Ghi tang dan (flush ngay moi dong) -- 28 dong la nho nen it rui ro, nhung van
giu dung quy uoc cua du an de tranh mat du lieu neu bi ngat giua chung.
"""

import os
import sys
import time
import csv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/
while PROJECT_ROOT != os.path.dirname(PROJECT_ROOT) and not os.path.isdir(os.path.join(PROJECT_ROOT, 'src')):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
assert os.path.isdir(os.path.join(PROJECT_ROOT, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'agents'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from architecture_agent import ArchitectureAgent
from parser_agent import ParserAgent
from taxonomies.trainticket import TRAINTICKET_CALL_CHAINS, TRAINTICKET_GATEWAY
from llm_backends import get_llm as _factory_get_llm, BACKENDS as _BACKENDS, has_credentials

BACKEND_NAME = os.environ.get("TT_SCOPE_GATE_BACKEND", "gemini-flash-lite")
for _a in sys.argv:
    if _a.startswith('--backend='):
        BACKEND_NAME = _a.split('=', 1)[1]

GRAPH_PATH = os.path.join(PROJECT_ROOT, 'src', 'graph', 'trainticket_agent_graph.json')
PROMPTS_PATH = os.path.join(PROJECT_ROOT, 'data', 'benchmark',
                            'trainticket_scope_gate_prompts_DRAFT.csv')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results',
                           'trainticket_scope_gate_layerA_validation.csv')


def get_live_llm():
    """Khong fallback am tham. Loi/thieu key -> SystemExit, khong sinh du lieu gia."""
    cfg = _BACKENDS.get(BACKEND_NAME)
    if cfg is None:
        raise SystemExit(f"[FATAL] Backend '{BACKEND_NAME}' khong co trong BACKENDS. "
                          f"Chon trong: {sorted(_BACKENDS)}")
    if not has_credentials(cfg['provider']):
        raise SystemExit(f"[FATAL] Thieu API key cho provider '{cfg['provider']}' "
                          f"(backend='{BACKEND_NAME}'). Khong chay fallback synthetic.")
    try:
        llm = _factory_get_llm(BACKEND_NAME, temperature=0.2, max_retries=5)
        from langchain_core.messages import HumanMessage
        llm.invoke([HumanMessage(content="Reply with exactly: OK")])
        print(f"[LLM] Using LIVE {cfg['provider']} {cfg['model']} (tier={cfg['tier']}).")
        return llm
    except Exception as e:
        raise SystemExit(f"[FATAL] Live call that bai tren backend '{BACKEND_NAME}': "
                          f"{str(e)[:300]}\n        Khong fallback synthetic -- doi quota/kiem "
                          f"tra mang roi chay lai.")


def _sleep_between_calls():
    cfg = _BACKENDS.get(BACKEND_NAME, {})
    if cfg.get('provider') == 'google':
        time.sleep(6.0 if cfg.get('tier') == 'frontier' else 4.5)
    elif cfg.get('provider') == 'groq':
        time.sleep(2.5)


def main():
    print("=" * 80)
    print("  TRAIN TICKET SCOPE GATE -- FEATURE EXTRACTION (live LLM, 28 prompts)")
    print("=" * 80)

    import pandas as pd
    prompts_df = pd.read_csv(PROMPTS_PATH)
    print(f"Loaded {len(prompts_df)} prompts tu {os.path.basename(PROMPTS_PATH)} "
          f"({(prompts_df['source'] == 'TT_legitimate').sum()} hop le, "
          f"{(prompts_df['source'] == 'TT_adversarial').sum()} doi khang).")

    llm = get_live_llm()
    arch = ArchitectureAgent(GRAPH_PATH)
    parser = ParserAgent(
        llm=llm, arch_agent=arch, system_name="TrainTicket",
        domain_description="an online train ticket booking platform (FudanSELab Train Ticket)",
        call_chains=TRAINTICKET_CALL_CHAINS, default_gateway=TRAINTICKET_GATEWAY,
    )  # Scope Gate dung he so/nguong SockShop mac dinh o day -- CHI de debug
       # (xem scope_gate_score in trong log), KHONG dung lam ket qua cuoi. Buoc
       # fit lai rieng (script ke tiep) moi tao bo he so/nguong THAT cho Train Ticket.

    fieldnames = ['prompt_id', 'source', 'true_should_refuse', 'expected_type',
                  'rb_similarity', 'llm_pick_similarity', 'keyword_similarity_score',
                  'llm_is_customer_facing', 'scope_gate_score_sockshop_coef_DEBUG_ONLY',
                  'predicted_refuse_sockshop_thresholds', 'predicted_review_sockshop_thresholds',
                  'requirement']

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    rows = []
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, p in enumerate(prompts_df.itertuples(), start=1):
            try:
                result = parser.parse(p.requirement)
            except Exception as e:
                f.flush()
                raise SystemExit(
                    f"\n[STOP] Loi khi goi LLM tai prompt {i}/{len(prompts_df)} "
                    f"({p.prompt_id}): {str(e)[:300]}\n"
                    f"  Da ghi an toan {len(rows)} dong vao {OUTPUT_PATH}\n"
                    f"  Neu la 429/quota: doi roi chay lai toan bo (chi 28 dong, re hon resume).")

            raw = result.scope_gate_raw_features or {}
            row = {
                'prompt_id': p.prompt_id,
                'source': p.source,
                'true_should_refuse': p.true_should_refuse,
                'expected_type': p.expected_type if isinstance(p.expected_type, str) else '',
                'rb_similarity': raw.get('rb_similarity'),
                'llm_pick_similarity': raw.get('llm_pick_similarity'),
                'keyword_similarity_score': raw.get('keyword_similarity_score'),
                'llm_is_customer_facing': raw.get('llm_is_customer_facing'),
                'scope_gate_score_sockshop_coef_DEBUG_ONLY': result.scope_gate_score,
                'predicted_refuse_sockshop_thresholds': result.is_out_of_scope,
                'predicted_review_sockshop_thresholds': result.needs_human_review,
                'requirement': p.requirement,
            }
            rows.append(row)
            writer.writerow(row)
            f.flush()

            outcome = ('REFUSE' if result.is_out_of_scope else
                       'REVIEW' if result.needs_human_review else 'PASS')
            print(f"  [{i:2d}/{len(prompts_df)}][{p.prompt_id}] true_refuse={p.true_should_refuse!s:5s} "
                  f"outcome(SockShop coef, debug)={outcome:7s} kw_sim={row['keyword_similarity_score']:.2f} "
                  f"cust_facing={row['llm_is_customer_facing']}")

            _sleep_between_calls()

    print(f"\n[OK] Saved {len(rows)} rows -> {OUTPUT_PATH}")

    df = pd.DataFrame(rows)
    y = df['true_should_refuse']
    print("\n" + "=" * 80)
    print("[DIAGNOSTIC ONLY -- khong phai ket qua cong bo] Hanh vi voi NGUONG/HE SO")
    print("CUA SOCKSHOP (chua hieu chinh cho Train Ticket):")
    print(f"  Auto-refuse dung (true out-of-scope bi refuse):   "
          f"{(df['predicted_refuse_sockshop_thresholds'] & y).sum()}/{y.sum()}")
    print(f"  Auto-refuse SAI (legitimate bi refuse oan):        "
          f"{(df['predicted_refuse_sockshop_thresholds'] & ~y).sum()}/{(~y).sum()}")
    print(f"  Day sang nguoi xem (NEEDS_HUMAN_REVIEW):           "
          f"{df['predicted_review_sockshop_thresholds'].sum()}/{len(df)}")
    print("=" * 80)
    print("\n[NEXT STEP] Day MOI chi la dac trung tho + ket qua debug voi he so SockShop.")
    print("  Can mot script rieng (theo dung phuong phap scope_gate_conformal_calibration.py)")
    print(f"  doc {os.path.basename(OUTPUT_PATH)}, fit logistic regression + 2 nguong conformal")
    print("  MOI cho Train Ticket tu 4 cot rb_similarity/llm_pick_similarity/")
    print("  keyword_similarity_score/llm_is_customer_facing + true_should_refuse o tren.")
    return df


if __name__ == '__main__':
    main()
