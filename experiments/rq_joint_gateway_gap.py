# -*- coding: utf-8 -*-
"""
RQ MOI (thay the C_5b da bi bac): "gap" giua CAN THIEP DONG THOI va CONG DON LE
==================================================================================
Boi canh: RQ5b (RQ_FRAMEWORK_V3.md) da bac bo claim rang do-calculus co uu the
uoc luong hon conditioning O DUNG mien can thiep hien tai cua he thong (do(x)
tai 1 gateway, in-degree=0 -> P(Y|do(x))=P(Y|x) theo dinh ly Pearl, khong cat
canh backdoor nao ca). Hai phuong an thay the (do tren node noi bo/CPU-quota;
XAI/anomaly-attribution) da bi loai vi LECH MIEN: Parser Agent chi neo yeu cau
NGON NGU TU NHIEN cua khach hang vao GATEWAY (Gateway Invariance, xem
parser_agent.py), va bai toan la DU PHONG (chua co gi xay ra) nen khong co
factual anomaly de attribution phan ra.

Phuong an con lai dung vung (thao luan hoi thoai): CAN THIEP DONG THOI tai
gateway khi NHIEU yeu cau tinh nang duoc duyet trong CUNG mot cua so trien
khai -- kich ban rat thuc te trong bao tri phan mem, va la dung cho ma
Proposition 3 (capacity_agent.py: joint_certified_envelope, compare_joint_
vs_naive_sum) da chung minh bang Jensen's inequality: KHONG THE suy ra tu
conditioning/thong ke thong thuong, vi khong co du lieu lich su nao ghi nhan
"2 yeu cau tinh nang cung tang tai dong thoi" de ma dieu kien hoa len.

Diem khac voi Proposition 3 (2 gateway KHAC NHAU): du lieu RQ3 thuc te
(parser_ablation_benchmark.csv, cau hinh Guarded_Hybrid_Parser -- config
PRODUCTION duy nhat) cho thay ca 50/50 prompt deu injection tai CUNG MOT
gateway ('front-end') -- SockShop chi co 1 diem vao khach hang thuc su dung
trong 7 core services. Vi vay kich ban thuc te nhat KHONG PHAI "2 gateway
khac nhau" ma la "2 yeu cau CUNG gateway, delta khac nhau, duyet cung luc":

  - "Cong don le" (naive_sum)          : mo phong rieng do(front-end, d1),
                                          rieng do(front-end, d2), CONG hai
                                          ket qua lai (day la cach mot ky su
                                          se lam neu chi co bao cao tung
                                          feature rieng le, khong biet ca 2
                                          se len production cung dot).
  - "Gop dong thoi" (joint)             : mo phong MOT LAN do(front-end,
                                          d_joint) voi d_joint la muc tang tai
                                          THUC SU khi ca 2 dac tinh cung chay
                                          -- gia dinh CONG DON (d_joint =
                                          d1+d2), khop voi dinh nghia delta
                                          ADDITIVE dang dung trong Bounded
                                          Projection Pi (xem parser_agent.py,
                                          docs/UNIFIED_REFERENCE_DOC.md muc 4).

Gia thuyet (theo Proposition 3): voi node CPU/Mem (affine, sieu vi tuyen tinh
positive coefficients) gap ~ 0 (sieu vi cong tinh dung). Voi node Latency
(queueing, LOI: phi(w)=w/(c-w)) gap > 0 NGHIEM TRONG hon khi ca hai delta deu
duong -- boi vi phi loi tang nhanh hon tuyen tinh gan capacity, nen "cong hai
ket qua rieng" se DANH GIA THAP rui ro thuc te.

Du lieu can thiep: 50 cap (delta_pred, prompt_id) THAT tu RQ3 (config Guarded_
Hybrid_Parser, repeat_id=1 -- LLM that, khong phai gia lap) -- lay TAT CA
C(50,2)=1225 cap requirement, mo phong doi mot.

Output: data/processed/scm_results/rq_joint_gateway_gap.csv
        + <..>_summary.csv (ty le node co gap>0, gap trung binh/median theo
          loai metric, phan bo theo % gap tuong doi so voi baseline)
"""

import os
import sys
import itertools
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))
from capacity_agent import CapacityAgent

GATEWAY = 'front-end'


def load_real_deltas():
    """50 delta THAT tu RQ3 (Guarded_Hybrid_Parser, repeat_id=1) -- moi
    prompt_id mot delta duy nhat, LLM that (khong SYNTHETIC_)."""
    path = os.path.join(OUT_DIR, 'parser_ablation_benchmark.csv')
    df = pd.read_csv(path)
    g = df[(df['model'] == 'Guarded_Hybrid_Parser') & (df['repeat_id'] == 1)]
    g = g[g['injection_service'] == GATEWAY]
    assert not str(g['llm_backend'].iloc[0]).startswith('SYNTHETIC_'), \
        "Du lieu la gia lap offline (SYNTHETIC_*), khong phai LLM that -- dung dung"
    return g[['prompt_id', 'requirement', 'pred_delta']].reset_index(drop=True)


def main():
    print("[1/3] Huan luyen CapacityAgent (SockShop, Global DAG)...")
    agent = CapacityAgent(system_type='sockshop')

    inj_col = f"{GATEWAY}_workload"
    base_wl = float(agent.global_df_baseline[inj_col].mean())
    print(f"  Baseline {inj_col} = {base_wl:.2f} req/s")

    print("\n[2/3] Nap 50 delta that tu RQ3 (Guarded_Hybrid_Parser)...")
    reqs = load_real_deltas()
    print(f"  {len(reqs)} yeu cau, delta trong [{reqs['pred_delta'].min()}, {reqs['pred_delta'].max()}]")

    baseline_vals, _ = agent._deterministic_forward(inj_col, base_wl)

    # Cache: mo phong tung delta DON LE 1 lan (khong lap lai cho tung cap)
    print("\n  Mo phong 50 kich ban DON LE (cache)...")
    single_cache = {}
    for _, row in reqs.iterrows():
        d = row['pred_delta']
        if d not in single_cache:
            target = base_wl * (1.0 + d / 100.0)
            vals, breached = agent._deterministic_forward(inj_col, target)
            single_cache[d] = (vals, breached)

    print("\n[3/3] Mo phong TAT CA C(50,2) cap (gop dong thoi vs cong don le)...")
    rows = []
    pairs = list(itertools.combinations(reqs.index, 2))
    for i, j in pairs:
        r1, r2 = reqs.loc[i], reqs.loc[j]
        d1, d2 = r1['pred_delta'], r2['pred_delta']
        d_joint = d1 + d2  # dinh nghia ADDITIVE, dong bo voi Bounded Projection Pi

        vals1, _ = single_cache[d1]
        vals2, _ = single_cache[d2]
        target_joint = base_wl * (1.0 + d_joint / 100.0)
        vals_joint, breached_joint = agent._deterministic_forward(inj_col, target_joint)

        for node, base_v in baseline_vals.items():
            if node == inj_col:
                continue
            m1 = vals1.get(node, base_v) - base_v  # hieu ung rieng cua yeu cau 1
            m2 = vals2.get(node, base_v) - base_v  # hieu ung rieng cua yeu cau 2
            naive_sum = base_v + m1 + m2
            joint = vals_joint.get(node, base_v)
            if node in breached_joint or joint == float('inf') or naive_sum == float('inf'):
                continue
            gap = joint - naive_sum
            gap_rel_pct = (gap / abs(base_v) * 100.0) if base_v != 0 else np.nan
            rows.append({
                'prompt_id_1': r1['prompt_id'], 'delta_1': d1,
                'prompt_id_2': r2['prompt_id'], 'delta_2': d2,
                'node': node,
                'node_type': 'latency' if node.endswith('_latency-50') else
                             ('cpu' if node.endswith('_cpu') else
                              ('mem' if node.endswith('_mem') else 'workload')),
                'baseline': base_v,
                'naive_sum': naive_sum,
                'joint': joint,
                'gap': gap,
                'gap_rel_pct': gap_rel_pct,
            })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'rq_joint_gateway_gap.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\n[OK] Da luu {len(df_out)} dong: {out_path}")

    print("\n" + "=" * 78)
    print("  TONG HOP theo loai node")
    print("=" * 78)
    summary = df_out.groupby('node_type').agg(
        n=('gap', 'count'),
        pct_gap_gt0=('gap', lambda s: (s > 1e-9).mean() * 100),
        median_gap_rel_pct=('gap_rel_pct', 'median'),
        p90_gap_rel_pct=('gap_rel_pct', lambda s: s.quantile(0.90)),
        max_gap_rel_pct=('gap_rel_pct', 'max'),
    ).round(4)
    print(summary.to_string())
    summary.to_csv(os.path.join(OUT_DIR, 'rq_joint_gateway_gap_summary.csv'))

    print("\n  Top 5 (service,cap-delta) co gap tuong doi lon nhat (latency):")
    lat = df_out[df_out['node_type'] == 'latency'].nlargest(5, 'gap_rel_pct')
    print(lat[['node', 'delta_1', 'delta_2', 'naive_sum', 'joint', 'gap_rel_pct']].to_string(index=False))


if __name__ == '__main__':
    main()
