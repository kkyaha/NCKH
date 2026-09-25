# -*- coding: utf-8 -*-
"""
RQ6 — Does Interventional Shapley Attribution Correctly Localize Impact?
==========================================================================
RQ6: "Does the interventional Shapley attribution mechanism (Budhathoki et
      al.'s outlier root-cause algorithm, run forward on do(x) samples
      instead of backward on observed outliers) behave like a trustworthy
      explanation rather than an arbitrary one?"

This validates experiments/future_rca.py's FutureRCAEngine — the module that
actually calls gcm.attribute_anomalies() on interventional samples (the
Capacity Agent evaluated elsewhere in this paper does NOT call it; it uses a
fixed-threshold heuristic for bottleneck ranking instead — see the RQ2
caveat in the paper). Nothing here touches the LLM API: it is pure SCM/DoWhy
computation on the already-trained 28-node Sock Shop DAG, so it runs
regardless of API quota.

Part A (this script) — cheap validity checks that do not require a second
labeled dataset, only repetition and a known control condition:

  A.1 Dose-response monotonicity: does the injection node's own anomaly
      score / Shapley contribution increase monotonically as the injected
      workload delta increases (0% -> 300%)? A sensible attribution
      mechanism should respond to a bigger dial with a bigger reading.

  A.2 Null-condition false-positive rate: run the "no real change" scenario
      (do(front-end_workload = its own training mean)) N independent times
      (different Monte Carlo draws only) and measure how often the
      risk-level heuristic (>=30% change or anomaly>=3.0 => "critical")
      fires on a node anyway, purely from sampling noise. A single earlier
      run of this scenario already flagged one node "critical" at 0% real
      load change — this quantifies how often that happens rather than
      leaving it as a one-off anecdote.

Deferred (Part A.3, not run here): a topology-distance check on Train
Ticket's larger graph (which — unlike Sock Shop — has nodes that are
genuinely NOT downstream of a given injection point, giving a cleaner
"attribution should be near-zero here" test). Left for a follow-up run
since it needs a second trained DAG wired in.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
import _paths  # noqa: F401  -- dua cac nhom con khac vao sys.path
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'scm'))

from evaluation_suite import build_and_train_global_dag
from future_rca import FutureRCAEngine

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scm_results')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# Injection node and the node we expect to react most directly to it.
INJECTION_NODE = 'front-end_workload'
PRIMARY_TARGET = 'front-end_cpu'


def run_rq6_part_a(n_repeats_null=20, n_repeats_dose=5):
    for arg in sys.argv:
        if arg.startswith('--null-repeats='):
            n_repeats_null = int(arg.split('=', 1)[1])
        if arg.startswith('--dose-repeats='):
            n_repeats_dose = int(arg.split('=', 1)[1])

    print("=" * 80)
    print("   RQ6 (Part A): INTERVENTIONAL SHAPLEY ATTRIBUTION — VALIDITY CHECKS")
    print(f"   Null-condition repeats: {n_repeats_null} | Dose-response repeats/level: {n_repeats_dose}")
    print("   No LLM API calls — pure SCM/DoWhy computation on Sock Shop's 28-node DAG.")
    print("=" * 80)

    print("\n[Setup] Training Global 28-node DAG (Sock Shop)...")
    global_model, df_sub, g_sub = build_and_train_global_dag()
    engine = FutureRCAEngine(global_model, df_sub, g_sub)
    base_wl = df_sub[INJECTION_NODE].mean()
    print(f"[Setup] Baseline {INJECTION_NODE} (training mean): {base_wl:.3f}")

    # ================================================================
    # A.2 — NULL-CONDITION FALSE-POSITIVE RATE
    # (run first: cheapest, and its finding — is the threshold noisy? —
    #  contextualizes how to read A.1's dose-response magnitudes.)
    # ================================================================
    print(f"\n[A.2] Running the null condition ({INJECTION_NODE} = its own "
          f"training mean, i.e. no real change) {n_repeats_null} independent times...")
    null_rows = []
    any_flag_per_repeat = []
    for i in range(n_repeats_null):
        result = engine.analyze(f"NULL-{i}", {INJECTION_NODE: base_wl})
        flagged_nodes = [nr.node for nr in result.node_risks if nr.risk_level != 'normal']
        any_flag_per_repeat.append(len(flagged_nodes) > 0)
        for nr in result.node_risks:
            null_rows.append({
                'repeat': i, 'node': nr.node, 'change_pct': nr.change_pct,
                'z_score': nr.z_score, 'anomaly_score': nr.anomaly_score,
                'shapley_contribution': nr.shapley_contribution,
                'risk_level': nr.risk_level,
            })
        print(f"  [null {i+1}/{n_repeats_null}] flagged non-normal: {flagged_nodes if flagged_nodes else '(none)'}")

    null_df = pd.DataFrame(null_rows)
    node_level_fp_rate = 100.0 * (null_df['risk_level'] != 'normal').mean()
    scenario_level_fp_rate = 100.0 * np.mean(any_flag_per_repeat)
    per_node_fp = (
        null_df.assign(flagged=lambda d: d['risk_level'] != 'normal')
        .groupby('node')['flagged'].mean().mul(100).round(1).sort_values(ascending=False)
    )

    null_csv = os.path.join(OUTPUT_DIR, 'rq6_null_condition_fp_rate.csv')
    null_df.to_csv(null_csv, index=False, encoding='utf-8')
    print(f"\n[OK] Null-condition raw results saved to: {null_csv}")
    print(f"     Node-level false-positive rate (any node, any repeat): {node_level_fp_rate:.1f}%")
    print(f"     Scenario-level false-positive rate (>=1 node flagged per repeat): {scenario_level_fp_rate:.1f}%")
    print("     Per-node false-positive rate:")
    print(per_node_fp.to_string())

    # ================================================================
    # A.1 — DOSE-RESPONSE MONOTONICITY
    # ================================================================
    levels_pct = [0.0, 20.0, 50.0, 150.0, 300.0]
    print(f"\n[A.1] Running dose-response across levels {levels_pct}% "
          f"x {n_repeats_dose} repeats each...")
    dose_rows = []
    for level_pct in levels_pct:
        wl = base_wl * (1.0 + level_pct / 100.0)
        for i in range(n_repeats_dose):
            result = engine.analyze(f"DOSE-{level_pct}-{i}", {INJECTION_NODE: wl})
            for nr in result.node_risks:
                dose_rows.append({
                    'level_pct': level_pct, 'repeat': i, 'node': nr.node,
                    'change_pct': nr.change_pct, 'z_score': nr.z_score,
                    'anomaly_score': nr.anomaly_score,
                    'shapley_contribution': nr.shapley_contribution,
                    'risk_level': nr.risk_level,
                })
        print(f"  [dose {level_pct}%] done ({n_repeats_dose} repeats)")

    dose_df = pd.DataFrame(dose_rows)
    dose_csv = os.path.join(OUTPUT_DIR, 'rq6_dose_response.csv')
    dose_df.to_csv(dose_csv, index=False, encoding='utf-8')
    print(f"[OK] Dose-response raw results saved to: {dose_csv}")

    primary = dose_df[dose_df['node'] == PRIMARY_TARGET]
    dose_summary = primary.groupby('level_pct').agg(
        anomaly_mean=('anomaly_score', 'mean'), anomaly_std=('anomaly_score', 'std'),
        shapley_mean=('shapley_contribution', 'mean'), shapley_std=('shapley_contribution', 'std'),
    ).reset_index()

    spearman_anomaly = dose_summary['level_pct'].corr(dose_summary['anomaly_mean'], method='spearman')
    spearman_shapley = dose_summary['level_pct'].corr(dose_summary['shapley_mean'], method='spearman')
    is_monotonic_anomaly = dose_summary['anomaly_mean'].is_monotonic_increasing
    is_monotonic_shapley = dose_summary['shapley_mean'].is_monotonic_increasing

    print(f"\n[A.1] Dose-response summary for '{PRIMARY_TARGET}' "
          f"(mean +/- std over {n_repeats_dose} repeats/level):")
    print(dose_summary.to_string(index=False))
    print(f"\n     Spearman(level, mean anomaly_score) = {spearman_anomaly:.3f} "
          f"| strictly monotonic increasing: {is_monotonic_anomaly}")
    print(f"     Spearman(level, mean shapley_contribution) = {spearman_shapley:.3f} "
          f"| strictly monotonic increasing: {is_monotonic_shapley}")

    dose_summary_csv = os.path.join(OUTPUT_DIR, 'rq6_dose_response_summary.csv')
    dose_summary.to_csv(dose_summary_csv, index=False, encoding='utf-8')

    # ================================================================
    # MARKDOWN REPORT
    # ================================================================
    md = f"""# 📊 BÁO CÁO RQ6 (Phần A) — KIỂM CHỨNG ĐỘ TIN CẬY CỦA INTERVENTIONAL SHAPLEY ATTRIBUTION

Tài liệu này được **sinh tự động** từ `rq6_null_condition_fp_rate.csv` và
`rq6_dose_response.csv`, kiểm chứng `experiments/future_rca.py` (module gọi
`gcm.attribute_anomalies` trên mẫu can thiệp `do(x)`, KHÔNG phải Capacity
Agent đang dùng ở các RQ khác — Capacity Agent dùng ngưỡng cố định đơn giản
hơn cho bottleneck ranking).

**Không dùng LLM API** — toàn bộ là tính toán SCM/DoWhy trên 28-node DAG đã
huấn luyện của Sock Shop, không phụ thuộc quota.

---

## A.2 — Tỷ lệ báo động giả ở điều kiện null (không có thay đổi thật)

Chạy `do({INJECTION_NODE} = training mean)` (tức KHÔNG can thiệp thật) **{n_repeats_null} lần độc lập**
(chỉ khác nhau ở lần lấy mẫu Monte Carlo):

* **Tỷ lệ báo động giả cấp node** (bất kỳ node nào, bất kỳ lần lặp nào bị gắn cờ
  "warning"/"critical"): **{node_level_fp_rate:.1f}%**
* **Tỷ lệ báo động giả cấp kịch bản** (ít nhất 1 node bị gắn cờ trong 1 lần chạy):
  **{scenario_level_fp_rate:.1f}%**

Theo từng node:

```
{per_node_fp.to_string()}
```

**Diễn giải**: đây là bằng chứng định lượng cho phát hiện ban đầu (1 lần chạy cũ
từng báo `catalogue_cpu` "critical" dù không có tải thật) — nếu tỷ lệ trên cao,
ngưỡng risk-level hiện tại (`change_pct >= 30%` hoặc `anomaly_score >= 3.0`)
đang quá nhạy so với nhiễu Monte Carlo tự nhiên của việc lấy mẫu, cần hiệu
chỉnh lại trước khi dùng ngưỡng này cho bất kỳ tuyên bố "phát hiện bottleneck"
nào trong bài báo.

---

## A.1 — Dose-response cho `{PRIMARY_TARGET}` (node phản ứng trực tiếp nhất với injection)

Mean $\\pm$ std qua {n_repeats_dose} lần lặp độc lập mỗi mức can thiệp:

```
{dose_summary.to_string(index=False)}
```

* Spearman(mức can thiệp, anomaly\\_score trung bình) = **{spearman_anomaly:.3f}**
  (đơn điệu tăng nghiêm ngặt: **{is_monotonic_anomaly}**)
* Spearman(mức can thiệp, shapley\\_contribution trung bình) = **{spearman_shapley:.3f}**
  (đơn điệu tăng nghiêm ngặt: **{is_monotonic_shapley}**)

**Diễn giải**: nếu cả 2 hệ số Spearman gần 1.0 và đơn điệu tăng đúng, đây là bằng
chứng cơ chế attribution phản ứng hợp lý theo cường độ can thiệp — một thuộc
tính tối thiểu mà bất kỳ cơ chế "giải thích" nào cũng cần có trước khi được tin
tưởng. Nếu KHÔNG đơn điệu hoặc Spearman thấp, đây là bằng chứng đáng lo ngại
về độ tin cậy của con số Shapley hiện tại, cần báo cáo trung thực thay vì bỏ qua.

---

## 📁 Dữ liệu thô
* `data/processed/scm_results/rq6_null_condition_fp_rate.csv` — {len(null_df)} bản ghi ({n_repeats_null} lần lặp x 7 node CPU).
* `data/processed/scm_results/rq6_dose_response.csv` — {len(dose_df)} bản ghi ({len(levels_pct)} mức x {n_repeats_dose} lần lặp x 7 node).
* `data/processed/scm_results/rq6_dose_response_summary.csv` — tổng hợp cho `{PRIMARY_TARGET}`.

## ⏭️ Chưa làm (Part A.3, để sau)
Kiểm tra khớp topology trên Train Ticket (đồ thị đủ lớn để có node THẬT SỰ
không downstream của injection point — Sock Shop quá nhỏ để có phép thử này
sạch) — cần train thêm 1 Global DAG cho Train Ticket, chưa nối vào script này.
"""
    md_path = os.path.join(DOCS_DIR, 'docs/HE_THONG.md (muc 6)')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"\n[OK] Scientific markdown report generated: {md_path}")
    print("=" * 80)
    return null_df, dose_df, dose_summary


if __name__ == '__main__':
    run_rq6_part_a()
