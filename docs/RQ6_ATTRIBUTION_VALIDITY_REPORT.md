# 📊 BÁO CÁO RQ6 (Phần A) — KIỂM CHỨNG ĐỘ TIN CẬY CỦA INTERVENTIONAL SHAPLEY ATTRIBUTION

Tài liệu này được **sinh tự động** từ `rq6_null_condition_fp_rate.csv` và
`rq6_dose_response.csv`, kiểm chứng `experiments/future_rca.py` (module gọi
`gcm.attribute_anomalies` trên mẫu can thiệp `do(x)`, KHÔNG phải Capacity
Agent đang dùng ở các RQ khác — Capacity Agent dùng ngưỡng cố định đơn giản
hơn cho bottleneck ranking).

**Không dùng LLM API** — toàn bộ là tính toán SCM/DoWhy trên 28-node DAG đã
huấn luyện của Sock Shop, không phụ thuộc quota.

---

## A.2 — Tỷ lệ báo động giả ở điều kiện null (không có thay đổi thật)

Chạy `do(front-end_workload = training mean)` (tức KHÔNG can thiệp thật) **20 lần độc lập**
(chỉ khác nhau ở lần lấy mẫu Monte Carlo):

* **Tỷ lệ báo động giả cấp node** (bất kỳ node nào, bất kỳ lần lặp nào bị gắn cờ
  "warning"/"critical"): **0.0%**
* **Tỷ lệ báo động giả cấp kịch bản** (ít nhất 1 node bị gắn cờ trong 1 lần chạy):
  **0.0%**

Theo từng node:

```
node
carts_cpu        0.0
catalogue_cpu    0.0
front-end_cpu    0.0
orders_cpu       0.0
payment_cpu      0.0
shipping_cpu     0.0
user_cpu         0.0
```

**Diễn giải**: đây là bằng chứng định lượng cho phát hiện ban đầu (1 lần chạy cũ
từng báo `catalogue_cpu` "critical" dù không có tải thật) — nếu tỷ lệ trên cao,
ngưỡng risk-level hiện tại (`change_pct >= 30%` hoặc `anomaly_score >= 3.0`)
đang quá nhạy so với nhiễu Monte Carlo tự nhiên của việc lấy mẫu, cần hiệu
chỉnh lại trước khi dùng ngưỡng này cho bất kỳ tuyên bố "phát hiện bottleneck"
nào trong bài báo.

---

## A.1 — Dose-response cho `front-end_cpu` (node phản ứng trực tiếp nhất với injection)

Mean $\pm$ std qua 5 lần lặp độc lập mỗi mức can thiệp:

```
 level_pct  anomaly_mean  anomaly_std  shapley_mean  shapley_std
       0.0       0.11816     0.088509       0.39202     0.059813
      20.0       0.14632     0.022921       0.48612     0.013608
      50.0       0.17282     0.048812       0.69586     0.012136
     150.0       0.12294     0.039282       0.80180     0.015491
     300.0       0.10460     0.070239       0.80022     0.030329
```

* Spearman(mức can thiệp, anomaly\_score trung bình) = **-0.300**
  (đơn điệu tăng nghiêm ngặt: **False**)
* Spearman(mức can thiệp, shapley\_contribution trung bình) = **0.900**
  (đơn điệu tăng nghiêm ngặt: **False**)

**Diễn giải**: nếu cả 2 hệ số Spearman gần 1.0 và đơn điệu tăng đúng, đây là bằng
chứng cơ chế attribution phản ứng hợp lý theo cường độ can thiệp — một thuộc
tính tối thiểu mà bất kỳ cơ chế "giải thích" nào cũng cần có trước khi được tin
tưởng. Nếu KHÔNG đơn điệu hoặc Spearman thấp, đây là bằng chứng đáng lo ngại
về độ tin cậy của con số Shapley hiện tại, cần báo cáo trung thực thay vì bỏ qua.

---

## 📁 Dữ liệu thô
* `data/processed/scm_results/rq6_null_condition_fp_rate.csv` — 140 bản ghi (20 lần lặp x 7 node CPU).
* `data/processed/scm_results/rq6_dose_response.csv` — 175 bản ghi (5 mức x 5 lần lặp x 7 node).
* `data/processed/scm_results/rq6_dose_response_summary.csv` — tổng hợp cho `front-end_cpu`.

## ⏭️ Chưa làm (Part A.3, để sau)
Kiểm tra khớp topology trên Train Ticket (đồ thị đủ lớn để có node THẬT SỰ
không downstream của injection point — Sock Shop quá nhỏ để có phép thử này
sạch) — cần train thêm 1 Global DAG cho Train Ticket, chưa nối vào script này.


---

# RQ6 Part A.3 — Does Shapley Attribution Respect Graph Topology? (Train Ticket)

Testbed: Train Ticket accurate-path DAG (112 nodes, 150 edges after filtering
to the 28 named services). Injection: `ts-preserve-service` (+150.0%),
10 independent repeats. From this gateway, the fitted DAG has
**17 reachable services** (hops 1-3: ['ts-assurance-service', 'ts-basic-service', 'ts-config-service', 'ts-contacts-service', 'ts-food-map-service', 'ts-food-service', 'ts-order-other-service', 'ts-order-service', 'ts-price-service', 'ts-route-service', 'ts-seat-service', 'ts-security-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-train-service', 'ts-travel-service', 'ts-user-service']) and
**10 genuinely unreachable services** (no path at all:
['ts-admin-basic-info-service', 'ts-admin-travel-service', 'ts-auth-service', 'ts-consign-price-service', 'ts-consign-service', 'ts-inside-payment-service', 'ts-payment-service', 'ts-preserve-other-service', 'ts-travel2-service', 'ts-ui-dashboard']) — Sock Shop's 7-node graph is too small/densely connected for
this test, which is why it was deferred there.

## Summary by topological group

```
                n_service_repeats  mean_shapley  mean_anomaly  mean_change_pct  flagged_rate_pct
group                                                                                           
injection_self                 10        0.1614        0.4639          37.0190               0.0
reachable                     170        0.2416        0.3797           7.5023               0.0
unreachable                   100        0.1917        0.4703           3.2738               0.0
```

## Precision@17

Of the top-17 Shapley-ranked service CPU nodes per repeat (excluding
`ts-preserve-service`'s own CPU), the fraction that are genuinely reachable
from the injection point: **75.9%** (mean over 10
repeats), against a **63.0%**
chance baseline if Shapley ranked services at random with respect to
topology.

## Data
* `data/processed/scm_results/rq6_topology_check.csv` — 280 rows (10 repeats x 28 CPU nodes).
* `data/processed/scm_results/rq6_topology_check_summary.csv` — grouped summary.
