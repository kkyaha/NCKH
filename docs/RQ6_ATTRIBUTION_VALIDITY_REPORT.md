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
       0.0       0.13876     0.067917       0.39040     0.095444
      20.0       0.13962     0.057268       0.47674     0.022727
      50.0       0.19028     0.032630       0.68048     0.013095
     150.0       0.16502     0.034487       0.78492     0.026070
     300.0       0.14184     0.051207       0.78510     0.014697
```

* Spearman(mức can thiệp, anomaly\_score trung bình) = **0.600**
  (đơn điệu tăng nghiêm ngặt: **False**)
* Spearman(mức can thiệp, shapley\_contribution trung bình) = **1.000**
  (đơn điệu tăng nghiêm ngặt: **True**)

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

## ⏭️ Đã làm tiếp (xem Part A.3 và Part B bên dưới)
Kiểm tra khớp topology trên Train Ticket, và kiểm tra tách Tier-1/Tier-2 cho
cả 2 hệ thống, đã được thực hiện trong các phần tiếp theo của báo cáo này.


---

# RQ6 Part A.3 — Does Shapley Attribution Respect Graph Topology? (Train Ticket)

Testbed: Train Ticket accurate-path DAG (112 nodes, 150 edges after filtering
to the 28 named services). Injection: `ts-preserve-service` (+150.0%),
10 independent repeats. From this gateway, the fitted DAG has
**17 reachable services** (hops 1-3: `ts-assurance-service`, `ts-basic-service`,
`ts-config-service`, `ts-contacts-service`, `ts-food-map-service`, `ts-food-service`,
`ts-order-other-service`, `ts-order-service`, `ts-price-service`, `ts-route-service`,
`ts-seat-service`, `ts-security-service`, `ts-station-service`, `ts-ticketinfo-service`,
`ts-train-service`, `ts-travel-service`, `ts-user-service`) and
**10 genuinely unreachable services** (no path at all: `ts-admin-basic-info-service`,
`ts-admin-travel-service`, `ts-auth-service`, `ts-consign-price-service`,
`ts-consign-service`, `ts-inside-payment-service`, `ts-payment-service`,
`ts-preserve-other-service`, `ts-travel2-service`, `ts-ui-dashboard`) — Sock Shop's
7-node graph is too small/densely connected for this test.

## Run 1 — before synchronizing the monotonicity constraint with Sock Shop

At this point `CapacityAgent.train_accurate_path()` (used for Train Ticket) did
**not** apply Sock Shop's non-negative-coefficient constraint to CPU/Mem nodes
or to Tier-1 workload-propagation edges (`evaluation_suite.py`'s Sock Shop DAG did).

```
                n_service_repeats  mean_shapley  mean_anomaly  mean_change_pct  flagged_rate_pct
group
injection_self                 10        0.1924        0.5487          44.4780               0.0
reachable                     170        0.1425        0.6781           6.9462               0.0
unreachable                   100        0.1793        0.5640           3.9159               0.0
```

**Precision@17 = 60.0%** (chance baseline 63.0%).

## Fix applied

`src/agents/capacity_agent.py`'s `train_accurate_path()` was updated to apply
the identical `LinearRegression(positive=True)` constraint Sock Shop's DAG
uses, on both CPU/Mem nodes (Tier 2) and non-root workload nodes (Tier 1) —
closing a real gap between the evaluation harness and the actual product
Capacity Agent `orchestrator.py` runs (see Section "Extrapolation-Sign
Failure Mode" in the paper).

## Run 2 — after synchronizing the monotonicity constraint

Same injection, same 10 repeats, identical protocol — only the DAG-fitting
constraint changed.

```
                n_service_repeats  mean_shapley  mean_anomaly  mean_change_pct  flagged_rate_pct
group
injection_self                 10        0.2046        0.5178          41.4960               0.0
reachable                     170        0.1250        0.6583           6.8039               0.0
unreachable                   100        0.1688        0.5436           2.9955               0.0
```

**Precision@17 = 60.6%** (chance baseline 63.0%).

## Conclusion

Precision@17 moved from 60.0% to 60.6% — statistically unchanged — and mean
Shapley contribution is still higher for unreachable services (0.169) than
reachable ones (0.125) in both runs. **This rules out the monotonicity-fit
confound as the explanation** for the negative topology-respect result: the
gap persists even after Train Ticket's DAG is fit with the exact same
constraint as Sock Shop's. The actual cause is still open — leading
candidates are Monte Carlo approximation error at this graph's larger size
(112 fitted nodes vs. Sock Shop's 28) and confounding shared across services
independent of the fitted causal edges.

## Data
* `data/processed/scm_results/rq6_topology_check.csv` — latest run (post-fix), 280 rows (10 repeats x 28 CPU nodes).
* `data/processed/scm_results/rq6_topology_check_summary.csv` — grouped summary (post-fix).

---

# RQ6 Part B — Does Shapley Attribution Correctly Split Tier 1 vs Tier 2?

**Câu hỏi**: paper_draft.tex tuyên bố Shapley attribution có thể phân rã "bao nhiêu % dội vào
Tầng 1 (lan truyền workload liên dịch vụ) vs Tầng 2 (tiêu thụ tài nguyên nội bộ)" cho MỘT
target node — đây là claim khác (và chưa từng được test) so với Part A (dose-response tại chính
node injection, không có Tier-1 ancestor để tách) và Part A.3 (so sánh TỔNG Shapley GIỮA các
service theo reachability, không tách theo TẦNG trong 1 node).

**Thiết kế**: với mỗi target, dựng 2 kịch bản có "nguyên nhân thật" đã biết trước (ground-truth
by construction):
- **Tier-1-driven**: `do(gateway_workload = 2.5×baseline)` thật, giá trị resource của target để
  mô hình tự dự đoán (không can thiệp thủ công) → biết chắc nguyên nhân là lan truyền.
- **Tier-2-driven**: KHÔNG có thay đổi workload thật, nhưng tự perturb giá trị resource của
  chính target lên `+5σ` → biết chắc nguyên nhân là cục bộ.

So `|tier1_contribution|` vs `|tier2_contribution|` (từ `gcm.attribute_anomalies`, vốn đã trả
về dict tách theo từng ancestor — chỉ là code cũ cộng dồn hết vào 1 số) — kiểm tra tầng nào
được dự đoán chiếm ưu thế có khớp với nguyên nhân thật không.

## Kết quả (10 lần lặp × 6 target Sock Shop, 10 lần lặp × 4 target Train Ticket)

| Hệ thống | Tier-2-driven (local) | Tier-1-driven (propagation) | Tổng |
|---|---|---|---|
| Sock Shop | **100.0%** (60/60) | 60.0% (36/60) | 80.0% |
| Train Ticket | **100.0%** (80/80) | **17.5%** (7/40) — *tệ hơn tung đồng xu* | 58.75% |

Chi tiết theo từng target (Sock Shop, tier1_driven): `carts`=100%, `orders`=100%, `user`=100%
(đều là con trực tiếp của `front-end`), nhưng `catalogue`=60%, `payment`=0%, `shipping`=0%
(`payment`/`shipping` có cha Tier-1 là `orders_workload`, tức 2 hop từ front-end — dấu hiệu tín
hiệu lan truyền bị mất qua hop trung gian, không đồng đều giữa các target).

## Kết luận
- **Attribution về nguyên nhân cục bộ (Tier 2): đáng tin cậy hoàn toàn** trên cả 2 hệ thống.
- **Attribution về lan truyền mạng (Tier 1): không đáng tin cậy** — trung bình trên Sock Shop,
  và **tệ hơn ngẫu nhiên** trên Train Ticket.
- Đã cập nhật `paper_draft.tex` (§III-C, RQ2, RQ6, Threats to Validity, Future Work) để phản
  ánh đúng: claim "phân rã theo tầng" giờ được nêu có điều kiện, không còn là năng lực mặc định.

## Dữ liệu thô
* `data/processed/scm_results/rq6_tier_decomposition.csv`
* `data/processed/scm_results/rq6_tier_decomposition_summary.csv`
