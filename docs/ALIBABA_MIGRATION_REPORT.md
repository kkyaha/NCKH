# Chuyển đánh giá forecasting sang Alibaba v2021 — báo cáo

> Sinh trong phiên 2026-09-14. Mọi số liệu dưới đây đến từ các script trong
> `experiments/alibaba_*.py`, dữ liệu thô lưu ở
> `data/processed/scm_results/alibaba_*.csv`.

## 1. Vì sao phải đổi dataset

RCAEval (RE2-SS) **không chứa tín hiệu** workload → resource:

| | RCAEval (gộp 90 run) | Alibaba v2021 (2,5h) |
|---|---|---|
| $R^2$ trung vị CPU | **0,0034** | **0,2209** |
| $R^2$ p75 CPU | — | 0,5908 |
| % service $R^2 > 0{,}3$ | — | **43,7%** |
| $R^2$ trung vị Memory | 0,0034 | 0,0256 |
| Biên độ tải tương đối | 1,70 | 0,62 |

Tín hiệu cao hơn **65×** dù biên độ tải chỉ bằng 1/3.

**Nguyên nhân cấu trúc:** benchmark RCA cố ý giữ giai đoạn normal ở trạng thái
dừng để fault nổi bật. Chính sự ổn định đó xoá mất tín hiệu tải mà capacity
forecasting cần. Hai yêu cầu xung đột trực tiếp — nên không benchmark nào phục
vụ được cả hai.

Script: `experiments/alibaba_signal_check.py`

## 2. Hệ quả: chuỗi kết quả âm trước đó có MỘT nguyên nhân gốc

Mọi kết quả âm trong phiên đều truy về $R^2 = 0{,}003$:

- RQ2: `SCM_Deployed` trùng khít `LinearReg` trên 86/105 cặp
- $R^2$ âm trên 16/21 cặp
- RQ8: không tìm thấy confounder (co-location p=0,55; tải toàn cục 1,2%)
- RQ9: hiệu chỉnh backdoor không cải thiện (thắng 51,2% = đồng xu)
- RQ11: phản thực quy về giữ phần dư (thắng 47,5%)

Không phải sáu thất bại. Là **một chẩn đoán**.

## 3. Chốt an toàn — đánh giá cũ KHÔNG THỂ sai

Memory đạt MAPE 2–4% và trông như thành công, trong khi $R^2 = 0{,}003$ chứng
minh memory gần như không đổi — nên **một hằng số** cũng cho 2–4%.

Giao thức mới có sáu chốt, mỗi chốt đều có thể cho kết quả âm:

| Chốt | Bắt được gì |
|---|---|
| 0. Signal gate ($R^2$ toàn bộ dữ liệu) | Phát hiện vấn đề RCAEval |
| 1. **Negative control** (hoán vị workload) | Xác nhận metric đo đúng thứ |
| 2. **Baseline hằng số** | Phơi artefact memory ngay |
| 3. Mẫu train khớp nhau | Bỏ confound cỡ mẫu |
| 4. **Skill score** thay MAPE | "Không tín hiệu" tự hiện ra |
| 5. Tách in-dist / OOD | Phân biệt "không học được" với "không ngoại suy được" |
| 6. Phân phối per-service | Không gộp che khuất |

## 4. Kết quả Tier-2 (workload → resource)

Negative control **PASS**: skill ≈ 0 hoặc âm trên mọi mô hình khi hoán vị.

### CPU — mô hình thắng thật

| model | mode | skill_med | %>0 | MAPE_med |
|---|---|---|---|---|
| constant | indist | 0,0000 | — | 4,45 |
| LinearReg | indist | **+0,2033** | **82,1%** | 3,08 |
| NNLS_deployed | indist | **+0,1723** | **81,8%** | 3,09 |
| GradBoost | indist | +0,1088 | 56,4% | 3,26 |
| constant | ood | 0,0000 | — | 5,33 |
| LinearReg | ood | **+0,1678** | **63,0%** | 4,11 |
| NNLS_deployed | ood | **+0,1355** | **65,9%** | 4,07 |

n = 1.292 service.

### Memory — artefact tái diễn và bị bắt

| model | mode | skill_med | MAPE_med |
|---|---|---|---|
| constant | indist | 0,0000 | **0,13** |
| LinearReg | indist | +0,0076 | 0,12 |
| NNLS_deployed | ood | **−0,0000** | 0,14 |

**MAPE 0,13 trông xuất sắc, skill ≈ 0.** Tái hiện trên bộ dữ liệu thứ hai →
**không phải đặc thù RCAEval**, mà là tính chất của memory như một metric.

Script: `experiments/alibaba_forecast_eval.py`

## 5. Kết quả Tier-1 (lan truyền theo đồ thị)

Topology dựng từ `MSCallGraph` (`um`→`dm`, chỉ `rpctype ∈ {rpc, http}`).
106 edge ≥ 5.000 lời gọi; **27 edge** có đủ dữ liệu workload trùng khớp.

Negative control **PASS**: `uniform_delta` sụp về −1,10 / −2,30.

| model | mode | skill_med | %>0 |
|---|---|---|---|
| constant | indist | 0,0000 | — |
| uniform_delta | indist | +0,4522 | 77,8% |
| **graph_NNLS** | indist | **+0,5408** | **92,6%** |
| uniform_delta | ood | +0,5291 | 70,4% |
| **graph_NNLS** | ood | **+0,5161** | **81,5%** |

### Đồ thị vs delta đều

| | n | graph thắng | skill graph | skill uniform |
|---|---|---|---|---|
| in-dist | 27 | **21/27 (77,8%)** | +0,5408 | +0,4522 |
| ood | 27 | 15/27 (55,6%) | +0,5161 | +0,5291 |

**Đồ thị thắng khi nội suy, tương đương khi ngoại suy.** Nhất quán với việc cơ
chế gần tuyến tính: một đường tuyến tính qua gốc xấp xỉ chính là tỉ lệ thuận.

So với RQ4 gốc (Sock Shop, n=6, bootstrap CI chứa 0, không có negative
control) thì đây là bằng chứng vững hơn hẳn — và nó **sửa** kết luận cũ.

Script: `experiments/alibaba_tier1_propagation.py`

## 6. Hạn chế

- Chỉ dùng 2,5 / 12 giờ dữ liệu; tải thêm chunk sẽ mở rộng
- Chỉ 27 / 106 edge có đủ workload trùng khớp (MCR và CallGraph không phủ cùng
  tập service)
- **Tên service bị hash** → nửa parser/guardrail **không** chạy được ở đây,
  phải giữ trên Sock Shop / Train Ticket
- 12 giờ chưa đủ một chu kỳ ngày–đêm đầy đủ
- Lấy mẫu 60 giây, thô hơn RCAEval

## 7. Cấu trúc đánh giá đề xuất cho paper

| Nửa | Dataset | Lý do |
|---|---|---|
| Parser + guardrail | Sock Shop, Train Ticket | Cần tên service có nghĩa |
| Forecasting + lan truyền | **Alibaba v2021** | Cần biên độ tải |
| Kiểm định can thiệp | RCAEval fault injection | Đúng mục đích của nó |
| **Chẩn đoán benchmark** | cả ba | Đóng góp mới |

Phải giải thích tường minh vì sao tách — chính đó là phát hiện về khoảng trống
dữ liệu: **không bộ nào có cả tên service có nghĩa lẫn biên độ tải rộng.**

## 8. Tái lập

```bash
# tải (145/25/12 chunk; script dùng chunk đầu)
#   base: https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces
#   MSRTQps_{0..24}.tar.gz  MSResource_{0..11}.tar.gz  MSCallGraph_{0..144}.tar.gz
export ALIBABA_DIR=<thu muc chua cac file .csv da giai nen>
python experiments/alibaba_signal_check.py        # gate tin hieu
python experiments/alibaba_forecast_eval.py       # Tier-2 + negative control
python experiments/alibaba_tier1_propagation.py   # Tier-1 + delta deu
```

Tên cột thực tế khác tài liệu: cột đầu là index không tên;
`instance_cpu_usage` / `instance_memory_usage`; bảng MCR dùng `metric` (số ít).
