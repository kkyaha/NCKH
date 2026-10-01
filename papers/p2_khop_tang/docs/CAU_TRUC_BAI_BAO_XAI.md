# Cấu trúc bài báo — trọng tâm: **định vị đúng tầng**

> Thay cho khung cũ ("interventional Shapley có trung thực không"). Khung cũ đặt
> phương pháp làm bị cáo; khung này đặt **cấu trúc thông tin của mô hình** làm chủ thể,
> và phần Shapley trở thành một hệ quả có thể kiểm.

## Mệnh đề trung tâm

> **Định vị được nguyên nhân trên tầng mà nguyên nhân nằm — và chỉ trên tầng đó.**
>
> Mô hình nhân quả của hệ microservice có hai tầng: **nhu cầu** (`*_workload`) và
> **tài nguyên** (`*_cpu`). Tranh chấp tài nguyên sinh ra cạnh **resource → resource**
> không phải đường lan truyền nhu cầu, nên nó làm nhiễu tầng tài nguyên mà không
> chạm tầng nhu cầu. Hệ quả: một phương pháp định vị chỉ đúng khi **tầng nó soi**
> trùng **tầng nguyên nhân nằm**. Thất bại của quy gán nguyên nhân lan truyền trong
> văn liệu là **lệch tầng**, không phải khuyết điểm phương pháp.

## Tám mảnh bằng chứng

| | bằng chứng | số | nguồn | vòng tròn? |
|---|---|---|---|---|
| **E1** | lỗi **tài nguyên** → định vị trên **tầng tài nguyên** | **98,3% top‑1 · 100% top‑3** (7 ứng viên, ngẫu nhiên 14,3%); tầng nhu cầu **13,3% = ngẫu nhiên** | `rq6_layer_localization_real_faults.py`, 90 run RE2‑SS, đáp án từ tên thư mục | **không** |
| **E2** | can thiệp **nhu cầu** → định vị trên **tầng nhu cầu** | ρ = **+0,790** (TT) / **+0,771** (SS) vs tầng CPU +0,471 / +0,234 | `rq6_localization_decircularized.py`, dữ liệu‑đối‑dữ liệu | **không** (phép C) |
| **E3** | lỗi **delay** → **không** định vị được trên tầng nào | CPU **13,3%** · workload **0,0%** (ngẫu nhiên 14,3%) | cùng E1 | không |
| **E4** | cơ chế: cạnh tranh chấp là CPU→CPU | chiếm **94,4%** thang hiệu dụng; gấp cha nhu cầu **8,3–66,8×** (trung vị 28,2×) | `rq6_hop1_node_diagnostic.csv`, hệ số NNLS — **độc lập Shapley** | không |
| **E5** | quy gán thất bại vì lệch tầng | strict 25,0 → 2,5 → **0,0%** theo hop; kèm cạnh tranh chấp: 40,0 → 92,5 → **90,0%** | `rq6_hop_distance_diagnostic.csv` (`tier1_driven`) | không |
| **E6** | quy mô **khuếch đại** chi phí lệch tầng | tỉ phần nhu cầu **81,0% → 15,7%** (7 → 28 service); tranh chấp 0,3% → 33,9% | `rq6_tier_decomposition_v2_summary.csv` | — |
| **E7** | biết trước khi nào vẫn không đủ | **Pearson 0,957** (p = 1,5e‑5, n = 10, hai hệ); ngưỡng 50% tách hoàn hảo | cùng E6 | nhẹ |
| **E8** | không phải lỗi của một phương pháp | **BARO (FSE '24)** 40–50% ở đúng node Shapley tệ nhất; cả hai 100% ở ca cục bộ | `rq6_baro_comparison.csv` | không |

## Bốn RQ

| RQ | câu hỏi | bằng chứng | điểm yếu |
|---|---|---|---|
| **RQ1** | Độ chính xác định vị có phụ thuộc **tầng mà nguyên nhân nằm** không? | **E1 · E2 · E3** | E1 chấm trên 7 service (không so được trực tiếp với BARO, xem §Threats) |
| **RQ2** | **Vì sao** — cơ chế nào tách hai tầng? | **E4** | — |
| **RQ3** | Điều đó có **giải thích** thất bại của quy gán nhân quả lan truyền không? | **E5 · E8** | cạnh tranh chấp là cạnh học (đã kiểm OOD **trước**) |
| **RQ4** | **Biết trước** được không, và quy mô ảnh hưởng thế nào? | **E7 · E6** | n = 10; **hai điểm** trên trục quy mô |

## Sáu đóng góp

1. **Nguyên lý khớp tầng** — định vị trên tầng nguyên nhân. Bằng chứng không vòng tròn: 90 lần tiêm lỗi thật + hai hệ thống.
2. **Cơ chế** — tranh chấp là resource→resource, chiếm 94,4% thang hiệu dụng; đó là thứ làm nhiễu tầng tài nguyên và không chạm tầng nhu cầu.
3. **Chẩn đoán lại thất bại của quy gán** — là **lệch tầng** (can thiệp tầng nhu cầu, quy gán tầng tài nguyên), không phải khuyết điểm phương pháp. BARO thất bại ở cùng chỗ nên không quy được cho Shapley.
4. **Một tầng thứ ba mà mô hình không biểu diễn** — lỗi `delay` nằm trên đường gọi; không định vị được từ cả hai tầng (13,3% / 0,0%). Giới hạn **biểu diễn**, không phải thống kê.
5. **Tiêu chí khả nhận** — tỉ phần nhu cầu của cơ chế đã fit dự đoán độ chính xác quy gán (Pearson 0,957), tính được **không cần đáp án** nên dùng được lúc triển khai.
6. **Bài học đánh giá** — một bộ phát hiện phải có **đối chứng dương**. Ngưỡng gán cứng `z ≥ 1,0` không đạt được (z max 0,448) khiến `flagged_rate = 0` ở mọi nhóm và tuyên bố "zero false positive" **rỗng**; hiệu chỉnh từ nửa null giữ lại (0,20/0,40) cho 0% FP **và** 100% phát hiện.

## Mục nào lấy từ đâu

| mục | nội dung | nguồn đã có |
|---|---|---|
| §Motivation | hai tầng, tranh chấp là gì, vì sao lệch tầng đắt | — |
| §RQ1a | lỗi tài nguyên → tầng tài nguyên | **mới hôm nay** — `rq6_layer_localization_real_faults.py` |
| §RQ1b | can thiệp nhu cầu → tầng nhu cầu, ba phép chấm | **mới hôm nay** — `rq6_localization_decircularized.py` |
| §RQ1c | delay: tầng thứ ba | **mới hôm nay** — cùng §RQ1a |
| §RQ2 | soi cơ chế NNLS | `rq6_hop1_node_diagnostic.py` |
| §RQ3a | tier‑1/tier‑2, hop‑distance, bug phân loại hai chiều | `rq6_tier_decomposition_v2.py`, `rq6_hop_distance_diagnostic.py` |
| §RQ3b | BARO, và BARO trên dữ liệu tiêm lỗi thật | `rq6_baro_comparison.py`, `baro_on_real_rcaeval_scenarios.py` |
| §RQ4a | tiêu chí khả nhận | `rq6_tier_decomposition_v2_summary.csv` |
| §RQ4b | chuyển pha theo quy mô | cùng trên |
| §Evaluation protocol | đối chứng dương, hiệu chỉnh ngưỡng | **mới hôm nay** — `recalibrate_risk_threshold.py` |
| §Sanity checks | null FP, dose‑response đơn điệu | `rq6_attribution_validity.py` (đã chạy lại, `--calibrated`) |

## Hạ cấp — không bỏ, chuyển vai

| trước là gì | giờ là gì |
|---|---|
| "interventional Shapley có trung thực không" — mệnh đề bài | **RQ3**: một hệ quả kiểm được của nguyên lý khớp tầng |
| Precision@17 (dao động 59,4 / 60,0 / 60,6 / 75,9% quanh ngẫu nhiên 63,0%) | **Threats** — bốn lần chạy, ±15 điểm nhiễu với 10 repeat; **không** dùng làm bằng chứng chính |
| topology respect | ghi đúng chiều: `change_pct` tôn trọng topology (39,1/5,5/3,0), Shapley thì không (0,159/0,159/0,173) — và đó là **bằng chứng cho RQ1** |

## Threats to Validity — viết trước, đừng chờ reviewer

1. **E1 chấm trên 7 service ứng dụng** (ngẫu nhiên 14,3%); BARO trong §RQ3b chấm trên ~14 service (ngẫu nhiên ~7%) với thống kê khác. **Không đặt 98,3% cạnh 26,7%** mà chưa khớp tập ứng viên. So sánh khớp là thí nghiệm riêng, chưa làm.
2. **E1 là định vị lỗi đã xảy ra**, không phải dự phóng. Nó khử vòng tròn cho câu hỏi **tầng**, không chứng minh gì về dự đoán tính năng mới.
3. **E6/E7: hai điểm trên trục quy mô, n = 10 node.** Ngưỡng 50% fit trong mẫu → báo là **mô tả**, chưa phải ngưỡng vận hành.
4. **E2 phép (A) có vòng tròn cấu trúc** — cạnh workload→workload dựng từ đồ thị gọi, mà đáp án cũng là đồ thị gọi. Khử bằng phép (B) 96,5% → 68,0% và phép (C) dữ liệu‑đối‑dữ liệu. **Ưu thế thu hẹp nhưng không mất** — phải báo cả ba.
5. **Cạnh tranh chấp là cạnh học.** Đã kiểm an toàn OOD **trước** khi có các kết quả này (`backpressure_edge_ood_safety.csv`), và độ lớn xác nhận độc lập bằng NNLS.
6. **Đồ thị gọi khai quá rộng**: Train Ticket có 17 service reachable theo đồ thị nhưng chỉ 10 theo dữ liệu quan sát, trùng nhau 9. Một phần thất bại quy gán là do **đồ thị đầu vào**, không do phương pháp.

## Ba việc làm bài mạnh hơn

| | việc | củng cố | chi phí |
|---|---|---|---|
| 1 | **so BARO với tập ứng viên khớp** (7 service, cùng cửa sổ, cùng đáp án) | Threat #1 → thành so sánh hợp lệ; nếu CPU‑layer vẫn thắng thì đó là một đóng góp nữa | thấp — dữ liệu đã có |
| 2 | **hệ thứ ba**: nối Online Boutique vào `CapacityAgent` (`RE2-OB` có 90 kịch bản, ~11 service) | E6 từ 2 → 3 điểm | trung — cần nhánh hệ thống mới |
| 3 | kiểm ngưỡng 50% trên tập giữ lại | E7 thành ngưỡng vận hành | thấp |

Việc 1 đáng nhất: rẻ, và nó biến điểm yếu lớn nhất của E1 thành một kết quả.
