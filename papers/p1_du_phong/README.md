# Bài 1 — Dự phóng tính khả thi khi thêm tính năng hướng người dùng

> **Bài toán:** từ log của hệ chạy bình thường, dự phóng hệ còn chịu được tới **tải nào**
> sau khi thêm một tính năng hướng người dùng.

Loại bài: **dự phóng** (trước khi xảy ra). Khác hẳn [bài 2](../p2_khop_tang/), là định vị
hậu nghiệm.

## Bắt đầu từ đâu

| notebook | nội dung |
|---|---|
| **[`notebooks/01_du_lieu_do_thi_nhan_qua_va_du_doan.ipynb`](notebooks/)** | làm sạch dữ liệu tự đo → dựng đồ thị nhân quả → bốn mô hình dự phóng → kết quả tiến cứu (91 ô) |
| **[`notebooks/02_quy_trinh_he_thong_va_thuc_nghiem.ipynb`](notebooks/)** | quy trình hệ thống chạy từng bước, chỉnh được, E0–E5 (39 ô) |

## Dữ liệu

**Sock Shop tự đo**: đo chuẩn → thêm tính năng → đo lại. `data/raw/SS-*` ở gốc repo,
đi qua **một cổng duy nhất** là `src/scm/measured_data.py` (chỉ đọc, không bao giờ sửa dữ
liệu thô; mỗi lý do loại bỏ đều tra được).

Kết quả đóng băng: `results_frozen/` (symlink tới `data/processed/frozen/`, 48 file có SHA-256).

## Kết quả hiện có

Sai số tuyệt đối trung bình của điểm gãy, bốn mô hình xếp theo thang tăng dần thông tin
(nguồn: `data/processed/frozen/p3_evaluation.csv`):

| tập | n | P1 (k=1) | P2 (c,x) | P3 (k đo được) | **P3+chain** |
|---|---|---|---|---|---|
| **TIẾN CỨU 2** (chính) | 12 | 23,3% | 10,4% | 7,3% | **5,0%** |
| dev | 4 | 37,6% | 8,5% | 4,8% | **4,8%** |
| **khoá** (k≈1, kỳ vọng không đổi) | 4 | 9,6% | −1,6% | −1,6% | **−1,6%** |
| ĐỘC LẬP | 5 | 119,7% | 87,5% | 72,6% | **70,6%** (trung vị 20,6%) |

Thang **đơn điệu** trên mọi tập: mỗi mẩu thông tin thêm vào thì sai số giảm.

**Điểm yếu chưa khắc phục:** tập ĐỘC LẬP có ca thảm hoạ (`expressx1`, 216,6%); n nhỏ;
16 tính năng đều nhỏ và an toàn; sàn nhiễu chưa cho phép tách các mô hình ở chênh lệch nhỏ.
Cả ba việc cần **đo thêm**, không cần phân tích thêm — và cần máy chạy Docker.

## Nhóm script

| nhóm | nội dung |
|---|---|
| `experiments/collect/` | harness đo (`load_sweep_collect.py`), probe `k`, kiểm hợp đồng dữ liệu, **và** hai bộ tải RE2 (`download_*_data.py` — bài 2 cũng dùng) |
| `experiments/feasibility/` | fit chi phí tính năng, đóng băng dự đoán, đánh giá P2/P3, kiểm throttling gate |
| `experiments/edges/` | chọn và kiểm chứng cạnh học của Global Causal DAG (tranh chấp CPU, lan truyền độ trễ), kiểm an toàn OOD |
| `experiments/model_eval/` | bộ đánh giá, cross-validation, nghiêm ngặt thống kê, sinh hình |
| `experiments/parser/` | ParserAgent + sáu guard, cổng phạm vi, hiệu chỉnh conformal, benchmark LLM |
| `experiments/datasets/` | đánh giá trên hệ thứ hai (Train Ticket), chuyển giao độ co giãn LOSO |

## Chạy lại (từ gốc repo)

```bash
python papers/p1_du_phong/experiments/collect/data_contract_check.py
python papers/p1_du_phong/experiments/feasibility/fit_feature_costs.py
python papers/p1_du_phong/experiments/feasibility/freeze_predictions.py
python papers/p1_du_phong/experiments/feasibility/evaluate_frozen.py
python papers/p1_du_phong/experiments/model_eval/make_figures.py
```

## Core dùng chung

| module | vai trò | bài 2 có dùng? |
|---|---|---|
| `src/agents/capacity_agent.py` | huấn luyện Global Causal DAG | **có** (E4b, E8 v3) |
| `src/scm/scm_graph_builder.py`, `scm_edge_selector.py` | bảng mẫu cạnh, chọn cạnh học | có (qua DAG) |
| `src/scm/data_processor.py`, `measured_data.py` | nạp dữ liệu | có |
| `src/scm/queueing_regressor.py`, `deterministic_forward.py`, `node_impact.py` | cơ chế hàng đợi, lan truyền tất định | không |
| `src/scm/feasibility_predictor.py` | **bộ dự phóng — chỉ bài này** | không |
| `src/agents/{feasibility,parser,architecture}_agent.py`, `orchestrator.py` | chuỗi agent — **chỉ bài này** | không |

## Tài liệu

- `docs/HE_THONG.md` — mô tả hệ thống đầy đủ
- `docs/khungtoanhoc.tex` / `.pdf` — khung toán
- `docs/CO_CHE_KHA_THI.md` — cơ chế dung lượng / throttling
- `docs/DATA_FRAMEWORK.md`, `docs/GIAO_THUC_VONG_2.md` — khung dữ liệu, giao thức đo
- `docs/paper_draft.tex` — bản nháp
