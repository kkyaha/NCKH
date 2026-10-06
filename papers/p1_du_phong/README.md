# Bài 1 — Dự phóng tính khả thi khi thêm tính năng hướng người dùng

> **Bài toán:** từ log của hệ chạy bình thường, dự phóng hệ còn chịu được tới **tải nào**
> sau khi thêm một tính năng hướng người dùng.

Loại bài: **dự phóng** (trước khi xảy ra). Khác hẳn [bài 2](../p2_khop_tang/), là định vị
hậu nghiệm.

## Bắt đầu từ đâu

| notebook | nội dung |
|---|---|
| **[`notebooks/01_du_lieu_do_thi_nhan_qua_va_du_doan.ipynb`](notebooks/)** | làm sạch dữ liệu tự đo → dựng đồ thị nhân quả → bốn mô hình dự phóng → kết quả tiến cứu → **B8: đánh giá mức phán quyết**, **B1.5: các tính năng thêm vào + sàn phân giải** (98 ô, chạy sạch 0 lỗi, có output) |
| **[`notebooks/02_quy_trinh_he_thong_va_thuc_nghiem.ipynb`](notebooks/)** | quy trình hệ thống chạy từng bước, chỉnh được, E0–E5, **6b: sàn nhiễu thật**, **6c: Δ do Parser dự đoán → phán quyết** (46 ô, chạy sạch 0 lỗi, có output) |

> **Đọc hai mục này trước khi viện dẫn bất kỳ con số nào:**
> [nb02 §6b](notebooks/02_quy_trinh_he_thong_va_thuc_nghiem.ipynb) — năng lực *cơ sở* của cùng một
> hệ thống trải **120–240 req/s** qua 16 phiên đo, và sàn nhiễu là 45,2% (qua phiên) chứ không phải
> 21,4% (trong một phiên). [nb01 §B8](notebooks/01_du_lieu_do_thi_nhan_qua_va_du_doan.ipynb) — ở mức
> phán quyết, bậc thang bốn mức sập xuống **hai** nhóm: chỉ P1 → P2 có bằng chứng (9–0, p = 0,004);
> `P1_ctrl` trùng P0 trên **288/288** quyết định và P3 không thắng P2 (p = 1,0).
> [nb01 §B1.5](notebooks/01_du_lieu_do_thi_nhan_qua_va_du_doan.ipynb) — **14/18 tính năng** nằm trong
> nhóm có khai báo giống hệt nhau, nên mô hình dựa trên khai báo *buộc* phải dự đoán giống nhau cho
> chúng; năng lực thật lệch tới **1,79 lần** bên trong một nhóm (`quickadd` 70 vs `wishlist` 125 req/s).
> Probe cũng không cứu: `k` đo được khác **ngược chiều** (nhiều lời gọi hơn lại đi với năng lực cao hơn),
> nên mọi mô hình đơn điệu theo số lời gọi — P2 hay P3 — xếp sai thứ tự hai cặp chặt nhất.
> [nb02 §6c](notebooks/02_quy_trinh_he_thong_va_thuc_nghiem.ipynb) — truyền **phân bố sai số thật**
> của Parser (600 lượt) vào phán quyết: bản có guard mất **0,63 điểm** độ đúng so với Δ hoàn hảo và
> chỉ 0,16% ca nguy hiểm, trong khi cùng LLM **không guard** cho MAE **844 điểm %**, 91% ảo tên dịch
> vụ. Guard là phần đóng góp, không phải LLM; LLM hơn luật thuần 0,52 điểm độ đúng nhưng giảm ca
> nguy hiểm **4 lần**. Δ vẫn chưa được thẩm định so với thực tế (hạn chế M5).
>
> Cả ba notebook chạy được bằng `nbclient` không cần Jupyter:
> `python3 -c "import nbformat;from nbclient import NotebookClient;nb=nbformat.read(F,as_version=4);NotebookClient(nb,kernel_name='python3',resources={'metadata':{'path':D}}).execute()"`
> Ô gọi LLM thật trong nb02 tự bỏ qua nếu thiếu `python-dotenv` / `GOOGLE_API_KEY`; phần còn lại không cần LLM.

## Phạm vi — giả định một gateway (G0)

Toàn bộ P0–P3 được xây **theo công thức** cho hệ có đúng một điểm vào (`GATEWAY = 'front-end'`,
`src/scm/feasibility_predictor.py:44,111`) — chưa từng chạy trên hệ nào khác. ParserAgent và
CapacityAgent (các lớp phía trên) đã tổng quát hoá sang đa gateway, kể cả một lỗi thật đã bắt
và sửa; giới hạn chỉ nằm ở tầng dự đoán khả thi. Chi tiết: nb01 mục C1 (bảng G0–G4), sơ đồ
[`01_kien_truc_duong_ong.puml`](docs/figures/puml/README.md).

## Hai ngăn xếp — đọc trước khi trích số

Bài 1 có **hai** mô hình, đừng lẫn:

| | **hệ A** `CapacityAgent` | **hệ B** `FeasibilityPredictor` (P0–P3) |
|---|---|---|
| cấu trúc | DAG 28 node, 32 cạnh (sơ đồ `03a`) | nan hoa 3 bước, **0 cạnh service→service** (sơ đồ `03b`) |
| dùng cho | RQ1–RQ4 | RQ5 — **mọi con số công bố** |
| chạy thật `assess()` | **luôn** đi qua | chỉ khi có `L_peak` |
| `freeze_predictions.py` | **0 tham chiếu** | chỉ dùng cái này |

Hai hệ tương đương **ở mức `u`** (lệch ≤ 0,30%, **0,00%** tại nút nghẽn) nhưng **không** ở mức
workload (`user` 13,25%) — đo bằng
[`feasibility/doi_chieu_hai_he.py`](experiments/feasibility/doi_chieu_hai_he.py).

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
