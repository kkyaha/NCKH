# DỰ ÁN DỰ BÁO TẢI VÀ RỦI RO NĂNG LƯỢNG CHO TÍNH NĂNG MỚI BẰNG MÔ HÌNH NHÂN QUẢ SCM
## Structural Causal Models (SCM) & Do-Calculus for Zero-Shot Feasibility Forecasting

Hệ thống đánh giá **tính khả thi** khi có một **yêu cầu tính năng mới** (do khách hàng/PM đưa ra,
bằng ngôn ngữ tự nhiên) trong bảo trì phần mềm microservices — dự báo tài nguyên (CPU, Memory,
Socket, Latency) và cảnh báo rủi ro quá tải *trước khi* triển khai, bằng Mô Hình Nhân Quả Cấu Trúc
(SCM) và phép toán can thiệp $do(Workload)$, kết hợp Multi-Agent System (Parser Agent + Capacity
Agent) để dịch yêu cầu ngôn ngữ tự nhiên thành can thiệp $do(x)$.

**Phạm vi**: hệ thống nhắm tới các yêu cầu xấp xỉ được bằng một archetype hiệu chỉnh đã có trong
taxonomy của hệ đang chạy (xem `docs/paper_draft.tex` mục Scope), không phải dự đoán mở cho lĩnh
vực ứng dụng hoàn toàn chưa biết.

---

## 📂 CẤU TRÚC REPO

Repo tách rõ **hệ thống lõi** (sản phẩm thật, mô tả trong Section III của bài báo) khỏi
**hạ tầng thực nghiệm** (chỉ dùng để sinh số liệu cho các RQ, không phải một phần của sản phẩm):

```
NCKH/
├── src/                            # ============ HỆ THỐNG LÕI (the actual product) ============
│   ├── agents/
│   │   ├── orchestrator.py             # Điều phối ReAct đa tác tử — ĐIỂM VÀO của hệ thống
│   │   ├── parser_agent.py             # NL requirement -> do(x): 2-tier fast-path + 5-Guard LLM
│   │   ├── capacity_agent.py           # SCM tool: Fast-path (bivariate) + Accurate-path (Global DAG)
│   │   └── architecture_agent.py       # Topology mapping / blast-radius (graph BFS)
│   ├── scm/                            # Thư viện lõi dùng trực tiếp bởi agents
│   │   ├── data_processor.py               # Load/split dữ liệu, danh mục METRICS/SERVICES
│   │   ├── request_router.py               # CALL_CHAINS: bảng hiệu chỉnh + phân loại request (SockShop)
│   │   └── trainticket_router.py           # Tương đương request_router cho Train Ticket (chưa nối vào orchestrator)
│   └── graph/
│       ├── sockshop_agent_graph.json       # Topology 7 service SockShop
│       ├── trainticket_agent_graph.json    # Topology 28 service Train Ticket
│       ├── extract_graph.py                # Trích xuất topology SockShop
│       └── extract_trainticket_graph.py    # Trích xuất topology Train Ticket
│
├── experiments/                    # ======= HẠ TẦNG THỰC NGHIỆM (sinh số liệu cho RQ1-RQ6) =======
│   ├── evaluation_suite.py             # RQ1, RQ2, RQ4: accuracy, model comparison, propagation value
│   ├── model_comparison.py             # RQ2: SCM vs LinearReg/GradBoost/GaussianProcess
│   ├── compare_with_ground_truth.py    # RQ1: ground-truth direct match (bằng chứng mạnh nhất, không bucket)
│   ├── parser_benchmark_suite.py       # RQ3: ablation LLM parser (LLM thật)
│   ├── future_rca.py                   # Pre-mortem RCA (OOD guard + Shapley) — demo/eval riêng, chưa nối vào orchestrator
│   ├── trainticket_evaluation.py       # RQ6: cùng methodology RQ1/RQ2 áp cho Train Ticket
│   ├── download_trainticket_data.py    # Tải dữ liệu Train Ticket thật từ Hugging Face (RCAEval)
│   └── scm_pipeline.py                 # Helper (hàm `mape`) dùng bởi evaluation_suite.py
│
├── run_all_experiments.py          # Entry point: chạy toàn bộ experiments/* cho SockShop, tuần tự
├── data/
│   ├── raw/                            # RE2-SS (SockShop) + trainticket/ (tải qua download_trainticket_data.py)
│   ├── benchmark/                      # 50 prompt cho RQ3 (parser_benchmark_prompts.json)
│   └── processed/scm_results/          # Toàn bộ output CSV — xem bảng bên dưới
├── docs/                            # Báo cáo mỗi RQ (sinh tự động từ CSV) + bản thảo bài báo
└── tests/                           # Unit test (smoke test, không thay thế đánh giá khoa học)
```

**Vì sao tách vậy**: `src/` là những gì một kỹ sư triển khai thật sẽ dùng (gọi `orchestrator.py`
với 1 câu yêu cầu, nhận về `FeasibilityReport`). `experiments/` chỉ tồn tại để trả lời các câu hỏi
nghiên cứu (RQ1-RQ6) — không được `src/agents/*.py` import, và không cần thiết nếu chỉ muốn chạy
hệ thống trên production.

---

## ⚡ CHẠY

```bash
# --- Chạy HỆ THỐNG LÕI (1 yêu cầu tính năng mới -> FeasibilityReport) ---
# orchestrator.py là 1 LangGraph StateGraph đã compile sẵn thành `feasibility_analyzer`;
# chạy trực tiếp file sẽ dùng requirement mẫu trong khối if __name__ == "__main__":
python src/agents/orchestrator.py
# Hoặc gọi với requirement khác:
python -c "
from src.agents.orchestrator import feasibility_analyzer
result = feasibility_analyzer.invoke({'input_requirement': 'Áp mã giảm giá 20% khi thanh toán'})
print(result['feasibility_report'])
"

# --- Chạy TOÀN BỘ THỰC NGHIỆM (sinh số liệu cho RQ1/RQ2/RQ4 trên SockShop) ---
# Cần .env chứa GOOGLE_API_KEY để chạy đầy đủ RQ3 (LLM Parser thật).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python run_all_experiments.py

# Ép RQ3 dùng bộ giả lập offline (CI smoke-test, KHÔNG dùng để công bố):
python -c "import sys; sys.argv=['x','--offline']; from experiments.parser_benchmark_suite import run_parser_benchmark; run_parser_benchmark()"

# --- RQ6: đánh giá trên hệ thống thứ hai (Train Ticket, 28 microservices) ---
python experiments/download_trainticket_data.py     # 30 kịch bản (thêm --all cho 90)
python experiments/trainticket_evaluation.py         # chạy full pipeline tương đương SockShop
```

`run_all_experiments.py` chạy tuần tự các bước cho **SockShop** — xem docstring đầu file để biết
chi tiết. Các bước tạo ra **nhiều nguồn bằng chứng độc lập** về độ chính xác SCM (bucket-average
OOD, kiểm định thống kê theo %, so khớp điểm-thật không bucket trên toàn bộ 90 run, và giá trị
của lan truyền Tầng 1 vs naive) — nên đối chiếu tất cả khi viết kết luận.

---

## 📊 KẾT QUẢ THỰC NGHIỆM (`data/processed/scm_results/`)

| File | Nội dung | Sinh bởi |
|---|---|---|
| `test_f1_rmse_evaluation.csv` | Độ chính xác SCM (RMSE/MAE/MAPE/SMAPE/R²) theo service×metric, cả bản bucket-average và full-resolution (`*_full_res`) | `evaluation_suite.run_f1_rmse_benchmark` |
| `05_model_comparison.csv` | Đối chiếu SCM vs LinearReg/GradBoost/GaussianProcess, cả bucket và full-resolution | `model_comparison.py` |
| `p_value_statistical_test.csv` | Wilcoxon/Friedman theo TỪNG metric, cả `mape_pct` và `mape_full_res_pct` (không gộp đơn vị RMSE khác nhau) | `evaluation_suite.run_statistical_significance` |
| `ground_truth_direct_match.csv` | ~400k điểm so khớp SCM với giá trị đo thật trên toàn bộ 90 (scenario×run) của RE2-SS, train/test tách theo thời gian trong từng run, không bucket | `compare_with_ground_truth.py` |
| `ground_truth_direct_match_summary.csv` | Tổng hợp sai số theo (service, metric) từ file trên | như trên |
| `rq4_propagation_value_test.csv` | RQ4: SCM lan truyền Tầng 1 (Workload→Workload theo topology thật) vs giả định "delta đều" | `evaluation_suite.run_rq4_propagation_value_test` |
| `test_new_features_simulation.csv` | Mô phỏng do(workload) cho các tính năng mới + Flash Sale/Black Friday trên Global 28-Node DAG. **`risk_status` là heuristic tự đặt**, có cột `extrapolation_suspect` đánh dấu các điểm ngoại suy bất thường | `evaluation_suite.test_new_features_simulation` |
| `14_node_causal_propagation.csv` | Lan truyền do(front-end_workload=+50%) qua 2 tầng Workload→CPU | `evaluation_suite.test_14_node_causal_graph` |
| `parser_ablation_benchmark.csv`, `parser_ablation_by_category.csv` | RQ3: ablation 4 cấu hình parser × 50 prompt × N lần lặp, dùng LLM thật (cột `llm_backend`) | `parser_benchmark_suite.py` |
| `future_rca_results.csv` | Pre-mortem RCA (OOD guard + Shapley attribution) | `future_rca.py` |
| `trainticket_f1_rmse_evaluation.csv`, `trainticket_model_comparison.csv`, `trainticket_p_value_statistical_test.csv`, `trainticket_ground_truth_direct_match_summary.csv` | RQ6: cùng pipeline đánh giá như SockShop nhưng chạy trên Train Ticket (28 service) | `experiments/trainticket_evaluation.py` |
| `01_system_telemetry_train_test.csv`, `02_system_test_cases_catalog.csv` | Dữ liệu viễn trắc gộp + danh mục test case | `src/scm/data_processor.py` |

⚠️ Trước khi trích dẫn `parser_ablation_benchmark.csv`, kiểm tra cột `llm_backend`: nếu bắt đầu
bằng `SYNTHETIC_` là bộ giả lập offline (chỉ dùng CI, không dùng làm bằng chứng khoa học).

---

## 📖 TÀI LIỆU (`docs/`)
Mỗi RQ có 1 báo cáo riêng, **sinh tự động từ CSV** (không viết tay số liệu):
`RQ1_SCM_ACCURACY_REPORT.md`, `RQ2_MODEL_COMPARISON_REPORT.md`, `RQ3_LLM_PARSER_BENCHMARK_REPORT.md`,
`RQ4_PROPAGATION_VALUE_REPORT.md`, `RQ6_TRAINTICKET_GENERALIZATION_REPORT.md`, `RQ5_RQ6_STATUS.md`.

- `paper_draft.tex` — bản thảo bài báo (IEEE), số liệu lấy trực tiếp từ các CSV/report trên.
- `table_rq3_parser_ablation.tex` — bảng LaTeX cho RQ3, sinh tự động.
- `UNIFIED_REFERENCE_DOC.md` — theo dõi tiến độ nghiên cứu theo từng RQ.

---

## ⚠️ Giới hạn đã biết (đọc trước khi viết luận điểm khoa học)
1. **`request_router.CALL_CHAINS['expected_delta_pct']`** là bảng hiệu chỉnh do nhóm tự đặt, dùng
   cả để tra cứu fast-path lẫn làm `expected_anchor` trong RQ3 cho nhóm In-Distribution — MAE thấp
   ở nhóm này phản ánh việc tra bảng đúng, không phải suy luận độc lập.
2. **Yêu cầu ngoài phạm vi taxonomy** hiện chưa được từ chối tường minh: `classify_request()` rơi
   về `GET_CATALOGUE` khi không khớp từ khóa nào, và LLM luôn bị ép chọn archetype gần nhất thay vì
   trả lời "không đủ dữ liệu hiệu chỉnh". `OODGuard` (đánh giá độ tin cậy theo percentile, có cơ sở
   lý thuyết) tồn tại trong `experiments/future_rca.py` nhưng **chưa nối vào `orchestrator.py`**.
   Chi tiết: `docs/paper_draft.tex` mục "Behavior at the Edge of the Declared Scope".
3. **Risk threshold** trong `test_new_features_simulation` (30%/15%/100%) là heuristic tự đặt,
   chưa hiệu chỉnh từ dữ liệu sự cố thật. Ngoại suy cực đoan (+150%/+300%) còn một số điểm bị đánh
   dấu `extrapolation_suspect` dù đã ràng buộc hệ số hồi quy không âm.
4. **LLM live** trong RQ3 có tính ngẫu nhiên (temperature=0.2), hiện chỉ chạy N=1 lần lặp do giới
   hạn quota API — luôn đọc `N Repeats` trước khi trích dẫn.
5. Free-tier Gemini API có quota theo phút/ngày khác nhau tuỳ model — xem comment trong
   `experiments/parser_benchmark_suite.py` (`LLM_MODEL_NAME`) nếu benchmark báo lỗi 429.
