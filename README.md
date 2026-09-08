# DỰ ÁN DỰ BÁO TẢI VÀ RỦI RO NĂNG LƯỢNG CHO TÍNH NĂNG MỚI BẰNG MÔ HÌNH NHÂN QUẢ SCM
## Structural Causal Models (SCM) & Do-Calculus for Zero-Shot Capacity Planning

Hệ thống dự báo tài nguyên (CPU, Memory, Socket, Latency) và cảnh báo rủi ro quá tải cho
**tính năng phần mềm MỚI** bằng Mô Hình Nhân Quả Cấu Trúc (SCM) và phép toán can thiệp
$do(Workload)$, kết hợp một Multi-Agent System (Parser Agent + Capacity Agent) để dịch
yêu cầu ngôn ngữ tự nhiên thành can thiệp $do(x)$.

---

## ⚡ CHẠY TOÀN BỘ THỰC NGHIỆM

```bash
# Cần .env chứa GOOGLE_API_KEY để chạy đầy đủ bước RQ3 (LLM Parser). Không có key
# vẫn chạy được 6/7 bước còn lại (SCM, model comparison, ground-truth matching).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python run_all_experiments.py

# Nếu muốn ép RQ3 dùng bộ giả lập offline (CI smoke-test, KHÔNG dùng để công bố):
python -c "import sys; sys.argv=['x','--offline']; from src.scm.parser_benchmark_suite import run_parser_benchmark; run_parser_benchmark()"
```

`run_all_experiments.py` chạy tuần tự 7 bước — xem docstring đầu file để biết chi tiết
từng bước. Ba bước 1, 5, 6 tạo ra **3 nguồn bằng chứng độc lập** về độ chính xác SCM
(bucket-average OOD, kiểm định thống kê theo %, và so khớp điểm-thật không bucket trên
toàn bộ 90 run) — nên đối chiếu cả ba khi viết kết luận.

---

## 📊 KẾT QUẢ THỰC NGHIỆM (`data/processed/scm_results/`)

| File | Nội dung | Sinh bởi |
|---|---|---|
| `test_f1_rmse_evaluation.csv` | Độ chính xác SCM (RMSE/MAE/MAPE/SMAPE/R²) theo service×metric, cả bản bucket-average và full-resolution (`*_full_res`) | `evaluation_suite.run_f1_rmse_benchmark` |
| `05_model_comparison.csv` | Đối chiếu SCM vs LinearReg/GradBoost/GaussianProcess, cả bucket và full-resolution | `model_comparison.py` |
| `p_value_statistical_test.csv` | Wilcoxon/Friedman theo TỪNG metric trên `mape_pct` (không gộp đơn vị RMSE khác nhau) | `evaluation_suite.run_statistical_significance` |
| `ground_truth_direct_match.csv` | ~400k điểm so khớp SCM với giá trị đo thật trên toàn bộ 90 (scenario×run) của RE2-SS, train/test tách theo thời gian trong từng run, không bucket | `compare_with_ground_truth.py` |
| `ground_truth_direct_match_summary.csv` | Tổng hợp sai số theo (service, metric) từ file trên | như trên |
| `test_new_features_simulation.csv` | Mô phỏng do(workload) cho các tính năng mới + Flash Sale/Black Friday trên Global 28-Node DAG. **`risk_status` là heuristic tự đặt, chưa đối chiếu với sự cố thật** | `evaluation_suite.test_new_features_simulation` |
| `14_node_causal_propagation.csv` | Lan truyền do(front-end_workload=+50%) qua 2 tầng Workload→CPU | `evaluation_suite.test_14_node_causal_graph` |
| `parser_ablation_benchmark.csv` | RQ3: ablation 4 cấu hình parser × 50 prompt × N lần lặp, dùng LLM thật (cột `llm_backend` cho biết backend nào) | `parser_benchmark_suite.py` |
| `parser_ablation_by_category.csv` | RQ3 breakdown theo 4 nhóm prompt (In-Distribution/Complex/Subtle/Adversarial) | như trên |
| `future_rca_results.csv` | Pre-mortem RCA (OOD guard + Shapley attribution) cho các kịch bản tăng tải | `future_rca.py` |
| `01_system_telemetry_train_test.csv`, `02_system_test_cases_catalog.csv` | Dữ liệu viễn trắc gộp + danh mục test case | `data_processor.py` |

⚠️ Trước khi trích dẫn `parser_ablation_benchmark.csv` trong bài báo, kiểm tra cột
`llm_backend`: nếu bắt đầu bằng `SYNTHETIC_` nghĩa là chạy bằng bộ giả lập offline
(không phải LLM thật) — chỉ dùng cho CI, không dùng làm bằng chứng khoa học.

---

## 📖 TÀI LIỆU (`docs/`)
- `RQ3_LLM_PARSER_BENCHMARK_REPORT.md` — sinh **tự động** từ `parser_ablation_benchmark.csv` (không viết tay số liệu), có mục giới hạn phương pháp luận.
- `table_rq3_parser_ablation.tex` — bảng LaTeX cho bài báo, sinh tự động cùng lúc.
- `UNIFIED_REFERENCE_DOC.md` — tài liệu theo dõi tiến độ nghiên cứu theo từng RQ.

Các báo cáo tổng hợp trước đây (`COMPREHENSIVE_PAPER_DRAFT.md`, `FULL_SCM_EXPERIMENT_REPORT.md`,
`Q1_SCM_BENCHMARK_SYNTHESIS_REPORT.md`) đã bị xoá vì chứa số liệu không khớp với các file CSV
gốc (ví dụ p-value, MAPE/SMAPE bị chép tay và lệch pha sau các lần chạy lại). Khi viết bản thảo
mới, hãy lấy số liệu trực tiếp từ CSV trong bảng ở trên, không chép tay.

---

## 📂 CẤU TRÚC MÃ NGUỒN

```
NCKH/
├── run_all_experiments.py         # Chạy toàn bộ 7 bước thực nghiệm
├── data/
│   ├── raw/RE2-SS/                # Dữ liệu viễn trắc gốc (30 scenario × 3 run × 7 service)
│   ├── benchmark/                 # 50 prompt cho RQ3 (parser_benchmark_prompts.json)
│   └── processed/scm_results/     # Toàn bộ output — xem bảng ở trên
├── src/
│   ├── scm/
│   │   ├── data_processor.py          # Load/split dữ liệu, danh mục CALL_CHAINS/METRICS
│   │   ├── evaluation_suite.py        # RQ1: accuracy, simulation, 14-node graph, thống kê
│   │   ├── model_comparison.py        # RQ1: đối chiếu SCM vs 3 baseline ML
│   │   ├── compare_with_ground_truth.py # Bằng chứng ground-truth mạnh nhất (không bucket)
│   │   ├── parser_benchmark_suite.py  # RQ3: ablation LLM parser (LLM thật)
│   │   ├── future_rca.py              # Pre-mortem RCA (OOD guard + Shapley)
│   │   ├── request_router.py          # Phân loại request + bảng hiệu chỉnh CALL_CHAINS
│   │   └── scm_pipeline.py
│   ├── agents/
│   │   ├── parser_agent.py            # 2-tier: fast-path bảng hiệu chỉnh + LLM có guard
│   │   ├── capacity_agent.py, architecture_agent.py, performance_agent.py, simulation_agent.py
│   │   └── orchestrator.py            # Điều phối ReAct đa tác tử
│   ├── graph/sockshop_agent_graph.json # Topology 7 service SockShop
│   └── etl/
└── tests/                          # Unit test (smoke test, không thay thế đánh giá khoa học)
```

---

## ⚠️ Giới hạn đã biết (đọc trước khi viết luận điểm khoa học)
1. **`request_router.CALL_CHAINS['expected_delta_pct']`** là bảng hiệu chỉnh do nhóm tự đặt,
   dùng cả để tra cứu fast-path lẫn làm `expected_anchor` trong benchmark RQ3 cho nhóm
   In-Distribution — MAE thấp ở nhóm này phản ánh việc tra bảng đúng, không phải suy luận
   độc lập. Tín hiệu tổng quát hoá thật nằm ở nhóm Complex/Subtle/Adversarial.
2. **Risk threshold** trong `test_new_features_simulation` (30%/15%/100%) là heuristic tự
   đặt, chưa hiệu chỉnh từ dữ liệu sự cố thật.
3. **LLM live** trong RQ3 có tính ngẫu nhiên (temperature=0.2); luôn đọc số lần lặp
   (`N Repeats`) và độ lệch chuẩn trước khi trích dẫn.
4. Free-tier Gemini API có quota theo phút/ngày khác nhau tuỳ model — xem comment trong
   `parser_benchmark_suite.py` (`LLM_MODEL_NAME`) nếu benchmark báo lỗi 429.
