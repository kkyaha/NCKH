# DỰ ÁN DỰ BÁO TẢI VÀ RỦI RO NĂNG LƯỢNG CHO TÍNH NĂNG MỚI BẰNG MÔ HÌNH NHÂN QUẢ SCM
## Structural Causal Models (SCM) & Do-Calculus for Zero-Shot Feasibility Forecasting

Hệ thống đánh giá **tính khả thi** khi có một **yêu cầu tính năng mới** (do khách hàng/PM đưa ra,
bằng ngôn ngữ tự nhiên) trong bảo trì phần mềm microservices — dự báo tài nguyên (CPU, Memory,
Socket, Latency) và cảnh báo rủi ro quá tải *trước khi* triển khai, bằng Mô Hình Nhân Quả Cấu Trúc
(SCM) và phép toán can thiệp $do(Workload)$, kết hợp Multi-Agent System (Parser Agent + Capacity
Agent) để dịch yêu cầu ngôn ngữ tự nhiên thành can thiệp $do(x)$.

**Phạm vi**: hệ thống nhắm tới các yêu cầu xấp xỉ được bằng một archetype hiệu chỉnh đã có trong
taxonomy của hệ đang chạy (xem `docs/paper_draft.tex` mục "Scope"), không phải dự đoán mở cho lĩnh
vực ứng dụng hoàn toàn chưa biết.

> 📄 **Bản thảo bài báo (nguồn số liệu chính thức duy nhất)**: `docs/paper_draft.tex`.
> Tài liệu này (README) chỉ là bản đồ định hướng code/data — nếu có mâu thuẫn giữa README và
> paper, **paper luôn đúng**, hãy báo lỗi README.

---

## ⚠️ ĐỌC TRƯỚC TIÊN: repo có BA thế hệ đánh số RQ, và chúng KHÔNG khớp nhau

Đây là nguồn nhầm lẫn lớn nhất khi đọc repo. Chữ "RQ3" trong tên file, trong báo cáo cũ, và
trong bài báo hiện tại là **ba thứ khác nhau**. Bảng dịch:

| Bài báo **hiện tại** (`paper_draft.tex`) | Khung V3 *(tài liệu đã xoá)* | Hệ **cũ** RQ1–RQ12 (tên file `rq*_.py`) |
|---|---|---|
| **RQ1 Necessity** — verifier có cần không | RQ1 Necessity | RQ3 (benchmark parser 50 prompt) |
| **RQ2 Guarantee** — bảo đảm gì, ngừng đúng ở đâu | RQ2 Attainability **+** RQ3 Realisation *(đã gộp)* | — |
| **RQ3 Measurability** — dữ liệu đỡ được tuyên bố dự báo không | RQ4 Measurability | RQ1 (SCM accuracy), RQ2 (model comparison) |
| **RQ4 Attribution** — graph hay do-operator mang nội dung nhân quả | RQ5 Attribution | RQ4 (propagation value), RQ7–RQ11 |
| **RQ5 Prospective** — phán quyết khả thi có đúng khi có ground truth | *(chưa tồn tại khi viết V3)* | RQ12 (prospective attribution) |
| — | — | RQ5 (coordination overhead) → còn trong bài, mục RQ1 |
| — | — | RQ6 (attribution validity) → **tách sang bài đồng hành** `docs/xai_attribution_paper_draft.tex` |

**Quy tắc khi đọc:**
- Nhãn LaTeX `sec:rq2`…`sec:rq6` trong `paper_draft.tex` **giữ nguyên tên cũ có chủ đích**
  (để 67 tham chiếu chéo không gãy) nên **số trong nhãn lệch số hiển thị** — mỗi chỗ lệch có
  một comment `% NOTE:` ngay trên đó.
- File `experiments/rq<số>_*.py` **thuộc hệ cũ**. Giữ lại vì chúng là nguồn gốc của các kết quả
  âm đã báo cáo, **không phải** vì còn được dùng. Các báo cáo `docs/RQ*_REPORT.md` đã được gộp
  vào [`docs/HE_THONG.md`](docs/HE_THONG.md) và xoá.
- Khi viết script mới: **đừng đặt tên theo số RQ.** Đặt theo cơ chế mà nó đo.

📂 **Mục lục toàn bộ 83 script thực nghiệm**: [`experiments/README.md`](experiments/README.md)

---

## 📂 CẤU TRÚC REPO

Repo tách rõ **hệ thống lõi** (sản phẩm thật, mô tả trong Section III của bài báo) khỏi
**hạ tầng thực nghiệm** (chỉ dùng để sinh số liệu cho các RQ, không phải một phần của sản phẩm):

```
NCKH/
├── src/                            # ============ HỆ THỐNG LÕI (the actual product) ============
│   ├── agents/
│   │   ├── orchestrator.py             # Điều phối 4-node LangGraph — ĐIỂM VÀO của hệ thống
│   │   ├── parser_agent.py             # NL requirement -> do(x): Three-Layer Verification
│   │   │                                 #   (Scope Gate + Entity Grounding + Bounded Projection)
│   │   ├── capacity_agent.py           # SCM tool: Fast-path (bivariate) + Accurate-path (Global DAG)
│   │   │                                 #   + G7 (post-simulation OOD-confidence annotation)
│   │   └── architecture_agent.py       # Topology mapping / blast-radius (graph BFS)
│   ├── scm/                            # Thư viện lõi dùng trực tiếp bởi agents
│   │   ├── data_processor.py               # Load/split dữ liệu, danh mục METRICS/SERVICES
│   │   └── request_router.py               # CALL_CHAINS: bảng hiệu chỉnh SockShop (viết tay,
│   │                                         #   xem "Giới hạn đã biết" #1 — CHƯA tham số hóa)
│   └── graph/
│       ├── sockshop_agent_graph.json       # Topology 7 service SockShop
│       ├── trainticket_agent_graph.json    # Topology 28 service Train Ticket
│       ├── extract_graph.py                # Trích xuất topology SockShop
│       └── extract_trainticket_graph.py    # Trích xuất topology Train Ticket
│
├── experiments/                    # ==== HẠ TẦNG THỰC NGHIỆM (sinh số liệu cho paper) ====
│   │                                 # Gom theo VAI TRÒ. Mục lục đầy đủ: experiments/README.md
│   ├── collect/      (6)               # Thu thập dữ liệu — CẦN hệ thật đang chạy
│   │                                   #   load_sweep_collect, probe_feature_chain, data_contract_check
│   ├── feasibility/  (7)               # RQ5: đóng băng → đánh giá  (freeze_* CHẠY TRƯỚC khi đo ramp)
│   │                                   #   freeze_predictions, evaluate_frozen, evaluate_p3
│   ├── model_eval/   (9)               # RQ3: so sánh mô hình, thống kê Holm, hình cho bài
│   │                                   #   model_comparison, statistical_rigor, make_figures
│   ├── parser/      (13)               # RQ1/RQ2: parser, các lớp guard, scope gate
│   ├── edges/       (19)               # RQ4: chọn cạnh SCM, so sánh lớp cơ chế
│   ├── datasets/    (10)               # Alibaba, Train Ticket, Online Boutique, BARO
│   └── legacy/      (19)               # ⚠ hệ đánh số CŨ (RQ6–RQ12) — giữ để tái lập, không còn dùng
│
├── run_all_experiments.py          # Entry point: chạy chuỗi RQ1/RQ2/RQ4 cho SockShop, RỒI RQ3
│                                     #   (live LLM, cần GOOGLE_API_KEY trong .env)
├── data/
│   ├── raw/                            # RE2-SS (SockShop) + trainticket/ (tải qua download script)
│   ├── benchmark/                      # 50 prompt cho RQ3 (parser_benchmark_prompts.json)
│   └── processed/scm_results/          # Toàn bộ output CSV — xem bảng bên dưới
├── docs/
│   ├── HE_THONG.md                     # ★ TÀI LIỆU TIẾNG VIỆT DUY NHẤT — đọc file này trước
│   ├── paper_draft.tex                 # bản thảo — NGUỒN SỐ LIỆU CHÍNH THỨC DUY NHẤT
│   ├── DATA_FRAMEWORK.md               # nhật ký thực nghiệm chi tiết chiến dịch SS-*
│   ├── xai_attribution_paper_draft.tex # bài đồng hành (nhánh XAI, tạm gác)
│   └── figures/                        # hình PDF, sinh bởi experiments/model_eval/make_figures.py
└── tests/                           # Unit test (smoke test, không thay thế đánh giá khoa học)
```

**Vì sao tách vậy**: `src/` là những gì một kỹ sư triển khai thật sẽ dùng (gọi `orchestrator.py`
với 1 câu yêu cầu, nhận về `FeasibilityReport`). `experiments/` chỉ tồn tại để trả lời các câu hỏi
nghiên cứu — không được `src/agents/*.py` import (trừ việc `experiments/*.py` import ngược lại
`src/` để dùng đúng code sản phẩm khi đánh giá), và không cần thiết nếu chỉ muốn chạy hệ thống.

---

## 🗺️ BẢN ĐỒ TRA CỨU: Claim trong paper ↔ Script ↔ Data

Dùng bảng này để đi từ 1 câu trong `paper_draft.tex` thẳng tới đúng script/CSV sinh ra số liệu đó
— tránh phải đọc lại toàn bộ lịch sử phát triển.

| Mục trong paper | Script sinh số liệu | CSV/report kết quả |
|---|---|---|
| §V RQ1 (accuracy, Sock Shop) | `run_all_experiments.py` → `evaluation_suite.py` | `test_f1_rmse_evaluation.csv`, `ground_truth_direct_match*.csv` |
| §V RQ1/RQ2 (Train Ticket, 90 kịch bản) | `download_trainticket_data.py --all` rồi `trainticket_evaluation.py` | `trainticket_*.csv` |
| §V RQ2 (SCM vs 3 baseline) | `model_comparison.py`, kiểm định trong `evaluation_suite.run_statistical_significance` | `05_model_comparison.csv`, `p_value_statistical_test.csv` |
| §V RQ3 (4-config ablation, live LLM) | `parser_benchmark_suite.py --live-llm` | `parser_ablation_benchmark.csv`, `docs/HE_THONG.md (muc 6)` |
| §V RQ4 (graph propagation vs uniform) | `evaluation_suite.run_rq4_propagation_value_test` + `run_rq4_multisplit_replication` | `rq4_propagation_value_test.csv`, `rq4_multisplit_summary.csv` |
| §V RQ5 (single-call vs Guarded MAS) | `rq5_coordination_overhead.py` | `rq5_coordination_overhead.csv`, `rq5_status_agreement.csv` |
| §III-B (G2) knee-point cho `[5,50]` | `g_threshold_sensitivity.py` | `g_threshold_sensitivity.csv`, `g2_sensitivity_vs_unguarded_raw.csv` |
| §III-B (G3) độc lập với chỉ dẫn prompt | `g3_independent_verification.py` | `g3_independent_verification.csv` |
| §III-B (G1: 18/18, G3: 33/36 adversarial) | `g1_g3_adversarial_stress_test.py` | `g1_g3_adversarial_stress.csv` |
| §"Edge of Scope" — Scope Gate evade 100% (phát hiện gốc) | `g6_scope_gate_adversarial_test.py` | `g6_scope_gate_adversarial.csv` |
| Thử semantic embedding — ĐÃ LOẠI BỎ, có bằng chứng | `scope_gate_embedding_calibration.py` (chạy với `--model=` / `--enriched`) | `scope_gate_embedding_calibration*.csv`, `scope_gate_threshold_sweep*.csv` |
| Fix cuối (self-declare + HITL + bỏ fast-path), **số liệu chính thức trong paper** | `g6_scope_gate_hitl_test.py` | `g6_scope_gate_hitl_validation.csv` |
| §"Post-Simulation Uncertainty" (G7) — sweep + đánh giá quy mô RQ3 | `g7_ood_guard_test.py` | `g7_ood_feasibility*.csv`, `g7_rq3_scale_evaluation.csv` |
| §"General Specification" — path enumeration khớp `services` | (kiểm chứng thủ công bằng `networkx.all_simple_paths`, không có script riêng — xem lịch sử hội thoại) | — |
| Nhánh XAI/attribution (RQ6 cũ, tạm gác) | `future_rca.py`, `rq6_attribution_validity.py`, `rq6_topology_check.py`, `rq6_tier_decomposition.py` | `rq6_*.csv`, `docs/HE_THONG.md (muc 6)`, `docs/xai_attribution_paper_draft.tex` |

**Lưu ý về `g6_scope_gate_layerA_test.py` vs `g6_scope_gate_hitl_test.py`**: cả hai cùng kiểm chứng
Scope Gate nhưng ở 2 thời điểm khác nhau. `layerA_test.py` gọi thẳng `_llm_full_parse()` (bỏ qua
fast-path), cho kết quả "11/11 đã sửa" — **kết quả này SAI/không phản ánh pipeline thật**, vì
fast-path (khi đó còn tồn tại) khiến 8/11 prompt đối kháng lọt qua trong thực tế. `hitl_test.py` gọi
đúng `parser.parse()` (API công khai), phát hiện lỗi trên, và là **số liệu duy nhất nên trích dẫn**.
File `layerA_test.py` được giữ lại vì lý do lịch sử/minh họa lỗi phương pháp luận, không dùng để
công bố số liệu.

---

## ⚡ CHẠY

```bash
# --- Chạy HỆ THỐNG LÕI (1 yêu cầu tính năng mới -> FeasibilityReport) ---
python src/agents/orchestrator.py
# Hoặc gọi với requirement khác:
python -c "
from src.agents.orchestrator import feasibility_analyzer
result = feasibility_analyzer.invoke({'input_requirement': 'Áp mã giảm giá 20% khi thanh toán'})
print(result['feasibility_report'])
"

# --- Chạy RQ1/RQ2/RQ4 (+ RQ3 live LLM ở bước cuối) trên SockShop ---
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python run_all_experiments.py   # cần .env chứa GOOGLE_API_KEY cho bước RQ3

# --- RQ3: ablation parser, LLM thật (cần .env chứa GOOGLE_API_KEY) ---
python experiments/parser_benchmark_suite.py --live-llm
# Re-run CHỈ 1 config (vd sau khi sửa code chỉ ảnh hưởng Guarded), không tốn quota cho phần còn lại:
python experiments/parser_benchmark_suite.py --live-llm --only=Guarded_Hybrid_Parser
# Chỉ regenerate report/table từ CSV đã có, không gọi LLM:
python experiments/parser_benchmark_suite.py --live-llm --only=__NONE__

# --- RQ1/RQ2 trên Train Ticket (90 kịch bản) ---
python experiments/download_trainticket_data.py --all
python experiments/trainticket_evaluation.py

# --- RQ5 ---
python experiments/rq5_coordination_overhead.py

# --- Kiểm chứng lại Scope Gate (đối kháng + HITL), LLM thật ---
python experiments/g6_scope_gate_hitl_test.py
```

Mọi script live-LLM đều rate-limit ~4.5s/call (free-tier Gemini) và ghi CSV **theo từng dòng**
(flush ngay) — nếu bị ngắt giữa chừng do hết quota, dữ liệu đã chạy không mất, chạy lại sẽ ghi đè
từ đầu (không tự resume).

---

## 📊 KẾT QUẢ THỰC NGHIỆM (`data/processed/scm_results/`)

Bảng đầy đủ hơn nằm ở mục "Bản đồ tra cứu" phía trên (nối trực tiếp tới claim trong paper). Vài
ghi chú chung:

- File `*_adversarial*.csv`, `*_hitl*.csv`, `*sensitivity*.csv`, `*calibration*.csv` (đầu `g1_`,
  `g3_`, `g6_`, `g7_`, `g_`, `scope_gate_`) là **bằng chứng thô cho các claim về guard** trong
  §III-B/Discussion — không phải dữ liệu benchmark chính (RQ1-RQ5).
- `parser_ablation_benchmark.csv`: kiểm tra cột `llm_backend` trước khi trích dẫn — nếu bắt đầu
  bằng `SYNTHETIC_` là bộ giả lập offline (chỉ CI, không dùng làm bằng chứng khoa học).
- `rq6_*.csv`: thuộc nhánh XAI/attribution đã tách khỏi `paper_draft.tex`, chỉ còn dùng trong
  `docs/xai_attribution_paper_draft.tex` (bài đồng hành, chưa hoàn thiện).

---

## 📖 TÀI LIỆU (`docs/`)

- **`paper_draft.tex`** — bản thảo bài báo chính (IEEE), nguồn số liệu duy nhất nên trích dẫn.
- `xai_attribution_paper_draft.tex` — bài đồng hành về độ tin cậy attribution (Shapley), tách
  riêng khỏi bài chính, **tạm gác lại**.
- `khungtoanhoc.tex` — khung toán học nền tảng (ngoài phạm vi công việc hiện tại).
- `table_rq3_parser_ablation.tex` — bảng LaTeX RQ3, sinh tự động bởi `parser_benchmark_suite.py`.
- `RQ*_REPORT.md` — báo cáo tự động sinh từ CSV cho từng RQ.
- `docs/HE_THONG.md (muc 6)` — báo cáo nhánh XAI (Part A/B), phục vụ bài đồng hành.
- `docs/HE_THONG.md` — nhật ký quyết định thiết kế qua các giai đoạn.

---

## ⚠️ Giới hạn đã biết (đọc trước khi viết luận điểm khoa học)

1. **`CALL_CHAINS` (bảng hiệu chỉnh, `request_router.py`) chưa tham số hóa.** `ParserAgent` đã
   nhận `system_name`/`domain_description`/`known_services` làm tham số (tự suy từ graph), nhưng
   `CALL_CHAINS` vẫn là `import` cố định cho SockShop — trỏ agent vào graph hệ thống khác **không
   đủ** để triển khai thật, vì bảng archetype vẫn là của SockShop. Xem paper §"General
   Specification for Deploying to a New Target System".
2. **Scope Gate (Layer 1) không hoàn hảo** ngay cả sau khi sửa: 5/50 prompt hợp lệ nhưng thụ
   động/tĩnh (FAQ, điều khoản dịch vụ, đa ngôn ngữ UI, footer, fraud-detection tự động) vẫn bị xử
   lý sai một phần — cơ chế human-in-the-loop (`needs_human_review`) giảm nhẹ nhưng không giải
   quyết triệt để (2/5 vẫn còn sai). Xem paper §"Behavior at the Edge of the Declared Scope".
3. **Không còn fast-path trong `ParserAgent.parse()`** (đã gỡ bỏ sau khi phát hiện fast-path khiến
   8/11 prompt đối kháng lọt qua Scope Gate hoàn toàn) — mọi request đều gọi LLM, mất lợi ích chi
   phí/độ trễ đã đo trong bản RQ3 trước đó (~34% bypass rate không còn áp dụng).
4. **G7 (post-simulation OOD annotation)** hiện có độ đặc hiệu thấp trên Sock Shop: bắn cờ 80% các
   trường hợp bình thường, chủ yếu do 1 node (`catalogue_cpu`) có phân phối lệch nặng — chưa nên
   dùng tỷ lệ cảnh báo của G7 làm tín hiệu quyết định, xem paper để biết chi tiết.
5. **LLM live** có tính ngẫu nhiên (temperature=0.2) — luôn đọc số lần lặp (N repeats) trước khi
   trích dẫn bất kỳ con số nào từ một script `--live-llm`.
6. Free-tier Gemini API có quota theo phút/ngày — nếu script báo lỗi 429, đợi quota reset (thường
   theo ngày UTC) rồi chạy lại; dữ liệu từng dòng đã ghi sẽ không mất (xem mục "CHẠY").
7. **Global DAG của Train Ticket không dự báo được latency (mọi service).** Phát hiện qua
   `src/scm/node_impact.py` (đánh giá elasticity/độ ổn định từng node): cả 28/28 cơ chế
   `<service>_latency-50` trong `train_accurate_path()` fit ra hệ số hồi quy **= 0 tuyệt đối**
   (`QueueingLatencyRegressor.model_.coef_ == [0, 0]`), tức mô hình luôn dự báo latency không
   đổi bất kể workload — không phải lỗi code: tương quan thô workload↔latency trên dữ liệu
   "bình thường" (trước inject fault) của Train Ticket chỉ ~0.0016 (gần như không có), có lẽ vì
   khoảng workload quan sát được trong giai đoạn bình thường quá hẹp để lộ hiệu ứng hàng đợi.
   Khác với Sock Shop, nơi hiện tượng này chỉ xảy ra ở 1/7 node (đã ghi trong docstring
   `QueueingLatencyRegressor`). Do đó: (a) `simulate_intervention()`/`certified_envelope()` trên
   Train Ticket vẫn chạy và trả về số, nhưng **mọi con số latency đều là hằng số baseline, không
   phản ánh can thiệp** — chỉ CPU/Memory/Socket (Fast Path) và các node `_cpu`/`_workload` trong
   Global DAG là đáng tin cho Train Ticket; (b) `rank_user_impact()` mặc định dùng target
   `_latency-50` sẽ trả về rỗng cho Train Ticket vì lý do này — dùng
   `target_nodes=[f'{s}_cpu' for s in services]` thay thế cho hệ thống này. Chưa có bản sửa;
   cần dữ liệu Train Ticket với dải workload rộng hơn (bao gồm giai đoạn tải cao) hoặc một cơ chế
   latency khác không phụ thuộc hoàn toàn vào workload tuyến tính. Trong lúc chưa sửa được, có
   thể đánh dấu tay 28 node này bằng `agent.mark_node_unreliable(node, reason=...)`
   (`capacity_agent.py`) để `get_metrics_for_service()`/`simulate_intervention()` trả về
   `..._confidence = 'manually_flagged'` cho chúng ngay cả khi `evaluate_node_stability()`
   chưa/không chạy lại — xem docstring `mark_node_unreliable()`. **Tự động cũng bắt được**
   (không cần đánh dấu tay): `evaluate_node_stability()`/`_node_confidence()` gắn cờ
   `'unstable'` cho cả 28/28 node này — nhưng ban đầu suýt không bắt được, vì latency của
   Train Ticket có biên độ rất hẹp (~0.01-0.05) nên một dự báo HẰNG SỐ vẫn có MAPE trông thấp
   (5-30%, có node còn tới mức "EXCELLENT") dù R² âm mọi node (-4.3 đến -0.002) và hệ số = 0 —
   đúng cái bẫy MAPE-một-mình mà `docs/HE_THONG.md (muc 6)` đã cảnh báo. Đã sửa bằng
   cách kiểm `coef_` trực tiếp (`node_impact._is_degenerate_mechanism`) thay vì chỉ suy từ
   MAPE/R² — coef≈0 luôn ép tag về `'POOR'`/`'unstable'` bất kể MAPE nói gì.

