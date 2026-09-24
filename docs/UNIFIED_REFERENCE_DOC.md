# Zero-Shot Capacity Planning trong Vi dịch vụ bằng SCM & Do-Calculus

> ⚠️ **TÀI LIỆU THEO HỆ ĐÁNH SỐ RQ CŨ (RQ1–RQ12).** Số RQ trong file này **không khớp**
> khung 5 RQ hiện tại của `docs/paper_draft.tex`. Xem bảng dịch ở đầu `README.md` gốc.
> Giữ lại làm hồ sơ gốc; **không dùng làm nguồn số liệu**.
## Tài liệu Tham chiếu Thống nhất — Bản chuẩn bị Public

> Đây là tài liệu hợp nhất TOÀN BỘ quyết định thiết kế cuối cùng, thay thế các file rời rạc trước đó (`Reference_C1-C11`, `Ke_hoach_thu_nghiem`, `Parser_Agent_Tong_hop`, `Cau_truc_du_lieu_Test`). Dùng file này làm nguồn duy nhất khi viết paper/code/README.

---

## 1. Tuyên bố nghiên cứu

Đánh giá tính khả thi của một yêu cầu thay đổi **trước khi triển khai** lên kiến trúc vi dịch vụ đang chạy, bằng cách mô phỏng tác động qua toán tử can thiệp `do()` trên một Structural Causal Model (SCM) đã fit từ dữ liệu vận hành thật — thay vì suy luận tương quan hoặc chờ sự cố xảy ra rồi mới chẩn đoán (RCA truyền thống).

**Không còn dùng khung "Meta-model M=⟨E,A,R,C,T,P⟩" làm khung trình bày chính** — giữ ở Phụ lục A cho mục đích luận văn nếu cấu trúc đề cương yêu cầu.

---

## 2. Kiến trúc hệ thống (bản chốt cuối)

```
Requirement (raw_text, + business_context_hint / design_doc / code_diff nếu có)
        │
        ▼
┌─────────────────────────────────────────────┐
│ AGENT 1 — Parser Agent                        │  Perception + Reasoning + Action (ReAct)
│  Regex → Calibration Table → Similarity-search│  3-Layer Verification (Scope Gate, Entity Grounding, Bounded Projection Π)
└─────────────────────────────────────────────┘
        │  ParsedRequirement{injection_service, injection_delta{low,mean,high}, confidence, reasoning}
        ▼
┌─────────────────────────────────────────────┐
│ TOOL — Workload Propagation (Tầng 1)          │  SCM biến Workload, do() lan truyền qua depends_on
│                                                │  Post-Simulation UQ Annotation (percentile-based OOD caveat)
└─────────────────────────────────────────────┘
        │  workload cục bộ từng node
        ▼
┌─────────────────────────────────────────────┐
│ AGENT 2 — Capacity Agent (đã gộp)             │  = PerformanceAgent + SimulationAgent (cũ)
│  Tool: SCM bivariate/28-node (Fast/Accurate)  │  Reasoning: nhận diện bottleneck, sinh khuyến nghị
└─────────────────────────────────────────────┘
        │
        ▼
   FeasibilityReport (verdict, metric_projected, confidence_score, reasoning_trace)
```

**Định nghĩa "Agent" dùng trong paper**: theo ReAct (Yao et al., 2022) — 1 Agent là vòng lặp Reasoning (LLM suy luận) xen kẽ Acting (gọi Tool), không phải hàm `model.predict()` trả số thẳng. SCM (bivariate lẫn 28-node) là **Tool**, không phải Agent — được Parser Agent và Capacity Agent gọi tới.

---

## 3. Cơ sở lý thuyết (đã xác minh)

Bảng cơ sở lý thuyết được chuẩn hóa và phân theo 4 trụ cột kiến trúc cốt lõi của hệ thống:

### Nhóm I: Cơ sở Lý thuyết Nhân quả & Can thiệp (Causal Foundation & SCM)
| Khái niệm / Định lý | Nguồn / Trích dẫn | Thành phần áp dụng trong hệ thống |
|---|---|---|
| Thang bậc nhân quả (Association / Intervention / Counterfactual) | Pearl (2009) | Khung lý thuyết phân biệt OOD so với ML tương quan |
| Structural Causal Model, Additive Noise Model (ANM) | Peters, Janzing & Schölkopf (2017); Hoyer et al. (2009, NeurIPS) | Tool: SCM Engine (Bivariate & 28-node DAG) |
| Do-calculus, toán tử can thiệp $do()$ & tính đầy đủ | Pearl (1995); Huang & Valtorta (2006, UAI — Best Student Paper) | Tool: SCM Simulation, cắt cạnh back-door |
| Outlier score, dropping-noise, Shapley Theorem 3 | Janzing, Budhathoki, Minorics & Blöbaum (2019/2022, ICML) | Agent 2: Capacity Agent (Phân bổ căn nguyên lỗi) |
| Giá trị Shapley gốc (Axiomatic Attribution) | Shapley (1953) | Agent 2: Báo cáo giải trình minh chứng (`evidence_refs`) |
| DoWhy-GCM (`arrow_strength`, `interventional_samples`) | Blöbaum et al. (2024, JMLR 25:147) | Nền tảng thư viện tính toán SCM dưới tầng lõi |
| `intrinsic_causal_influence` | Janzing, Blöbaum et al. (2024, AISTATS) | Đánh giá độ nhạy ảnh hưởng trực tiếp giữa các microservices |
| Tác động của đồ thị không hoàn hảo (Missing structural knowledge) | Okati, Garrido Mejia, Orchard, Blöbaum, Janzing (2024, UAI) | Mục 8: Threats to Validity (Độ nhạy của topology) |

### Nhóm II: Động học Xếp hàng & Lan truyền Đồ thị (Queueing Kinetics & Graph Cascade)
| Khái niệm / Định lý | Nguồn / Trích dẫn | Thành phần áp dụng trong hệ thống |
|---|---|---|
| Lý thuyết Xếp hàng M/M/1 (Queueing Theory) | Kleinrock (1975) | Tool SCM: Mô hình hóa tiệm cận bão hòa phi tuyến của Latency |
| Công thức Kingman tải nặng (Heavy-Traffic Approximation) | Kingman (1961) | Khắc phục điểm mù ngoại suy tuyến tính của ANM khi $C \to 100\%$ |
| Lan truyền tải đa chặng & Cascade Attenuation | Lý thuyết Call Graph / Distributed Tracing | Tool: Workload Propagation Engine (Tầng 1) |

### Nhóm III: Đa tác tử & Ngôn ngữ tự nhiên có Neo (Neuro-Symbolic MAS & Grounded NLP)
| Khái niệm / Kỹ thuật | Nguồn / Trích dẫn | Thành phần áp dụng trong hệ thống |
|---|---|---|
| ReAct: Synergizing Reasoning and Acting | Yao et al. (2022, ICLR) | Định nghĩa Agent (Parser Agent & Capacity Agent) |
| CausalPlan — SCM ràng buộc LLM trong Multi-Agent | arXiv:2508.13721 (2025) | Cơ chế kiểm soát LLM bằng mô hình hình thức |
| CAMEF — Văn bản sự kiện kết hợp Causal cho dự báo | Zhang et al. (2025, KDD) | Chuyển đổi ngữ nghĩa sự kiện thành delta can thiệp |
| Text2TimeSeries & Bounded Adjustment | Kurisinkel et al. (2024) | Parser Agent: Bảng hiệu chuẩn Calibration Table |
| Context as Boundary Adjustment (không dùng làm số chính) | Multi-Modal Time-Series (2024/2025) | Layer 3 (Bounded Projection): Inner Clamp giới hạn tinh chỉnh LLM $\le \pm 10\%$ quanh anchor |
| CAPTime — Đầu ra phân phối xác suất thay vì điểm đơn lẻ | arXiv:2505.10774 (2025) | Định dạng delta: `{low, mean, high}` của Parser |
| Synthetic Method of Analogues (Cosine Similarity Anchor) | Murph et al. (2025, PLoS Comp Biol) | Cost–accuracy parameter τ = 0.6: Ngưỡng tương đồng với template (fast-path gate) |
| Function Point Analysis (FPA) | Albrecht (1979) | Parser Agent Chế độ A.5 (Ước lượng từ Design Doc) |
| Just-In-Time Defect Prediction Features | Kamei et al. (2013); PyDriller | Parser Agent Chế độ B (Trích xuất từ Code Diff) |

### Nhóm IV: Dữ liệu thực nghiệm & Kiểm định Thống kê (Benchmarks & Statistical Rigor)
| Khái niệm / Benchmark | Nguồn / Trích dẫn | Thành phần áp dụng trong hệ thống |
|---|---|---|
| RCAEval (Benchmark chuẩn quốc tế cho Microservices) | Pham et al. (2025, WWW) | Bộ dữ liệu chính (SockShop, 14 node, 90 runs) |
| PetShop Benchmark (Môi trường vi dịch vụ bổ trợ) | Hardt, Orchard, Blöbaum et al. (2024, CLeaR) | Đối chứng mở rộng cho generalization (RQ6) |
| Khảo sát Causal RCA trong Microservices (đa dataset) | Soldani et al. (2024, ACM Comput. Surv., DOI 10.1145/3691620) | Cơ sở khẳng định tính nhạy cảm dataset & khoảng trống nghiên cứu |
| Kiểm định Thống kê so sánh nhiều mô hình (Wilcoxon, Friedman) | Demšar (2006, JMLR) | Phân tích RQ2: Kiểm định ý nghĩa thống kê giữa SCM vs ML |

**Đã kiểm chứng và LOẠI BỎ (bài học giữ lại)**: Shanmugam et al. (2015), Castelletti & Consonni (2020) — có thật nhưng thuộc Causal Structure Learning (tìm cấu trúc đồ thị khi chưa biết), sai lĩnh vực so với Causal Effect Estimation (đồ thị đã biết/đã fit từ architecture & tracing) của nghiên cứu này. Nguyên tắc rút ra: Luôn kiểm tra giả định nền tảng (known DAG vs unknown DAG) trước khi trích dẫn.

---

## 4. Three-Layer Symbolic Verification Architecture

### Kiến trúc mới (thay thế danh sách G1–G8 cũ)

| Layer | Tên | Mục đích toán học | Cơ chế trong code |
|---|---|---|---|
| **L1** | **Scope Gate** (Selective Abstention) | Từ chối trả lời khi yêu cầu ngoài taxonomy đã hiệu chuẩn | `similarity_score == 0.0` → `confidence = REFUSED`, `is_out_of_scope = True` |
| **L2** | **Entity Grounding** (Graph-Membership) | Ràng buộc mọi entity do LLM đề xuất phải tồn tại trong đồ thị | Gateway Invariance: `in-degree == 0`; Service Existence: `core_services ⊆ KNOWN_SERVICES` |
| **L3** | **Bounded Projection Π** (Nested Clamp) | Chiếu delta vào khoảng an toàn thực nghiệm | Outer: `clamp(δ, 5%, 50%)`; Inner: `clamp(adj, -10%, +10%)` |

### Các cơ chế đã hạ cấp / chuyển vị trí

| Cơ chế cũ | Quyết định mới | Lý do |
|---|---|---|
| G5 (Similarity-Gated Fine-Tuning) | **Design parameter τ** — không phải guard | Sweep τ ∈ [0.3, 0.8] không thay đổi PBVR/SHR/MAE; chỉ ảnh hưởng fast-path bypass rate |
| G7 (OOD-Confidence) | **Post-Simulation UQ Annotation** — chuyển sang Capacity Agent | Chạy sau SCM simulation, không thuộc parsing; fire rate 80% → mất tính chọn lọc |
| G8 (Additive shift multi-injection) | **Chưa implement** — out of scope | Pipeline chỉ hỗ trợ single-intervention |

### Preconditions (không phải per-request verification)

| Kiểm tra | Thời điểm | Mô tả |
|---|---|---|
| DAG acyclicity | Graph construction | `depends_on` phải là DAG, loại bỏ cycle trước khi dùng |
| Resource range [0, 100] | **Chưa implement** — thay bằng non-negative coefficients | Mitigated ở mức causal mechanism thay vì output clamp |

---

## 5. Research Questions (bản chốt cuối)

| RQ | Câu hỏi | Đo bằng | Trạng thái |
|---|---|---|---|
| RQ1 | SCM+do-calculus dự đoán đúng đến mức nào, trên từng hệ thống? | MAPE/RMSE/R² full-resolution + ground-truth direct match (không bucket) | Đã chạy đầy đủ trên **cả Sock Shop và Train Ticket** (paper §V-A, Table `tab:rq1`/`tab:rq1-gt`). Có thêm "RQ1 Follow-up" (mở rộng Backpressure) đã kiểm chứng và triển khai production trên cả 2 hệ. Số liệu THẬT nằm trong CSV — không chép tay vào tài liệu này. |
| RQ2 | Causal có vượt trội tương quan/hộp đen không, đặc biệt vùng OOD? | Wilcoxon/Friedman TÁCH theo từng metric trên MAPE full-resolution | Đã chạy trên cả 2 hệ thống (paper §V-B). Sock Shop: SCM đồng hạng nhất (7/21, có ý nghĩa vs LinearReg p=0.0137). Train Ticket: SCM thắng ÍT NHẤT (12/84) — kết quả KHÔNG lặp lại. Kết luận trung thực: KHÔNG claim SCM vượt trội độ chính xác điểm; giá trị nằm ở toán tử do(x). |
| RQ3 | Parser Agent neo dữ liệu có chính xác hơn LLM tự do không? | MAE, PBVR, SHR, GMR — LLM THẬT (gemini-flash-lite-latest), N lần lặp | Đã chạy với LLM thật (không còn dùng bộ giả lập offline). Xem `docs/RQ3_LLM_PARSER_BENCHMARK_REPORT.md` (sinh tự động từ CSV). |
| RQ4 | Workload Propagation có chính xác hơn gán đều delta không? | So MAPE có/không Tầng 1, đối chứng Uniform Delta Injection | Đã chạy (paper §V, RQ4): SCM thắng 5/6 service, nhất quán qua 5 tỷ lệ split (pooled Wilcoxon p=2×10⁻⁵); đơn split n=6 chưa có ý nghĩa (p=0.094). Xem `docs/RQ4_PROPAGATION_VALUE_REPORT.md`. |
| RQ5 | Kiến trúc đa tác tử (Parser+Capacity Agent) có vượt trội single-LLM-call không? | So PBVR/SHR/GMR/MAE + LLM calls/latency — LLM THẬT | Đã chạy với LLM thật (paper §V, RQ5): Guarded MAS đưa PBVR 27.3%→0%, Anchor MAE 53.6%→1.9%, đổi lấy 1.91× LLM calls và 3.9× latency; Status Agreement với phán quyết SCM = 47.1%. Xem `docs/RQ5_COORDINATION_OVERHEAD_REPORT.md`. |
| RQ6 (đã gộp) | Generalize sang Train Ticket không cần đổi kiến trúc? | — | **Không còn là 1 RQ riêng.** Đánh giá Train Ticket đã gộp thẳng vào RQ1/RQ2 làm so sánh theo-từng-hệ-thống (paper §IV "Datasets and Testbed", §V-A/B). Nhánh XAI/attribution (RQ6 cũ trong code, `rq6_*.py`) tách riêng sang bài đồng hành `xai_attribution_paper_draft.tex`, tạm gác lại. |

**Ưu tiên**: RQ1-RQ5 đã có số liệu đầy đủ trong `paper_draft.tex`; RQ6 (dạng câu hỏi riêng) không còn được theo dõi độc lập.

---

## 6. Việc cần sửa trong code hiện tại trước khi chạy chính thức

| # | Vấn đề | File |
|---|---|---|
| 1 | `test_14_node_causal_graph()` nối sai CPU->CPU trực tiếp, không dùng chung đồ thị đúng | thay bằng gọi `build_and_train_global_dag()` |
| 2 | Ngưỡng F1 dùng percentile trên `df` gộp cả train+test -> rò rỉ nhẹ | dùng `df_train` hoặc Policy cố định |
| 3 | Gộp `PerformanceAgent` + `SimulationAgent` -> `CapacityAgent`, thêm bước Reasoning sinh `reasoning_trace` thật | `src/agents/` |
| 4 | `KNOWN_GATEWAYS` hardcode string -> nên suy từ in-degree=0 | `parser_agent.py` |
| 5 | ~~Thiếu guard cứng cho delta/adjustment~~ — **ĐÃ XONG**: Bounded Projection Π (Layer 3) | `parser_agent.py` |
| 6 | `classify_blast_radius()` (G8) chưa có | `request_router.py` |
| 7 | Làm rõ đang dùng bivariate hay 28-node multi-hop trong mô tả "lan truyền qua call chain" | tài liệu + code comment |

---

## 7. Dữ liệu thực nghiệm — tóm tắt pipeline RCAEval

```
Case RCAEval (metrics.csv, traces.csv, metadata.json)
   -> pivot EAV -> tách baseline/post theo injection_timestamp
   -> pool baseline theo hệ thống (không trộn 3 hệ) -> fit SCM
   -> post-injection làm ground truth (silver-standard, leave-one-out)
   -> case study gold-standard (1-2 tính năng thật + k6 load test, dự đoán đóng băng trước deploy)
```
Đồ thị `depends_on`: tam giác hóa 3 nguồn (khai báo YAML đã sửa hướng cạnh queue-master/rabbitmq, quan sát từ traces, suy luận LiNGAM/NOTEARS).

---

## 8. Threats to Validity (khai báo tường minh)

- Đồ thị `depends_on` không thể đạt độ chính xác tuyệt đối (Markov equivalence, nhiễu tiềm ẩn — Okati et al.)
- Tính khả nghịch ANM mất hiệu lực khi chỉ số chạm trần vật lý (CPU=100%)
- Cơ chế dạng cây (nếu auto-assign chọn) không ngoại suy được ngoài phạm vi train — cần kiểm tra cơ chế thực tế từng node
- **[Xác nhận thực nghiệm 2025-09-08]** Ngay cả mechanism LinearRegression override (đã chọn để "an toàn" hơn tree-based) vẫn ngoại suy sai dấu tại do(workload) ≥ +150% (Flash Sale/Black Friday): một số node cho CPU/Memory delta ÂM trong khi workload tăng — vô lý về vật lý. Đã thêm cờ `extrapolation_suspect` trong `test_new_features_simulation.csv`; khoảng 48% số dòng ở 2 kịch bản cực đoan này bị đánh dấu không đáng tin, KHÔNG được trích dẫn làm "SCM dự báo đúng crash" nếu không diễn giải kèm giới hạn này.
- Kết quả RCA/causal nhạy với hệ thống/dataset (Pham 2024; khảo sát ACM) — RQ6 cần thiết, kết quả Sock Shop không mặc định đúng cho hệ khác
- Chồng 3 tầng suy luận (nhận dạng + mô phỏng SCM + Shapley) — không đảm bảo bảo toàn tính đúng tuyệt đối qua toàn pipeline
- Tách MAPE (CPU/Memory/Socket) khỏi F1-classification (Latency/Error) do bản chất toán học khác nhau (Kingman's formula — độ nhạy vô hạn gần điểm bão hòa)

---

## 9. Phụ lục A — Meta-model hình thức (tham khảo, không phải khung chính)

```
M = ⟨ E, A, R, C, T, P ⟩
E: Requirement, Microservice, Dependency, ResourceMetric, CodeChange, FeasibilityReport, RiskFactor, TraceLink
T: f_parse -> f_traverse -> f_normalize -> f_enrich -> f_project -> f_decide
```
Grounding chi tiết: TOSCA (Microservice/Dependency), OpenTelemetry Semantic Conventions (ResourceMetric), IEEE 29148 (Requirement), Gotel & Finkelstein 1994 (CodeChange/realizes). Dùng làm 1 chương ngắn nếu đề cương luận văn yêu cầu, không phải trọng tâm bài báo.

---
