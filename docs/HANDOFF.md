# HANDOFF — bàn giao phiên làm việc 2026-09-19 → 21

> **Đọc file này trước tiên nếu bạn là một phiên Claude Code mới.** Nó chứa toàn bộ
> quyết định, số liệu đã đo, và các ngõ cụt đã loại — đọc xong thì **không cần chạy
> lại** các thí nghiệm tốn kém bên dưới.
>
> Nguồn số liệu chính thức vẫn là `docs/paper_draft.tex`. File này bổ sung phần
> **chưa** vào paper.

---

## 0. Việc phải làm đầu tiên khi đổi máy

```bash
git clone https://github.com/kkyaha/NCKH.git && cd NCKH
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Dữ liệu: cho việc LOAD SWEEP thì KHÔNG cần tải gì cả

`data/raw/` bị gitignore nên không bộ nào đi theo git — nhưng **mục đích đổi máy là dựng
Sock Shop để SINH dữ liệu mới**, nên không cần bộ cũ nào. Chỉ cần code + Docker.

Chỉ tải lại nếu muốn **chạy lại các thí nghiệm phân tích** (mọi kết quả đã có sẵn ở Mục 3,
thường không cần chạy lại):

| Bộ | Dung lượng | Cách lấy | Cần cho |
|---|---|---|---|
| RE2-OB | 881 MB | `python experiments/download_onlineboutique_data.py --all` | chạy `CapacityAgent` trên OB |
| Train Ticket | 81 MB | `python experiments/download_trainticket_data.py --all` | RQ1/RQ2 Train Ticket |
| RE2-SS | 2.2 GB | HuggingFace `phamquiluan/RCAEval`, tiền tố `re2ss_*` → `data/raw/RE2-SS/<scenario>/<run>/` | RQ1–RQ5 Sock Shop |
| Alibaba v2021 | 69 GB | `MSRTQps_{0..24}` + `MSResource_{0..11}` từ `aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces`; `export ALIBABA_DIR=<thư mục>` | `alibaba_*`, `scm_mechanism_bakeoff` |

---

## 1. Dự án đang làm gì (30 giây)

Đánh giá **tính khả thi** của một yêu cầu tính năng mới (ngôn ngữ tự nhiên) trên kiến
trúc vi dịch vụ đang chạy: Parser Agent dịch yêu cầu thành can thiệp `do(x)` có kiểm
chứng 3 lớp → lan truyền qua Global DAG (SCM) → Capacity Agent dự phóng tài nguyên →
`FeasibilityReport`.

Ba hệ thử: **Sock Shop** (7 service), **Train Ticket** (28), **Online Boutique** (7 —
mới thêm trong phiên này), cộng **Alibaba v2021** (1.293 service, dữ liệu sản xuất thật).

---

## 2. Code đã đổi trong phiên (CHƯA COMMIT — phải push trước khi đổi máy)

### Sửa
| File | Thay đổi | Lý do |
|---|---|---|
| `src/agents/capacity_agent.py` | `simulate_intervention()` bỏ `gcm.interventional_samples()`, dùng `deterministic_forward()` | do(X)=condition(X) tại gateway (RQ5b). Nhanh hơn 13.6–14.0×, và **sửa bug bất định**: cùng input cho kết quả lệch tới 35.9% giữa 2 lần gọi |
| `src/agents/capacity_agent.py` | Thêm `compute_headroom()` + `summarize_headroom()` + `_ceiling_ci()`; `assess_capacity()` phán quyết theo **biên dự phòng** thay vì ngưỡng tuyệt đối | Ngưỡng cũ `cpu_pred > 70.0` giả định CPU là phần trăm — dữ liệu là core/byte nên **gần như không bao giờ kích hoạt** |
| `src/agents/capacity_agent.py` | Chốt an toàn trong `train_accurate_path()`: cảnh báo khi gateway/node thiếu cột metric | Gateway từng bị loại âm thầm khỏi DAG (xem Bẫy #1) |
| `src/scm/queueing_regressor.py` | `capacity_` từ `max(X)*1.5` → `percentile(X,99)*1.5`, thêm chặn X ở bước fit | `max` không ổn định theo cỡ mẫu — đã làm sụp bằng chứng Proposition 3 trên Alibaba khi mở cửa sổ 5h→10h |
| `src/scm/taxonomies/__init__.py` | Đăng ký `SockShop_LogMined`, `OnlineBoutique` | |

### Thêm mới
- `src/graph/onlineboutique_agent_graph.json` — graph OB (7 node, 9 cạnh), **đã sửa tay 2 chỗ**, xem Bẫy #1 và #2
- `src/scm/taxonomies/onlineboutique.py` — 4 archetype, phủ 7/7 node
- `src/scm/taxonomies/sockshop_from_logs.py` — CALL_CHAINS mine từ log (tuỳ chọn, **không** thay mặc định)
- `experiments/download_onlineboutique_data.py`
- `experiments/elasticity_transfer_loso.py`, `scm_mechanism_bakeoff.py`,
  `alibaba_latency_queueing_test.py`, `alibaba_workload_per_instance_test.py`
- `experiments/load_sweep_collect.py` — **chưa chạy được lần nào** (máy cũ không có Docker)

**Kiểm chứng:** `python -m pytest tests/ -q` → 4/4 pass sau mọi thay đổi trên.

---

## 3. Số liệu đã đo — KHÔNG cần chạy lại

### 3.1. Vấn đề gốc: dữ liệu RCAEval không có tín hiệu
Signal gate R² của `workload → CPU` (dữ liệu giai đoạn bình thường, gộp mọi run):

| Hệ | R² trung vị | % service R²>0.3 |
|---|---|---|
| Sock Shop | 0.015 | 0% |
| Train Ticket | 0.043 | 0% |
| Online Boutique | 0.042 | 0% |
| **Alibaba** | **0.102–0.22** | **30%** |

→ Trên chính 3 hệ tool được triển khai, **độ chính xác không đo được** (không phải
"thấp"). Thí nghiệm leave-one-system-out (`elasticity_transfer_loso.py`) xác nhận:
mọi nhánh đều ~0.0005 và **negative control không tách khỏi mô hình thật** (53.3% vs
53.3%) → phép đánh giá **vacuous** theo Definition 1 của chính bài.

Nguyên nhân là **thiết kế benchmark**: RCAEval cố tình giữ tải phẳng để lỗi tiêm vào
nổi bật — xung khắc trực tiếp với capacity forecasting. Thêm hệ thứ ba không cứu được.

### 3.2. Bake-off chọn cơ chế (Alibaba, OOD, n=158 cặp ghép, `scm_mechanism_bakeoff.py`)

| Mô hình | skill trung vị | Δ vs NNLS | Thắng NNLS | Wilcoxon p |
|---|---|---|---|---|
| LinearReg | 0.3753 | 0.0000 | 30.4% | 0.0018 * |
| **ElasticityPooled** | 0.3730 | **+0.0092** | **60.8%** | **0.0005 \*** |
| ElasticityPrior | 0.3603 | −0.0379 | 43.7% | 0.354 (ns) |
| *NNLS_deployed (mốc)* | *0.3569* | — | — | — |
| GradBoost | 0.2530 | −0.157 | 35.4% | 0.0059 * |
| Queueing | 0.2355 | −0.0000 | 27.2% | 0.0000 * |

Kết luận: giữ **NNLS** (ràng buộc không âm gần như miễn phí — nơi hai mô hình khác nhau
là nơi OLS cho hệ số âm vô lý), **partial pooling** là cải thiện có ý nghĩa duy nhất,
**bỏ GradBoost**, **không dùng dạng queueing cho CPU**.

### 3.3. Latency không dự báo được (`alibaba_latency_queueing_test.py`)
Signal gate workload→latency **tốt hơn cả CPU** (R² 0.166, 38.6% >0.3) nên phép kiểm hợp lệ.

| | indist | ood | ood_tail |
|---|---|---|---|
| LinearReg (không ràng buộc) | 0.448 | 0.331 | 0.316 |
| **Queueing / NNLS / Elasticity** | **0.0000** | **0.0000** | **0.0000** |

Nguyên nhân đo trực tiếp: **50.9% service sản xuất thật có hệ số dốc workload→latency ÂM**
(47.4% trên nhóm có tín hiệu). Với CPU chỉ 9.2% (2.1%). ⇒ `positive=True` triệt tiêu cơ
chế latency ở ~nửa số service. Đây **không phải đặc thù Train Ticket** (README Giới hạn
#7) mà là hệ thống.

**Hệ quả cho Proposition 2**: tiền đề đơn điệu đúng với CPU/Mem (97.9% dốc dương), **sai
với latency**. Chứng chỉ latency hoặc rỗng nội dung (cơ chế dự báo hằng số) hoặc phải bỏ
ràng buộc (mất Prop 2 cho latency).

### 3.4. Giả thuyết autoscaling — ĐÃ BÁC BỎ (`alibaba_workload_per_instance_test.py`)
`corr(workload, n_inst)` trung vị = **0.0047**, chỉ 1.0% service có corr>0.5 → không có
autoscaling trong cửa sổ dữ liệu. Chuẩn hoá theo instance **làm xấu đi** (R² 0.192→0.053,
`raw_total` thắng `per_inst` ở 82.5% service, p≈6e-18). **Đừng thử lại hướng này.**

### 3.5. Call-chain mining từ log
`src/scm/callchain_from_logs.py` mine được `services` của archetype từ access-log.
Kiểm trên 90 run RE2-SS: **Jaccard trung bình 0.60** (REGISTER 0.67, GET_CATALOGUE 0.67,
ADD_TO_CART 0.40, PLACE_ORDER 0.67).

**Train Ticket có `logs.parquet`** (script download hiện KHÔNG tải file này) nhưng là log
nghiệp vụ Spring Boot, **0/197.087 dòng khớp regex access-log** → phương pháp này không áp
dụng được. **Đã chốt: dừng đầu tư, dùng BFS-from-seed cho Train Ticket.**

---

## 4. Quyết định đã chốt

| Quyết định | Lý do |
|---|---|
| Giữ Alibaba trong dự án | Bằng chứng mạnh nhất (RQ4 n=1.293, RQ5a n=27 cạnh); không có bộ thay thế |
| Bỏ `do()` khỏi đường dự đoán chính | do(X)=condition(X) tại gateway in-degree=0 (RQ5b) — máy do-calculus chỉ tốn thêm |
| Đầu ra là **biên dự phòng**, không phải con số latency | Trần năng lực C không định danh được từ dữ liệu chưa bão hoà |
| `SockShop_LogMined` là tuỳ chọn, không thay mặc định | Jaccard 0.40–0.67 không đồng đều |
| Không rename `RE2-SS`/`trainticket` | 72 tham chiếu trong 30 file, nhiều file là hạ tầng đông cứng sinh số liệu paper |

### Đã cân nhắc và LOẠI (đừng làm lại)
- **Dạng hàm log-log elasticity** — không tìm được tiền lệ cho microservice, và **xung đột
  với nền Kingman/queueing** đã trích dẫn trong bài. Partial pooling + conformal thì có nền
  vững (James–Stein, Efron–Morris; split conformal).
- **Lấy trần C từ K8s resource limit** — không dataset nào trong 4 bộ có, làm mô hình không
  đánh giá được trên chính benchmark đang dùng.
- **`CEILING_TAIL_RATIO_MAX = 10.0`** — từng đặt rồi bỏ: đo phân bố `max/P99` trên 2 hệ cho
  thấy liên tục (P50=1.67, P90=5.2, P95=26.3), **không có vách ngăn tự nhiên tại 10**. Thay
  bằng khoảng tin cậy bootstrap + trạng thái `UNDECIDED`.

---

## 5. Việc tiếp theo: LOAD SWEEP (đây là lý do đổi máy)

**Mục tiêu**: sinh bộ dữ liệu không bộ công khai nào có — **tên service thật + biên độ tải
thật** — để biến "không đo được độ chính xác" thành "đo được", và mở đường cho validation
tiến cứu (dự báo → deploy → đo lại), thứ duy nhất cho phép tuyên bố độ chính xác.

### Chọn hệ
**Sock Shop** (không phải Online Boutique):
- OB **đã bỏ `docker-compose`** (k8s-first) → cần thêm minikube/kind/kubectl
- Sock Shop còn `docker-compose.yml` (HTTP 200, 15 service)
- `front-end` của Sock Shop ghi **access-log** `METHOD /path STATUS DURATION ms` → lấy được
  workload + latency **không cần Prometheus/Jaeger**
- Cùng hệ với RE2-SS đã có → dữ liệu mới **so sánh trực tiếp được** với dữ liệu
  fault-injection cũ (cùng tên service, cùng topology, cùng taxonomy)

### Kiến trúc — kiểm tra ĐẦU TIÊN trên máy mới
Máy cũ là **Apple M5 (arm64)** còn ảnh Sock Shop là **single-arch amd64** (Weaveworks xuất
bản 2017–2020, công ty đã đóng cửa 2024) → phải giả lập, chậm 2–5×, bão hoà host sớm, và
số CPU đo được phản ánh chi phí giả lập.

```bash
uname -m                                   # PC x86_64 thì hết vấn đề này
docker manifest inspect weaveworksdemos/front-end:0.3.12 | grep architecture
```

**Nếu PC là x86_64 → chạy thẳng, không còn rào cản kiến trúc.**

### Các bước
```bash
# 1. Dựng
curl -sL https://raw.githubusercontent.com/microservices-demo/microservices-demo/master/deploy/docker-compose/docker-compose.yml -o docker-compose.yml
docker compose up -d
docker ps                                   # phải thấy ~15 container

# 2. Kiểm môi trường của harness
python experiments/load_sweep_collect.py --check

# 3. Quét tải (mỗi mức 10 phút; chạy thử 2 mức ngắn trước khi chạy full)
python experiments/load_sweep_collect.py --levels 5,10 --hold 120 --interval 10
python experiments/load_sweep_collect.py --levels 5,10,25,50,100,200 --hold 600
```

Đầu ra: `data/raw/SS-LOADSWEEP/level_<N>/metrics.csv`, **đúng quy ước cột của repo**
(`<svc>_workload`/`_cpu`/`_mem`/`_latency-50`) nên `load_multi_service_data()` đọc được
không cần sửa gì.

### Tiêu chí thành công — kiểm NGAY sau khi quét
```bash
# Signal gate: R^2 workload->CPU phải CAO HƠN HẲN 0.015 (mức RE2-SS hiện tại)
python experiments/elasticity_transfer_loso.py     # sau khi thêm SS-LOADSWEEP vào SYSTEMS
```
Hệ số biến thiên workload của RE2-SS hiện tại ≈ **0.18**; cần cao hơn rõ rệt. Nếu R² vẫn
~0.02 thì load sweep chưa đủ rộng — tăng dải mức tải, đừng đổi mô hình.

### Hai mối đe doạ tính hợp lệ, phải xử lý
1. **Load generator tranh CPU với hệ đang đo** — lý tưởng là chạy loadgen ở máy khác. Tối
   thiểu: ghi lại CPU toàn máy để phát hiện và loại mẫu bị nhiễm.
2. **Giả lập kiến trúc** (nếu PC vẫn là arm64) — phải khai báo trong Threats to Validity.

### `experiments/load_sweep_collect.py` — CHƯA KIỂM CHỨNG
Mới chạy được `--check`. Phần `docker stats`/`docker logs` là bản thảo viết mù. Soát kỹ:
- Tên container thật có khớp tên service trong graph không (`docker ps --format '{{.Names}}'`)
- Regex access-log có khớp log thật của `front-end` không
- Lệnh `--loadgen` mặc định (`weaveworksdemos/load-test`) có chạy không

---

## 6. Bẫy đã biết

1. **Gateway bị loại âm thầm khỏi DAG.** Bảng metrics của OB gọi gateway là `frontend`,
   graph (trích từ trace, cột `serviceName`) gọi `frontendservice`. Hệ vẫn chạy trọn và in
   `[OK] Đã khớp Global DAG 24 nodes` trong khi **thiếu hẳn điểm vào**. Đã sửa + thêm chốt
   cảnh báo, nhưng **kiểm lại tên service giữa graph và metrics mỗi khi thêm hệ mới**.

2. **`extract_graph_generic.py` gán nhầm service thành `database`.** Heuristic
   "sink (out-degree=0) ⇒ database" đúng với SockShop/Train Ticket (sink là Mongo/MySQL
   thật) nhưng **sai với OB**: trace không ghi lời gọi datastore nên 4 service ứng dụng thật
   bị gán nhầm → `bfs_closure()` loại chúng khỏi mọi archetype. Đã sửa tay trong JSON.

3. **`compute_headroom` từng so trung bình với P99.** Hai họ thống kê khác nhau → với phân
   phối lệch phải nặng, trung bình có thể vượt P99 → biên âm → CRITICAL giả. Đã sửa thành
   khoảng **P50→P99**. Đừng quay lại dùng mean làm mốc dưới.

4. **MAPE thấp không có nghĩa là mô hình tốt.** Target gần như bất biến cho MAPE thấp dưới
   mọi predictor. Luôn dùng **skill score + negative control**, không dùng MAPE một mình.

5. **Đừng sửa `experiments/evaluation_suite.py`, `select_scm_edges.py`,
   `backpressure_edge_ood_safety_test.py`** — là các bản sao đông cứng có chủ đích, sinh số
   liệu đã công bố.

---

## 7. Còn tồn đọng (ngoài load sweep)

1. **Validation tiến cứu** — chưa từng làm: dự báo → deploy thật → đo lại. Là thứ **duy
   nhất** cho phép tuyên bố độ chính xác. Load sweep mở đường cho việc này.
2. **Đầu ra biên dự phòng chưa có bằng chứng nào** — chưa có thí nghiệm nào cho thấy "tiêu
   71% biên" tương ứng điều gì có thật. Mới xây trong phiên này.
3. **Backend sweep vẫn n=2** — điểm yếu lớn nhất của claim trung tâm RQ1.
4. **Trục cỡ mẫu chỉ có 1 điểm** (n_train=30): workload và resource của Alibaba chỉ chồng
   lấn 60 mốc thời gian bất kể đọc bao nhiêu file.
5. **Proposition 3 trên Alibaba chưa test lại** sau khi sửa `capacity_` — script gốc
   (cửa sổ 5h→10h) không có trong repo, phải viết mới.
6. **README chưa cập nhật** theo các thay đổi phiên này (Giới hạn #7 giờ có bằng chứng
   Alibaba; `capacity_` đã đổi; thêm hệ thứ 3).
