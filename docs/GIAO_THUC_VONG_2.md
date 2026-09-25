# Giao thức vòng đo TIẾN CỨU thứ hai — runbook

> **Mục tiêu:** đưa nhánh tiến cứu từ **n = 2** lên **n = 6–8** đơn vị độc lập.
> Đây là điểm yếu nặng nhất của bài: tài sản phương pháp luận mạnh nhất (đóng băng dự đoán
> trước khi đo) hiện chỉ đứng trên **hai** điểm dữ liệu.
>
> **Ước tính:** ~4–5 giờ máy. Chạy được trong một tối trên i5-12400F / 16 GB.

---

## ⚠️ QUY TẮC BẤT KHẢ XÂM PHẠM

**Thứ tự quyết định tất cả. Sai thứ tự thì cả vòng đo thành vô giá trị** — bạn sẽ có thêm
dữ liệu *hồi cứu*, thứ đã có thừa, chứ không có thêm bằng chứng nào.

```
  1. Agent MÙ cài tính năng
  2. Probe đo  (k, method, D_feat)   ← KHÔNG dùng một hạt dữ liệu tải nào
  3. ĐÓNG BĂNG + HASH dự đoán
  4. ───────── chỉ SAU khi có hash mới được chạy ramp ─────────
  5. Ramp
  6. So sánh
```

Nếu ở bước 5 phát hiện sai sót phải sửa mô hình → **vòng này hỏng**, kết quả chỉ được báo
cáo là hồi cứu. Không sửa lùi rồi coi như tiến cứu.

---

## BƯỚC 0 — Chuẩn bị máy (15 phút)

Sự cố **trôi điểm gãy** đã từng buộc phải chạy lại toàn bộ chiến dịch `SS-LIMITS-INDEP`.
Nguyên nhân **không phải thiếu CPU** (host chỉ dùng 26%) mà là tranh chấp tài nguyên tích
luỹ trong Docker Desktop/WSL sau phiên dài.

- [ ] Đóng mọi ứng dụng nặng (trình duyệt nhiều tab, IDE, game launcher)
- [ ] `wsl --shutdown`
- [ ] Khởi động lại Docker Desktop
- [ ] `docker compose -f deploy/sockshop/docker-compose.yml up -d` → kiểm `docker ps` thấy ~15 container
- [ ] `python experiments/collect/load_sweep_collect.py --check`
- [ ] Kiểm dữ liệu cũ **có mặt và đúng layout phẳng**: `data\raw\SS-TRAIN` (16 run) và
      `data\raw\SS-LIMITS` (21 run). Git **không** mang theo `data/raw/` — nó nằm trong
      `.gitignore`. Nếu còn lồng trong `SS_Deployed\` thì phải gỡ ra một cấp.
- [ ] Tạo thư mục **rỗng** `data\raw\SS-PROSP2` — bước đóng băng cần nó tồn tại và chưa có ramp
- [ ] Đặt biến cho gọn: `set F=login,register,...` (Windows) hoặc `F=login,register,...` (bash)

**Chốt chặn:** nếu `vm_cpu_util` vượt **0,85** ở bất kỳ bậc nào → dừng, restart, chạy lại
ramp đó. Đừng cố đo tiếp.

---

## BƯỚC 1 — Agent mù cài tính năng (dài nhất, làm trước)

### Chọn bao nhiêu tính năng

**6 tính năng × 1 cường độ** > 3 tính năng × 2 cường độ, dù cùng số ô.

Lý do: hai cường độ của **cùng** một tính năng dùng chung `k`, chung chain, chung bản cài —
chúng **không phải hai quan sát độc lập**. Reviewer sẽ gọi đó là pseudo-replication và tính
lại `n`. Sáu tính năng ở ×1 cho **6 đơn vị độc lập thật**.

Nếu còn thời gian, thêm ×2 cho **2 tính năng** để kiểm mô hình có bám đúng Δ không.

### Cách ly agent — bắt buộc

Agent cài đặt **chỉ được thấy**:
- mã front-end gốc
- danh mục API backend trung lập (không nhóm theo archetype)
- yêu cầu bằng ngôn ngữ tự nhiên + giao diện HTTP bắt buộc

Agent **không được thấy**: `SOCKSHOP_CALL_CHAINS`, bất kỳ file nào trong `data/processed/frozen/`,
`docs/HE_THONG.md`, `docs/DATA_FRAMEWORK.md`, `paper_draft.tex`, kết quả các vòng trước, và
**mã của các tính năng đã cài trước đó**.

- [ ] Cài trong thư mục cách ly **ngoài repo** (như `indep_sandbox` vòng trước)
- [ ] Quét sạch dấu vết taxonomy trước khi giao việc
- [ ] Agent **không được chạy hệ thống** — chỉ viết mã
- [ ] Lỗi chức năng thì chuyển lại cho agent, **không** chuyển gợi ý về thiết kế

> **Lưu ý:** `LOGIN` và `REGISTER` là hai archetype duy nhất chưa dùng. Bốn tính năng còn lại
> sẽ ánh xạ vào archetype đã có — **điều đó không sao**. Tính tiến cứu đến từ *dự đoán được
> đóng băng trước khi đo* và *người cài mù*, không đến từ archetype mới.

### Đăng ký tính năng — **BA chỗ trong mã, tất cả làm TRƯỚC khi probe**

Trước đây phải sửa 6 chỗ; ba chỗ còn lại nằm ở khâu *phân tích* nay đã nhận tham số dòng lệnh,
nên **không còn phải sửa mã sau khi đã đo** — điều đó sẽ phá tính tiến cứu của cả vòng.

Ba chỗ dưới đây là kiến thức miền, không tự suy ra được, nên vẫn phải khai báo tay.

**(a) `experiments/collect/load_sweep_collect.py`** — bảng `FEATURES` cắm cứng (~dòng 213):

```python
'login': {'req': ('POST', '/login/quick'), 'anchor': 10, 'archetype': 'LOGIN'},
```
`anchor` = `expected_delta_pct` của archetype trong `SOCKSHOP_CALL_CHAINS`.

**(b) `src/scm/feasibility_predictor.py`** — **chỉ một bảng**:

```python
FEATURE_ARCHETYPE = { ..., 'login': 'LOGIN', 'register': 'REGISTER', ... }
```

> `FEATURE_SETS` **không cần đụng tới nữa**. `freeze_predictions.py` và `evaluate_frozen.py`
> nay nhận `--features login,register,...` trực tiếp. Tuyệt đối không sửa `prosp` — đó là dấu
> vết của vòng `browse` đã đóng băng.

**(c) `experiments/collect/probe_feature_chain.py`** — hàm `body_for()`, nếu endpoint cần body:

```python
if feat == 'login':
    return {'username': ..., 'password': ...}
```

Chốt chặn có sẵn: nếu quên (b), `freeze_predictions.py` **từ chối chạy** với thông báo
`TU CHOI: [...] chua co trong FEATURE_ARCHETYPE`. Không có đường đi vòng — thiếu ánh xạ
archetype thì không có call chain để dự đoán.

- [ ] Chạy `pytest tests/ -q` — phải vẫn **50 passed**

---

## BƯỚC 2 — Probe (10 phút) · **KHÔNG dùng dữ liệu tải**

```bash
python experiments/collect/probe_feature_chain.py \
    --features $F \
    --n 20 --n-latency 200 --user-every 20 \
    --save data/processed/frozen/k_measured_v2.json
```

Probe đo ba thứ, **tất cả đều không cần tải**:

| | Dùng cho |
|---|---|
| `measured_per_use` — bội số gọi `k_s` | P3 |
| `per_use_by_method` — tách đọc/ghi | hiệu chỉnh chi phí ghi |
| `latency_idle` — `D_feat` (p99 khi hệ rảnh) | **cổng nhu cầu phục vụ** |

- [ ] **Lưu ra `k_measured_v2.json`**, KHÔNG ghi đè `k_measured.json` (file cũ là đầu vào của
      các phân tích đã đóng băng)
- [ ] Kiểm `chain_measured` so với `chain_taxonomy` — khác nhau là **dữ liệu**, không phải lỗi
- [ ] Kiểm `latency_idle.p99`: nếu đã > 250 ms thì tính năng đó **vỡ SLO ngay khi hệ rảnh**
      (như `express×2` vòng trước) — ghi nhận, vẫn đo

---

## BƯỚC 3 — ĐÓNG BĂNG (5 phút) · **cổng không thể quay lui**

Đặt `F=login,register,<4 tính năng còn lại>` cho gọn.

```bash
# P0/P1/P1_ctrl + P2 cho các tính năng mới
python experiments/feasibility/freeze_predictions.py \
    --train-dir data/raw/SS-TRAIN \
    --ramp-dir  data/raw/SS-PROSP2 \
    --limits RE2 \
    --feature-set prosp2 --features $F \
    --p2-params data/processed/frozen/p2_params_dev.json
# -> data/processed/frozen/predictions_frozen_RE2_P2_prosp2.json

# P3 (dùng k vừa probe) — lặp cho TỪNG tính năng
python experiments/feasibility/freeze_p3_prospective.py \
    --feature login \
    --k-file data/processed/frozen/k_measured_v2.json \
    --base-frozen data/processed/frozen/predictions_frozen_RE2_P2_prosp2.json \
    --ramp-dir data/raw/SS-PROSP2
```

⚠️ `--feature-set prosp2` là **nhãn đặt tên file**, `--features` mới là danh sách thật. Vì
nhãn khác `main`, script tự bật chốt chặn tiền đăng ký: bắt buộc `--p2-params`, và **từ chối
nếu `--ramp-dir` đã có ramp của các tính năng đó**.

Thư mục `data/raw/SS-PROSP2` phải **tồn tại và rỗng** ở bước này.

`freeze_p3_prospective.py` **tự từ chối** nếu `--ramp-dir` đã có dữ liệu ramp của tính năng
đó — chốt chặn có sẵn, đừng tìm cách đi vòng.

- [ ] **Ghi lại SHA-256 của từng file đóng băng vào sổ tay/commit ngay** — đây là chứng cứ
- [ ] `git add` + commit các file đóng băng **trước khi chạy ramp** (dấu thời gian của git là
      bằng chứng độc lập với hash)

> ⚠️ **Cổng nhu cầu phục vụ và cổng throttling hiện CHƯA có script đóng băng riêng.** Nếu
> muốn kiểm chứng tiến cứu cả hai cổng này, phải viết bước đóng băng cho chúng **trước** khi
> đo. Nếu không kịp, chỉ báo cáo P1/P2/P3 là tiến cứu và ghi rõ hai cổng vẫn là hồi cứu.

---

## BƯỚC 4 — Ramp (~4 giờ)

### Lưới tải: dùng **hình học**, không phải số học

Lưới 20 req/s tuyệt đối của các vòng trước có khuyết tật: độ phân giải tương đối **10%** ở
điểm gãy 200 nhưng **50%** ở điểm gãy 40. Sai số tương đối vì thế **không so sánh được giữa
các ô** — `quickadd×2` bị thổi phồng một phần vì lý do này.

Bậc **≈ ×1,25**: `40, 50, 65, 80, 100, 125, 155, 195, 240` → độ phân giải tương đối **~25%
đồng đều** ở mọi mức, thay vì 10% ở điểm gãy 200 và 50% ở điểm gãy 40.

Truyền bằng `--ramp-levels`; cờ này **đè lên** `--ramp-start/stop/step`.

### Lệnh

```bash
python experiments/collect/load_sweep_collect.py --ramp \
    --limits RE2 \
    --features base,$F \
    --feature-scales 1 \
    --ramp-levels 40,50,65,80,100,125,155,195,240 \
    --repeats 2 --base-repeats 2 \
    --interval 3 --cooldown 60 \
    --out-dir data/raw/SS-PROSP2
```

Lưới được ghi tự động vào `ramp_manifest_*.json` (`steps_rps` và `args`) — không phải chép tay.

### Chống trôi — quan trọng ngang phép đo

- [x] ~~Xen ramp `base` giữa các tính năng~~ — **đã tự động**: script trộn kế hoạch bằng
      `random.Random(--seed).shuffle(plan)`, nên `base` rải đều. Không phải làm gì.
- [ ] Sau mỗi ~2 giờ: dừng, `wsl --shutdown`, restart Docker Desktop, chạy lại một ramp `base`
      để xác nhận điểm gãy chưa trôi
- [ ] **Điều kiện huỷ:** nếu điểm gãy baseline lệch quá **một bậc lưới** so với đầu phiên →
      dừng, xử lý máy, đo lại từ đầu. Đừng cố cứu dữ liệu đã nhiễu.

---

## BƯỚC 5 — Kiểm hợp đồng dữ liệu (5 phút)

```bash
python experiments/collect/data_contract_check.py --dir data/raw/SS-PROSP2 --role limits
```

- [ ] Mục tiêu **0 FAIL**
- [ ] Nếu có ramp thiếu `steps.json` → đó là lượt đo dở dang, xoá thư mục `runN` đó

---

## BƯỚC 6 — Đánh giá

```bash
# P0/P1/P1_ctrl/P2 trên tính năng MỚI
python experiments/feasibility/evaluate_frozen.py \
    --frozen data/processed/frozen/predictions_frozen_RE2_P2_prosp2.json \
    --ramp-dir data/raw/SS-PROSP2 \
    --split prosp2 --features $F

# P3 — thêm tập mới, KHÔNG động vào 4 tập cũ
python experiments/feasibility/evaluate_p3.py \
    --k-file data/processed/frozen/k_measured_v2.json \
    --extra-set "TIEN CUU 2:SS-PROSP2:$F"

# Thống kê + khoảng tin cậy bootstrap
python experiments/model_eval/statistical_rigor.py \
    --roots SS-LIMITS,SS-LIMITS-CLEAN,SS-PROSP2 \
    --k-file data/processed/frozen/k_measured_v2.json \
    --extra-split "tien cuu 2=$F"
```

⚠️ **Thiếu `--features` / `--extra-set` / `--extra-split` là hỏng âm thầm.** `evaluate_frozen.py`
sẽ báo lỗi (tốt), nhưng hai script kia **vẫn chạy trót lọt và in lại số của vòng cũ** — không
có dấu hiệu gì báo rằng dữ liệu mới chưa hề được chấm.

Mốc hồi quy: chạy `evaluate_p3.py` **không cờ nào** phải in đúng `119.7 / 87.5 / 72.6` cho tập
`DOC LAP`. Lệch nghĩa là có gì đó đã đổi ngoài ý muốn — dừng lại tìm nguyên nhân.

**Báo cáo bắt buộc gồm:**

| | Vì sao |
|---|---|
| Sai số **tuyệt đối** (req/s) **và** tương đối (%) | tương đối không so được giữa các ô |
| **Hướng nguy hiểm** riêng | báo khả thi giả là lỗi tốn kém nhất |
| Khoảng tin cậy bootstrap | với n = 6, khoảng sẽ rộng — cứ báo cáo |
| Hash của từng file đóng băng | chứng cứ tiền đăng ký |

---

## Tiêu chí thành công

| | Ngưỡng |
|---|---|
| Số ô tiến cứu | **≥ 6 đơn vị độc lập** (hiện: 2) |
| Ô báo khả thi giả | **0** (vòng `browse` đạt 0/2) |
| Điểm gãy baseline | không trôi quá một bậc lưới suốt phiên |
| `data_contract_check` | 0 FAIL |

**Kết quả âm vẫn là thành công.** Nếu sai số lớn hơn vòng `browse`, đó là thông tin thật — và
vì dự đoán đã đóng băng, nó là bằng chứng *tiến cứu* về giới hạn của mô hình. Điều đó **giá
trị hơn** một con số đẹp thu được bằng cách chỉnh mô hình sau khi xem đáp án.

---

## Rủi ro đã biết

| Rủi ro | Xử lý |
|---|---|
| Hết archetype chưa dùng (chỉ còn `LOGIN`, `REGISTER`) | Dùng lại archetype cũ với **bản cài mới** — vẫn hợp lệ |
| Agent vô tình thấy taxonomy | Quét sạch sandbox trước; nếu nghi ngờ thì **loại tính năng đó** khỏi nhánh tiến cứu |
| Trôi hiệu năng giữa phiên | Ramp `base` xen kẽ + restart định kỳ |
| Tính năng vỡ SLO ngay bậc đầu (như `express×2`) | Ghi nhận, giữ trong dữ liệu, **loại khỏi phép tính sai số** (không có điểm gãy đo được) |
| Hai cổng mới chưa có script đóng băng | Chỉ tuyên bố tiến cứu cho P1/P2/P3; ghi rõ hai cổng là hồi cứu |
| **Quên cờ ở Bước 6** → in lại số vòng cũ, không báo lỗi | Đối chiếu mốc hồi quy `119.7/87.5/72.6`; nếu bảng không có dòng `TIEN CUU 2` thì cờ đã bị bỏ sót |
| Sửa mã phân tích giữa chiến dịch | Đã dọn trước khi bàn giao. Nếu vẫn phát sinh: **ghi lại và báo cáo là hồi cứu**, đừng sửa lặng lẽ |
