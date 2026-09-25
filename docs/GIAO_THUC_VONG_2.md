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

### Đăng ký tính năng — **BA chỗ trong mã, thiếu một là hỏng giữa chừng**

Đây là bước dễ bị bỏ sót nhất. Làm đủ cả ba **trước** khi probe.

**(a) `experiments/collect/load_sweep_collect.py`** — bảng `FEATURES` cắm cứng (~dòng 213):

```python
'login': {'req': ('POST', '/login/quick'), 'anchor': 10, 'archetype': 'LOGIN'},
```
`anchor` = `expected_delta_pct` của archetype trong `SOCKSHOP_CALL_CHAINS`.

**(b) `src/scm/feasibility_predictor.py`** — HAI bảng:

```python
FEATURE_ARCHETYPE = { ..., 'login': 'LOGIN', 'register': 'REGISTER', ... }

FEATURE_SETS = { ..., 'prosp2': ['login', 'register', ...] }   # tập MỚI, đừng sửa 'prosp'
```

⚠️ `freeze_predictions.py --feature-set` chỉ nhận các khoá có trong `FEATURE_SETS`. Hiện là
`{main, indep, prosp}` — **`prosp2` chưa tồn tại**, phải thêm. Tạo **tập mới**, tuyệt đối
không sửa `prosp` (đó là dấu vết của vòng `browse` đã đóng băng).

**(c) `experiments/collect/probe_feature_chain.py`** — hàm `body_for()`, nếu endpoint cần body:

```python
if feat == 'login':
    return {'username': ..., 'password': ...}
```

- [ ] Sau khi sửa: `python -c "import sys; sys.path.insert(0,'src/scm'); import feasibility_predictor as F; print(F.FEATURE_SETS['prosp2'])"`
- [ ] Chạy `pytest tests/ -q` — phải vẫn **50 passed**

---

## BƯỚC 2 — Probe (10 phút) · **KHÔNG dùng dữ liệu tải**

```bash
python experiments/collect/probe_feature_chain.py \
    --features login,register,<4 tính năng còn lại> \
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

```bash
# P0/P1/P1_ctrl + P2 cho các tính năng mới
python experiments/feasibility/freeze_predictions.py --feature-set prosp2 ...

# P3 (dùng k vừa probe)
python experiments/feasibility/freeze_p3_prospective.py \
    --feature login \
    --k-file data/processed/frozen/k_measured_v2.json \
    --base-frozen data/processed/frozen/predictions_frozen_RE2_P2_prosp.json \
    --ramp-dir data/raw/SS-PROSP2
# lặp cho từng tính năng
```

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

```
--ramp-start 40 --ramp-stop 260 --ramp-step 20     # cũ, giữ để so sánh
```

Đề xuất thay bằng các bậc **≈ ×1,25**: `40, 50, 65, 80, 100, 125, 155, 195, 240`
→ độ phân giải tương đối **~25% đồng đều** ở mọi mức.

- [ ] Ghi rõ trong manifest là vòng này dùng lưới khác vòng trước

### Lệnh

```bash
python experiments/collect/load_sweep_collect.py --ramp \
    --limits RE2 \
    --features base,login,register,<4 tính năng còn lại> \
    --feature-scales 1 \
    --repeats 2 --base-repeats 2 \
    --interval 3 --cooldown 60 \
    --out-dir data/raw/SS-PROSP2
```

### Chống trôi — quan trọng ngang phép đo

- [ ] **Xen ramp `base` giữa các tính năng** (không dồn hết `base` vào đầu). Nếu điểm gãy
      baseline trôi thì phát hiện **ngay**, không phải sau khi đo xong tất cả.
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
python experiments/feasibility/evaluate_frozen.py \
    --frozen data/processed/frozen/predictions_frozen_RE2.json \
    --ramp-dir data/raw/SS-PROSP2 --split prosp

python experiments/feasibility/evaluate_p3.py
python experiments/model_eval/statistical_rigor.py
```

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
