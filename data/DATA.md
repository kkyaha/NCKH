# Bản đồ dữ liệu — `data/raw/`

> Sinh tự động: `python papers/p1_du_phong/experiments/collect/sinh_manifest_du_lieu.py`.
> Mọi vai trò **suy ra từ bằng chứng**, không gán tay. Chạy lại sau mỗi lần đo thêm.

| bằng chứng | mạnh cỡ nào |
|---|---|
| khớp lại cơ chế: `fit_mechanism` trên bộ nghi vấn trùng `mechanism` đã đóng băng, từng chữ số, 7/7 dịch vụ | mạnh nhất — chứng minh **kết quả** tái lập được, không chỉ byte giống nhau |
| khớp hash: `ramp_base_sha256` tính lại từ đĩa trùng fingerprint | mạnh — nhưng chỉ chứng minh byte giống nhau |
| cờ CLI `--train-dir` / `--ramp-dir` trong mã nguồn | trung bình — ý định, không phải xác minh |
| ô trong `p3_evaluation.csv` khớp thư mục chứa `ramp_<ô>` | yếu — ô `base` có ở **mọi** bộ, `cartsumx1` nằm ở hai bộ |
| cấu trúc thư mục (chỉ `ramp_base` / có `level_*` / không ramp) | yếu nhất, chỉ dùng khi không có gì khác |

## Không được đổi tên gì trong `data/raw/`

Đã đo: `FP.sha256_files` băm `basename(dirname(dirname(p)))` + nội dung tệp, nên **tên ramp**
(`ramp_base`, `level_150`) nằm trong hash, còn tên thư mục bộ dữ liệu thì không.

| đổi cái gì | hậu quả |
|---|---|
| `ramp_*`, `level_*` | `train_sha256` / `ramp_base_sha256` lệch → mất khả năng xác minh bằng hash |
| thư mục bộ dữ liệu | hash không đổi, nhưng chết mọi dòng CLI đã ghi trong docs/notebook và mọi đường dẫn ở đây |

## ⚠ `train_sha256` không tái lập được — và tại sao kết quả vẫn đứng

Hash train ghi trong **mọi** file đóng băng không khớp bất kỳ dữ liệu nào trên đĩa.
Đã kiểm và **loại** sáu giả thuyết:

| giả thuyết | kiểm bằng gì | kết quả |
|---|---|---|
| nội dung tệp đã đổi | mtime toàn bộ 16 tệp SS-TRAIN là 2026-09-21 19:50–20:58, **trước** lần đóng băng đầu (22:34) | loại |
| tập tệp đã đổi | `train_files: 16` khớp đúng 16 tệp trên đĩa | loại |
| hàm băm đã đổi | `git show <commit>:src/scm/feasibility_predictor.py` ở 3 commit đã ghi — `sha256_files` **giống nguyên văn** bản hiện tại | loại |
| dạng đường dẫn | thử chỉ-nội-dung, đường dẫn tương đối, đường dẫn tuyệt đối | loại |
| quy ước tên thư mục mức tải | thử `level_N`, `rps_N`, `N`, `load_N`, `ramp_N`, rỗng | loại |
| bộ train là bộ khác | tính hash cho cả 20 bộ có `simple_metrics.csv` | loại |

**Nguyên nhân: chưa xác định.** Nhưng điều mạnh hơn hash thì vẫn đúng: khớp lại
`fit_mechanism` trên `SS-TRAIN` hiện tại cho **đúng từng chữ số** (làm tròn 5) cả `alpha`
và `beta` của **7/7 dịch vụ** so với `mechanism` đã đóng băng, trên đúng 456 hàng
(tải ≤ 150 req/s). Hash chứng minh byte giống nhau; phép này chứng minh **kết quả tái
lập được** — đó mới là điều cần cho bài báo. Khi báo cáo, nói đúng như vậy: *cơ chế tái
lập chính xác, hash train hiện không dùng được làm phép xác minh*. Đừng viện hash train.

## Các bộ dữ liệu

| bộ dữ liệu | vai trò | tập đánh giá | base | tính năng | run | ngày đo | dung lượng |
|---|---|---|---|---|---|---|---|
| `RE2-OB` | ngoài (benchmark / trace, không phải phép đo của bài 1) | — | — | — | — | — | 881M |
| `RE2-SS` | ngoài (benchmark / trace, không phải phép đo của bài 1) | — | — | — | — | — | 2.2G |
| `SS-CALIBRATION` | hiệu chuẩn u* | — | — | — | — | — | 228K |
| `SS-LIMITS` | test | dev (kiem tra khong hoi quy), khoa (k~1 do duoc, ky vong khong doi) | ✓ | promox1, promox2, recsx1, recsx2, reviewx1… | 21 | 20260921 | 1.2G |
| `SS-LIMITS-C1` | test | — | ✓ | — | 3 | 20260921 | 23M |
| `SS-LIMITS-C2` | đã đo, chưa vào kết quả nào | — | ✓ | — | 1 | 20260921 | 46M |
| `SS-LIMITS-CLEAN` | test | DOC LAP (chinh, du lieu SACH), TIEN CUU (browse, du lieu SACH) | ✓ | browsex1, browsex2, cartsumx1, cartsumx2, expressx1… | 13 | 20260922 | 735M |
| `SS-LIMITS-INDEP` | test | DOC LAP (chinh, du lieu SACH) | ✓ | cartsumx1, cartsumx2, expressx1, expressx2, quickaddx1… | 16 | 20260922 | 683M |
| `SS-LIMITS-PROSP` | test | — | ✓ | — | 1 | 20260922 | 58M |
| `SS-LOADSWEEP` | đã đo, chưa vào kết quả nào | — | — | — | — | 20260921 | 272K |
| `SS-PROSP2/SS-PROSP2` | test | TIEN CUU 2 | ✓ | accountx1, catsearchx1, loginx1, loginx2, orderfullx1… | 51 | 20260925..20260926 | 1.9G |
| `SS-PROSP2/SS-PROSP2-R2` | test | TIEN CUU 2 | ✓ | accountx1, catsearchx1, loginx1, orderhistx1, previewx1… | 18 | 20260926 | 773M |
| `SS-PROSP2/SS-PROSP2-R3` | test | TIEN CUU 2 | ✓ | accountx1, catsearchx1, loginx1, orderhistx1, previewx1… | 18 | 20260926 | 381M |
| `SS-PROSP2/SS-PROSP4` | test | TIEN CUU 2 | ✓ | orderfullx1, reorderx1 | 9 | 20260925 | 404M |
| `SS-PROSP2/SS-PROSP4-CLEAN` | test | TIEN CUU 2 | ✓ | orderfullx1, reorderx1 | 13 | 20260925..20260926 | 776M |
| `SS-PROSP2/SS-PROSP4-OF` | test | TIEN CUU 2 | ✓ | orderfullx1 | 8 | 20260926 | 450M |
| `SS-PROSP2/SS-PROSP4-RE` | test | TIEN CUU 2 | ✓ | reorderx1 | 2 | 20260926 | 81M |
| `SS-PROSP2/SS-VERIFY2` | test | — | ✓ | — | 2 | 20260926 | 91M |
| `SS-PROSP2/SS-VERIFY3` | test | — | ✓ | — | 2 | 20260926 | 92M |
| `SS-PROSP2/SS-VERIFY4` | test | — | ✓ | — | 2 | 20260926 | 104M |
| `SS-TRAIN` | train (xác minh bằng cơ chế) | — | — | — | — | 20260921 | 431M |
| `SS-VERIFY` | đã đo, chưa vào kết quả nào | — | ✓ | — | 1 | 20260922 | 99M |
| `alibaba` | ngoài (benchmark / trace, không phải phép đo của bài 1) | — | — | — | — | — | 18M |
| `trainticket` | ngoài (benchmark / trace, không phải phép đo của bài 1) | — | — | — | — | — | 81M |

## Kết quả nào phụ thuộc bộ nào

Khớp bằng hash nội dung và khớp lại cơ chế, **không** khớp theo tên ô — ô `base` có trong
mọi bộ và `cartsumx1` nằm ở hai bộ, nên khớp tên sẽ gán sai. Các file `*_P3_*` không có
fingerprint riêng; chúng trỏ về cha qua `base_frozen_file` + `base_frozen_sha256`, và
chuỗi đó đã được nối vào đây. Xoá một bộ có tên dưới đây là làm các kết quả tương ứng
không còn tái lập được.

| bộ dữ liệu | file đóng băng phụ thuộc |
|---|---|
| `SS-LIMITS` | `predictions_frozen_RE2_P2.json`, `predictions_frozen_RE2_P2_indep.json`, `predictions_frozen_RE2_P2_prosp.json`, `predictions_frozen_RE2_P3_prosp_browse.json` |
| `SS-TRAIN` | `predictions_frozen_RE2.json`, `predictions_frozen_RE2_P2.json`, `predictions_frozen_RE2_P2_indep.json`, `predictions_frozen_RE2_P2_prosp.json`, `predictions_frozen_RE2_P2_prosp2.json`, `predictions_frozen_RE2_P2_prosp3.json`, `predictions_frozen_RE2_P2_prosp4.json`, `predictions_frozen_RE2_P3_prosp_account.json`, `predictions_frozen_RE2_P3_prosp_browse.json`, `predictions_frozen_RE2_P3_prosp_catsearch.json`, `predictions_frozen_RE2_P3_prosp_login.json`, `predictions_frozen_RE2_P3_prosp_orderfull.json`, `predictions_frozen_RE2_P3_prosp_orderhist.json`, `predictions_frozen_RE2_P3_prosp_preview.json`, `predictions_frozen_RE2_P3_prosp_register.json`, `predictions_frozen_RE2_P3_prosp_related.json`, `predictions_frozen_RE2_P3_prosp_reorder.json`, `predictions_frozen_RE2_P3_prosp_wishlist.json` |

### Liên kết yếu hơn — khớp qua ô dự đoán

Các bộ test còn lại không gán được bằng hash: frozen của chúng có `ramp_base_files: 0`
hoặc `ramp_base` không khớp. Khớp qua **ô dự đoán** thì gán được, nhưng yếu hơn — một ô
có thể nằm ở hai bộ. Mọi ô nhập nhằng đều được ghi rõ ở cột cuối.

| bộ dữ liệu | file đóng băng (qua ô) | ô nằm ở nhiều bộ |
|---|---|---|
| `SS-LIMITS` | `RE2.json` | — |
| `SS-LIMITS-CLEAN` | `RE2_P2_indep.json`, `RE2_P2_prosp.json`, `RE2_P3_prosp_browse.json` | cartsumx1, cartsumx2, expressx1, expressx2, quickaddx1, quickaddx2 |
| `SS-LIMITS-INDEP` | `RE2_P2_indep.json` | cartsumx1, cartsumx2, expressx1, expressx2, quickaddx1, quickaddx2 |
| `SS-PROSP2/SS-PROSP2` | `RE2_P2_prosp2.json`, `RE2_P2_prosp3.json`, `RE2_P2_prosp4.json`, `RE2_P3_prosp_account.json`, `RE2_P3_prosp_catsearch.json`, `RE2_P3_prosp_login.json` … (+7) | accountx1, catsearchx1, loginx1, orderfullx1, orderhistx1, previewx1, registerx1, relatedx1, reorderx1, wishlistx1 |
| `SS-PROSP2/SS-PROSP2-R2` | `RE2_P2_prosp2.json`, `RE2_P2_prosp3.json`, `RE2_P3_prosp_account.json`, `RE2_P3_prosp_catsearch.json`, `RE2_P3_prosp_login.json`, `RE2_P3_prosp_orderhist.json` … (+4) | accountx1, catsearchx1, loginx1, orderhistx1, previewx1, registerx1, relatedx1, wishlistx1 |
| `SS-PROSP2/SS-PROSP2-R3` | `RE2_P2_prosp2.json`, `RE2_P2_prosp3.json`, `RE2_P3_prosp_account.json`, `RE2_P3_prosp_catsearch.json`, `RE2_P3_prosp_login.json`, `RE2_P3_prosp_orderhist.json` … (+4) | accountx1, catsearchx1, loginx1, orderhistx1, previewx1, registerx1, relatedx1, wishlistx1 |
| `SS-PROSP2/SS-PROSP4` | `RE2_P2_prosp4.json`, `RE2_P3_prosp_orderfull.json`, `RE2_P3_prosp_reorder.json` | orderfullx1, reorderx1 |
| `SS-PROSP2/SS-PROSP4-CLEAN` | `RE2_P2_prosp4.json`, `RE2_P3_prosp_orderfull.json`, `RE2_P3_prosp_reorder.json` | orderfullx1, reorderx1 |
| `SS-PROSP2/SS-PROSP4-OF` | `RE2_P2_prosp4.json`, `RE2_P3_prosp_orderfull.json` | orderfullx1 |
| `SS-PROSP2/SS-PROSP4-RE` | `RE2_P2_prosp4.json`, `RE2_P3_prosp_reorder.json` | reorderx1 |

## Dữ liệu của bài 2 — nối với CSV kết quả, không với file đóng băng

Các file đóng băng là của **bài 1**; không file nào trỏ đến dữ liệu bài 2. Ánh xạ
`he` → thư mục đọc từ bảng `HE` trong `rq6_wrapper_rcaeval.py`, không gán tay. Bộ không có
cột `he` (như `alibaba`) được nối theo tiền tố tên tệp — suy luận theo tên, yếu hơn.
Số trong ngoặc là số dòng kết quả rút từ bộ đó.

| bộ dữ liệu | CSV kết quả (số dòng) |
|---|---|
| `RE2-OB` | `rq6_matched_grid.csv` (3240), `rq6_wrapper_rcaeval_v2_OnlineBoutique.csv` (356), `rq6_layer_selection.csv` (270), `rq6_thang_suy_bien.csv` (180) … tổng **6130 dòng** trong 28 tệp |
| `RE2-SS` | `rq6_matched_grid.csv` (3240), `rq6_wrapper_rcaeval_v2_SockShop.csv` (356), `rq6_layer_selection.csv` (270), `rq6_thang_suy_bien.csv` (180) … tổng **6158 dòng** trong 35 tệp |
| `alibaba` | `alibaba_forecast_eval.csv` (51720), `alibaba_node_load_tier.csv` (35896), `alibaba_per_instance_model_eval.csv` (17502), `alibaba_latency_queueing_test.csv` (14400) … tổng **131715 dòng** trong 8 tệp |
| `trainticket` | `rq6_matched_grid.csv` (3240), `rq6_tier_decomposition_v3.csv` (280), `rq6_layer_selection.csv` (270), `rq6_wrapper_rcaeval_v2_TrainTicket.csv` (182) … tổng **6050 dòng** trong 29 tệp |

## Bộ không có kết quả nào phụ thuộc

Đã đo nhưng chưa vào bất kỳ kết quả nào của cả hai bài — giữ để truy vết,
không dùng để công bố. Đây cũng là danh sách ứng viên nếu cần giải phóng dung lượng.

- `SS-CALIBRATION` — hiệu chuẩn u* (228K)
- `SS-LIMITS-C1` — test (23M)
- `SS-LIMITS-C2` — đã đo, chưa vào kết quả nào (46M)
- `SS-LIMITS-PROSP` — test (58M)
- `SS-LOADSWEEP` — đã đo, chưa vào kết quả nào (272K)
- `SS-PROSP2/SS-VERIFY2` — test (91M)
- `SS-PROSP2/SS-VERIFY3` — test (92M)
- `SS-PROSP2/SS-VERIFY4` — test (104M)
- `SS-VERIFY` — đã đo, chưa vào kết quả nào (99M)

Tổng: **9 bộ**.
