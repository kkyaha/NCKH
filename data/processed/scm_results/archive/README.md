# `archive/` — CSV kết quả không còn script nào sinh ra / không tài liệu nào trích dẫn

Giữ nguyên văn, **không xoá**. Ba file `rq6_doi_thang_trong_baro.csv`,
`rq6_vathang_baro.csv`, `rq6_uoc_luong_thang.csv` đã ở đây từ trước (xem
`papers/p2_khop_tang/experiments/archive/README.md` — script sinh ra chúng cho
**kết quả sai**, giữ làm vết).

Phần dưới đây thêm **2026-10-15**, dọn gọn codebase theo yêu cầu người dùng. Mỗi file qua
**hai lớp kiểm trước khi chuyển**, không gỡ theo cảm tính:
1. Grep toàn bộ `.py` hiện có trong repo cho tên file — không script nào ghi `to_csv` ra tên
   này (script sinh ra nó, nếu có, đã bị xoá/đổi tên ở một lần refactor trước, không còn
   trong cây hiện tại).
2. Grep toàn bộ `.md`/`.tex` cho tên file (kể cả so khớp tiền tố, để bắt pattern dạng
   `ten_file*.csv` trong README) — không tài liệu nào trích dẫn.

**Cảnh báo đã tự sửa khi audit (ghi lại để không lặp lại)**: hai vòng kiểm bằng grep tên
file literal đã bỏ sót cách sinh tên qua **f-string** (`f'tien_to_{bien}.csv'`), suýt
archive nhầm 6 file đang SỐNG: `canh_do_chinh_xac_sockshop_canh/mem_socket.csv`,
`..._trainticket_canh/da_cha.csv` (do `edges/canh_do_chinh_xac.py` sinh qua
`f'canh_do_chinh_xac_{a.he}_{a.che_do}.csv'`, script này được `HE_THONG.md` và
`experiments/README.md` trích dẫn tên) và `scm_edge_selection_sockshop/trainticket.csv`
(do `edges/select_scm_edges.py` sinh qua `f'scm_edge_selection_{args.system}.csv'`, cũng
được 2 tài liệu đó trích dẫn). Cả 6 file đã **phục hồi về `data/processed/scm_results/`**,
không nằm trong danh sách dưới. Hai lỗi dương giả khác bắt được ở vòng kiểm đầu: (a)
`README.md` gốc dùng pattern glob `scope_gate_embedding_calibration*.csv` nên các file đó
được giữ lại; (b) `bang_ket_qua_5_vong.py` không có tên riêng trong runbook nhưng **ghi**
`bang_ket_qua_5_vong.csv` — nguồn bảng kết quả chính của `papers/p1_du_phong/README.md`.

Mỗi file dưới đây đã qua **ba lớp kiểm**: (1) grep tên file literal trong mọi `.py`/`.md`/
`.tex`, (2) grep tiền tố glob-kiểu (bắt pattern `ten*.csv` trong README), (3) grep tiền tố
kết hợp với dòng có `to_csv`/f-string (bắt f-string construction) — **và với mỗi khớp ở
lớp 3, đã đọc trực tiếp script để xác nhận đó là script SỐNG hay đã nằm trong `archive/`**
(script sống → không archive output của nó, như 6 file ở trên; script đã archive → output
đi theo, như 2 nhóm `mechanism_trial_fastpath_*`/`saturating_mechanism_trial_*` dưới đây).

| file | nhóm | lý do coi là mồ côi |
|---|---|---|
| `_crca_done.csv` | checkpoint tiến độ | tên gợi ý file cờ "đã xong" của một batch job cũ, không script nào đọc/ghi lại |
| `g2g3_fresh_no_guard_capture.csv` | guard G2/G3 | capture tạm, không script/tài liệu nào nhắc |
| `g7_rq3_scale_diagnostic.csv` | guard G7 | khác `g7_rq3_scale_evaluation.csv` (file này README gốc trích dẫn) — bản "diagnostic" là trung gian, mồ côi |
| `mechanism_trial_fastpath_quantile*.csv`, `..._random*.csv`, `..._summary.csv` | cơ chế Fast Path | script sinh ra (`f'mechanism_trial_fastpath_{split_mode}.csv'`) là `experiments/archive/nonlinear_mechanism_trial.py` — **đã archive từ trước** (xem `papers/p1_du_phong/experiments/archive/README.md`); output đi theo script. File `mechanism_trial_fastpath.csv` (không hậu tố) vẫn giữ ở `scm_results/` vì KHÔNG khớp đúng pattern `_{split_mode}` (để riêng, chưa xác minh nguồn) |
| `parser_ablation_benchmark__gpt-oss-120b.csv`, `parser_ablation_by_category__gpt-oss-120b.csv` | parser benchmark | cùng `llm_backend` với bản `__LIVE-openai-gpt-oss-120b.csv` (vẫn giữ, đang được trích dẫn) — tên cũ trước khi quy ước thêm tiền tố `LIVE-`, trùng dữ liệu |
| `rq6_dac_trung_cong.csv`, `rq6_dich_chuyen_tang.csv`, `rq6_do_nhay_khong_gian_con.csv`, `rq6_sua_thang_do.csv` | bài 2, thăm dò | không khớp tên script nào hiện có trong `papers/p2_khop_tang/experiments/`; không nằm trong bảng E1–E9 |
| `rq6_v3_TT_cham_m0.csv`, `rq6_v3_causalrca_SS_m0/m1/m2.csv`, `rq6_v4_TT_m0.csv`, `rq6_v4_causalrca_SS_m0/m1/m2.csv` | bài 2, thăm dò | tên không khớp bất kỳ script hiện có (khác hẳn `rq6_tier_decomposition_v3.py` — file đó dùng tên khác, vẫn đang chạy) |
| `rq6_wrapper_rcaeval_cham.csv`, `rq6_wrapper_rcaeval_nhanh.csv` | bài 2, thăm dò | `data/DATA.md` trích `rq6_wrapper_rcaeval.py` và các file `_v2_<Hệ>.csv` — hai file `_cham`/`_nhanh` này không phải biến thể đó, không được nhắc; script duy nhất có nhắc tới tiền tố này (`rq6_thang_du_lan_truyen.py`, đã archive) chỉ ĐỌC `_v2_{h}.csv`, không ghi `_cham`/`_nhanh` |
| `saturating_mechanism_trial_quantile.csv`, `..._summary.csv` | cơ chế bão hoà | script sinh ra (`f'saturating_mechanism_trial_{split_mode}.csv'`) là `experiments/archive/saturating_mechanism_trial.py` — **đã archive từ trước**; output đi theo script |

**Không viện dẫn số nào từ các file trên.** Nếu cần tái lập, phải tìm lại script sinh ra
chúng trong lịch sử git (`git log --all --diff-filter=A -- <tên file>`), không có sẵn trong
cây hiện tại.
