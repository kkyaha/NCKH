# `archive/` — script đã gỡ khỏi luồng chính

Giữ nguyên văn, **không xoá**, để nếu phản biện hỏi thì còn tái lập được.
Mỗi file dưới đây kèm *phép đo* chứng minh nó bị gỡ — không gỡ theo cảm tính.

| file | lý do gỡ | bằng chứng |
|---|---|---|
| `rq6_tier_decomposition.py` | bị `_v2`/`_v3` thay thế | `rq6_tier_decomposition_v2.py:5` ghi rõ v1 `classify_contribs()` thiếu; notebook `03_khop_tang_*.ipynb` chỉ đọc `_v2.csv` và `_v3.csv`, không đọc `rq6_tier_decomposition.csv` |
| `rq6_doi_thang_trong_baro.py` | **kết quả sai** | Tôi tự cài lại BARO thay vì gọi bản thật → AC@1 27.78% trong khi BARO thật cho 34.81%; 23/270 ô khác nhau. Bản đúng là `rq6_2x2_baro.py`, vá `RobustScaler` ngay trong `RCAEval.e2e.baro` |
| `rq6_vathang_baro.py` | **kết quả sai** | Đánh cột suy biến thành NaN rồi `sorted()` — so sánh với NaN luôn False nên cột rơi vào vị trí *bất định*, sinh ra kết luận giả "MAD tốt nhất". Cách đúng: bỏ cột khỏi dataframe |
| `rq6_uoc_luong_thang.py` | **kết quả sai** | Cùng lỗi NaN-sort như trên; kết luận "MAD tốt nhất" không đứng |
| `rq6_cham_hang.py` | **mồ côi** | Không nằm trong bảng E1–E9 chính thức của `papers/p2_khop_tang/README.md`; không file `.md`/`.tex`/`.py` nào khác trong repo nhắc tên nó. CSV đã sinh (`rq6_cham_hang.csv`, 10/10) nhưng không được dùng ở đâu |
| `rq6_chung_kiem_soat.py` | **mồ côi** | Như trên; script không sinh CSV nào (có thể dừng giữa đường khi phát triển) |
| `rq6_dieu_kien_hang.py` | **mồ côi** | Như trên; CSV `rq6_dieu_kien_hang.csv` đã sinh, không ai đọc |
| `rq6_gia_thiet_thua.py` | **mồ côi** | Như trên; CSV `rq6_gia_thiet_thua.csv` đã sinh, không ai đọc |
| `rq6_pipeline_hai_truc.py` | **mồ côi** | Như trên; CSV `rq6_pipeline_hai_truc.csv` + `rq6_pipeline_do_chat_duong.csv` đã sinh, không ai đọc |
| `rq6_thang_du_lan_truyen.py` | **mồ côi** | Như trên; CSV `rq6_thang_du_lan_truyen.csv` đã sinh, không ai đọc |

CSV tương ứng của các file trên đã chuyển vào `data/processed/scm_results/archive/`
(xem `data/processed/scm_results/archive/README.md` cho nhóm "mồ côi" rộng hơn, không
chỉ của bài 2 — audit chạy 2026-10-15, kiểm hai lớp: không script hiện có trong repo
sinh ra file, và không tài liệu nào trích dẫn tên file).
**Không viện dẫn số nào từ chúng.**
