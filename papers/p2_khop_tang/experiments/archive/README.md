# `archive/` — script đã gỡ khỏi luồng chính

Giữ nguyên văn, **không xoá**, để nếu phản biện hỏi thì còn tái lập được.
Mỗi file dưới đây kèm *phép đo* chứng minh nó bị gỡ — không gỡ theo cảm tính.

| file | lý do gỡ | bằng chứng |
|---|---|---|
| `rq6_tier_decomposition.py` | bị `_v2`/`_v3` thay thế | `rq6_tier_decomposition_v2.py:5` ghi rõ v1 `classify_contribs()` thiếu; notebook `03_khop_tang_*.ipynb` chỉ đọc `_v2.csv` và `_v3.csv`, không đọc `rq6_tier_decomposition.csv` |
| `rq6_doi_thang_trong_baro.py` | **kết quả sai** | Tôi tự cài lại BARO thay vì gọi bản thật → AC@1 27.78% trong khi BARO thật cho 34.81%; 23/270 ô khác nhau. Bản đúng là `rq6_2x2_baro.py`, vá `RobustScaler` ngay trong `RCAEval.e2e.baro` |
| `rq6_vathang_baro.py` | **kết quả sai** | Đánh cột suy biến thành NaN rồi `sorted()` — so sánh với NaN luôn False nên cột rơi vào vị trí *bất định*, sinh ra kết luận giả "MAD tốt nhất". Cách đúng: bỏ cột khỏi dataframe |
| `rq6_uoc_luong_thang.py` | **kết quả sai** | Cùng lỗi NaN-sort như trên; kết luận "MAD tốt nhất" không đứng |

CSV tương ứng của ba file sai đã chuyển vào `data/processed/scm_results/archive/`.
**Không viện dẫn số nào từ chúng.**
