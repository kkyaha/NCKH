# `archive/` — script đã gộp hoặc gỡ khỏi luồng chính

Giữ **nguyên văn**, không xoá. Lý do giữ: các script này đã sinh ra những phát hiện
được dùng để chọn thiết kế mô hình; nếu phản biện hỏi "các anh đã thử gì rồi" thì
phải tái lập được chính bản đã chạy, không phải bản viết lại.

## Gộp vào `edges/dang_co_che_doi_chung.py`

Ba script là **cùng một khung đo** — cùng protocol, cùng 21 cặp service × metric,
cùng `fit_and_eval` — chỉ khác bộ cơ chế đem ra so. Đã đo độ trùng dòng:

| file | cơ chế nó so | trùng với |
|---|---|---|
| `nonlinear_mechanism_trial.py` | auto_gcm, linear_pos, hgbr_mono | — (bản gốc của khung) |
| `saturating_mechanism_trial.py` | linear_pos, hgbr_mono, sigmoid_sat | 73% với bản trên |
| `log_transform_trial.py` | linear_pos, log_linear (chỉ quantile) | 54% / 50% |

Bản gộp là **tập hợp trên chặt**: giữ nhánh đặc biệt `auto_gcm`, giữ trường
`used_sigmoid` và `mae` (bản log-transform không ghi `mae`; bản gộp ghi thêm, không bớt gì).

## Vì sao số của bản gộp khác số của ba bản này

Ba bản gốc **không gieo hạt** cho `gcm.interventional_samples`, nên không tái lập được.
Đã đo: chạy `nonlinear_mechanism_trial.py quantile` hai lần liên tiếp cho MAPE
13.5006% rồi 13.5660% — lệch trung bình 0.64 pp, cao nhất 3.03 pp, **21/21 cặp đều khác**.

Hệ quả phải nhớ khi viết bài: khoảng cách "linear_pos 12.90% vs hgbr_mono 13.67%" là
0.77 pp, **nhỏ hơn biên độ nhiễu chạy lại**. So sánh MAPE tổng hợp một mình không
chứng minh được gì — **đừng viện dẫn nó**. Kết luận "giữ linear_pos" vẫn đúng nhưng
dựa trên hai thứ khác, mạnh hơn: (1) lập luận cấu trúc — mô hình dựa trên cây không
ngoại suy được qua phạm vi Workload đã thấy trong train, nên đánh giá **thấp** nguy cơ
ở vùng tải cao; (2) cả ba họ hàm cùng thất bại trên đúng những node tệ nhất
(`catalogue_cpu` > 90% MAPE ở cả ba), tức nguyên nhân không phải "sai dạng hàm".

Bản gộp gieo hạt `HAT = 42` nên tất định (đã kiểm: hai lần chạy giống nhau đến 10
chữ số thập phân). Vì thế số của nó khác số ở đây — **đó là chủ ý**.

## Gộp vào `edges/canh_an_toan_ngoai_suy.py`

| file | hệ | trùng |
|---|---|---|
| `backpressure_edge_ood_safety_test.py` | Sock Shop, 3 cạnh ghi tay | — |
| `tt_backpressure_edge_ood_safety_test.py` | Train Ticket, 50 cạnh đọc từ chẩn đoán (gain > 0.03) | 65% với bản trên |

Khác biệt duy nhất là **cấu hình hệ** (danh sách service, đồ thị, dải delta, số mẫu
chiếu) và cách lấy cạnh. Cả hai đã vào bảng `HE` trong bản gộp.

Đã đối chiếu bản gốc với bản gộp trên Sock Shop: tổng số ca đổi dấu
OLD=28 / NEW=24 (gốc) so với OLD=29 / NEW=22 (gộp) — cùng độ lớn và **cùng kết luận**:
cạnh backpressure *giảm* số ca đổi dấu, không làm tăng. Chênh lệch là nhiễu do bản gốc
không gieo hạt. Bản gộp gieo `HAT = 42`.

## Gộp vào `edges/canh_do_chinh_xac.py`

| file | chế độ tương đương | hệ |
|---|---|---|
| `backpressure_edge_accuracy_test.py` | `--che-do canh` | Sock Shop (3 cạnh ghi tay) |
| `tt_backpressure_edge_accuracy_test.py` | `--che-do canh` | Train Ticket (50 cạnh, gain > 0.03) |
| `tt_backpressure_multiparent_accuracy_test.py` | `--che-do da-cha` | Train Ticket (20 node, toàn bộ caller cùng lúc) |
| `memsocket_edge_accuracy_test.py` | `--che-do mem-socket` | Sock Shop (own mem + socket) |

Trùng 65% / 51% / 47% / 45% số dòng giữa các cặp. Cả bốn dùng **cùng** phép chia
(sắp theo workload của callee, train 67% thấp → test 33% cao), **cùng**
`LinearRegression(positive=True)`, **cùng** quy tắc ngưỡng (mean + 0.5·std trên *tập
train*), **cùng** bộ chỉ số — khác biệt duy nhất là *cột nào là parent gốc, cột nào là
parent thêm*. Bản gộp quy về một hàm `danh_gia(goc, them)`.

### Đã đối chiếu — bản gộp tái lập chính xác cả bốn

| chế độ | bản gốc | bản gộp | khớp |
|---|---|---|---|
| SS `canh` | MAPE 48.443 → 40.273; R² −0.030 → 0.084 | giống hệt | **từng cạnh, từng chỉ số, sai khác < 1e-9** |
| SS `mem-socket` | MAPE 40.25 → 41.91; R² 0.0229 → 0.0340; cải thiện 2/7, 3/7 | giống hệt | ✓ |
| TT `canh` | MAPE 66.71 → 48.80; R² −0.1881 → 0.1371; F1 0.191 → 0.515; 47/50 | giống hệt | ✓ |
| TT `da-cha` | MAPE 71.09 → 44.93; R² −0.1502 → 0.3156; F1 0.176 → 0.592; 20/20 | giống hệt | ✓ |

Khác với hai bản gộp kia, nhóm này **không** dùng `gcm.interventional_samples` (thuần
sklearn) nên tất định sẵn — đó là lý do khớp được tuyệt đối, chứ không chỉ "cùng độ lớn".
