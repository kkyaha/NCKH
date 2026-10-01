# Hai bài báo

Repo này chứa **hai bài** dùng chung một core trong `src/`.

| | [`p1_du_phong/`](p1_du_phong/) | [`p2_khop_tang/`](p2_khop_tang/) |
|---|---|---|
| bài toán | thêm tính năng thì hệ chịu được tới tải nào | nguyên nhân gốc nằm ở **tầng** nào |
| loại | **dự phóng** (trước khi xảy ra) | **định vị hậu nghiệm** (sau khi xảy ra) |
| dữ liệu | Sock Shop **tự đo** (đo → thêm tính năng → đo lại) | RCAEval RE2, **270 ca** tiêm lỗi, 3 hệ |
| script | 62 | 21 |
| notebook | 01, 02 | **03** |
| kết quả | `data/processed/frozen/` (48 file, có SHA-256) | `data/processed/scm_results/` (`rq6_*`) |
| trạng thái | đóng băng; cần **đo thêm**, không cần phân tích thêm | đang củng cố; cần baseline của tác giả |

## Vì sao `src/` không tách

`src/agents/capacity_agent.py` được **29 file từ cả hai bài** import. Nó là core dùng chung,
không thuộc bài nào. Bảng phụ thuộc đầy đủ ở cuối README của bài 1.

Chỉ `src/scm/feasibility_predictor.py` là sạch một phía — toàn bộ 16 file import nó đều
thuộc bài 1.

## Quy ước đường dẫn

Mọi script tìm gốc repo bằng cách **leo lên đến thư mục chứa `src/`**, không dùng số cấp
`'..'`. Nên script chạy đúng ở bất kỳ độ sâu nào, và di chuyển file không làm sai đường dẫn
dữ liệu:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
```

Script import module ở nhóm khác thì dùng `experiments/_paths.py` ở gốc — nó quét **cả**
`experiments/*` và `papers/*/experiments/*`.

`results*/` trong mỗi bài là **symlink** tới `data/processed/`. CSV **không** bị di chuyển,
để không phải sửa đường dẫn ghi trong ~40 script. Đổi lại, `data/processed/scm_results/`
chứa kết quả của cả hai bài (`rq6_*` là bài 2).

## Chưa phân loại

`experiments/chua_phan_loai/` — 18 script chưa thuộc bài nào: `rq7`–`rq12` (tính hợp lệ can
thiệp, phản thực, nhiễu gây lẫn), `alibaba_*` (7, khảo sát bộ dữ liệu Alibaba), `future_rca.py`,
`rq_joint_gateway_gap.py`, `latency_covariate_diagnostic.py`.
