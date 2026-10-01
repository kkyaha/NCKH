# Bài 2 — Khớp tầng: định vị nguyên nhân trong hệ microservice

> **Mệnh đề:** độ chính xác định vị nguyên nhân bị quyết định bởi **tầng metric được soi**,
> không bởi thuật toán xếp hạng — và tầng đúng là tầng mà nguyên nhân nằm.

Loại bài: **định vị hậu nghiệm** (post-hoc RCA). **Không** phải dự phóng — xem [bài 1](../p1_du_phong/).

## Bắt đầu từ đâu

**[`notebooks/03_khop_tang_ly_thuyet_va_thuc_nghiem.ipynb`](notebooks/)** — cơ sở lý thuyết
và toàn bộ thực nghiệm trong một notebook (59 ô, chạy trọn, 0 lỗi). Mọi con số được tính lại
tại chỗ từ CSV.

## Dữ liệu

RCAEval **RE2**: 3 hệ × 90 kịch bản = **270 ca tiêm lỗi thật**.
Đáp án lấy từ **tên thư mục** — không suy từ đồ thị hay mô hình nào.

| | Sock Shop | Online Boutique | Train Ticket |
|---|---|---|---|
| service ứng dụng | 7 | 11 | 28 |
| cột ứng viên ở tầng `cpu` | 15 | 11 | 68 |
| service **bị tiêm** | 5 | 5 | 5 |

Thư mục: `data/raw/RE2-SS`, `data/raw/RE2-OB`, `data/raw/trainticket` (ở gốc repo, dùng chung).

## Thực nghiệm → script → CSV

| | thực nghiệm | script | CSV |
|---|---|---|---|
| **E1** | định vị trên ba tầng, ba hệ | `rq6_three_layer_and_scope.py` | `rq6_three_layer_localization.csv` |
| **E1b** | bộ chỉ số chuẩn RCAEval + sàn của **từng** chỉ số | `rq6_rcaeval_metrics.py` | `rq6_rcaeval_metrics_layers.csv` |
| **E2** | chiều can thiệp nhu cầu, ba phép chấm | `rq6_localization_decircularized.py` | `rq6_localization_decircularized.csv` |
| **E3** | điều kiện phạm vi, đo từ dữ liệu thô | `rq6_three_layer_and_scope.py` | `rq6_scope_condition_two_systems.csv` |
| **E4** | cơ chế NNLS, 4 node — **đã stale** | `rq6_hop1_node_diagnostic.py` | `rq6_hop1_node_diagnostic.csv` |
| **E4b** | cùng phép đo, **45 node · 3 hệ** | `rq6_mechanism_three_systems.py` | `rq6_mechanism_three_systems_nodes.csv` |
| **E5** | lưới khớp 2×2×2 với BARO | `rq6_baro_matched_comparison.py` | `rq6_baro_matched_comparison.csv` |
| **E5b** | lưới mở rộng 3 hệ × 4 tầng × 3 tập ứng viên + **đối chứng âm** | `rq6_matched_grid.py` | `rq6_matched_grid_summary.csv` |
| **E6** | đơn vị quy gán: node / cạnh / đường | `rq6_attribution_unit.py` | `rq6_attribution_unit.csv` |
| **E7** | hai mức sàn (trả lời arXiv 2609.27069) | `rq6_faultset_floor.py` | `rq6_faultset_floor_summary.csv` |
| **E8** | phân rã tầng + quy mô — **đã stale** | `rq6_tier_decomposition_v2.py` | `rq6_tier_decomposition_v2_summary.csv` |
| **E8 v3** | **thay v2**: 3 hệ, mọi node `_cpu`, ngưỡng **ngoài mẫu** | `rq6_tier_decomposition_v3.py` | `rq6_tier_decomposition_v3_nodes.csv` |
| **E9** | giao thức đánh giá: đối chứng dương | `recalibrate_risk_threshold.py` | `rq6_risk_threshold_recalibration.csv` |

Script ở `experiments/`, CSV ở `results/` (symlink tới `data/processed/scm_results/`).

> ⚠️ **Hai CSV stale.** `rq6_hop1_node_diagnostic.csv` (E4) và `rq6_tier_decomposition_v2*.csv`
> (E8) được tạo **12/09**, trước khi `src/graph/trainticket_scm_edges.json` được sinh lại
> **18/09**. Tập cạnh tranh chấp đổi hoàn toàn — **giao nhau của tập node dích là rỗng**.
> Dùng **E4b** và **E8 v3** thay thế.

## Chạy lại

```bash
# tu GOC REPO
python papers/p2_khop_tang/experiments/rq6_three_layer_and_scope.py
python papers/p2_khop_tang/experiments/rq6_rcaeval_metrics.py
python papers/p2_khop_tang/experiments/rq6_matched_grid.py --kiem-tai-lap
python papers/p2_khop_tang/experiments/rq6_mechanism_three_systems.py
python papers/p2_khop_tang/experiments/rq6_tier_decomposition_v3.py --repeats=5
python papers/p2_khop_tang/experiments/rq6_faultset_floor.py
python papers/p2_khop_tang/experiments/rq6_attribution_unit.py
python papers/p2_khop_tang/experiments/recalibrate_risk_threshold.py
```

**Thêm baseline của tác giả** (việc còn lại lớn nhất):

```bash
git clone https://github.com/phamquiluan/RCAEval ~/RCAEval && cd ~/RCAEval && pip install -e .
RCAEVAL_PATH=~/RCAEval python papers/p2_khop_tang/experiments/rq6_matched_grid.py
```

`rq6_matched_grid.py` tách phần đọc dữ liệu / chấm điểm / tính chỉ số khỏi phần xếp hạng,
nên thêm một baseline chỉ là viết **một** hàm adapter — xem hợp đồng adapter trong docstring
của file đó.

## Core dùng chung

Bài này dùng `src/` ở gốc repo. Chỉ **E4b** và **E8 v3** cần huấn luyện DAG
(`src/agents/capacity_agent.py`); mọi thực nghiệm khác đọc **trực tiếp dữ liệu thô** và
không phụ thuộc mô hình nào — đó là lý do chúng không vòng tròn.

## Tài liệu

- `docs/PHAT_HIEN_KHOP_TANG.md` — chi tiết từng phát hiện (⚠️ chưa cập nhật cho ba hệ)
- `docs/CAU_TRUC_BAI_BAO_XAI.md` — cấu trúc bài báo (⚠️ viết trước lưới BARO)
- `docs/POSITIONING_GROUNDING.md` — định vị so với văn liệu
- `docs/xai_attribution_paper_draft.tex` — bản nháp
