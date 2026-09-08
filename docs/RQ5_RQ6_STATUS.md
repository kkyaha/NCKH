# RQ5 & RQ6 — Trạng thái: CHƯA kiểm chứng

> **RQ5:** Kiến trúc đa tác tử (Parser Agent + Capacity Agent, ReAct) có vượt trội một lời gọi
> LLM đơn lẻ (single-LLM-call) không?
> **RQ6:** Kiến trúc MAS-SCM đề xuất có tổng quát hoá (Plug-and-Play) sang các hệ Microservices
> khác (Online Boutique, Train Ticket, ...) mà không cần hardcode không?

---

## RQ5 — CHƯA kiểm chứng
**Lý do**: chưa có baseline "single-LLM-call" (một lần gọi LLM duy nhất thay cho toàn bộ pipeline
ReAct 2-agent) để so sánh F1/MAPE/thời gian/chi phí. Cần xây dựng thêm 1 baseline runner
(tương tự cách `parser_benchmark_suite.py` đã làm cho RQ3) trước khi có thể tuyên bố bất kỳ kết
luận nào. Đây là công việc thuộc kiến trúc MAS, không phải lõi mô hình nhân quả SCM — không nằm
trong phạm vi đợt rà soát này (đã ưu tiên RQ1/RQ2/RQ4 theo yêu cầu tập trung vào SCM).

## RQ6 — CHƯA kiểm chứng
**Lý do**: repo hiện chỉ có dữ liệu RCAEval/RE2-SS (SockShop, 7 service). Không có dữ liệu
Online Boutique hay Train Ticket trong `data/raw/`. Không thể kiểm chứng generalization nếu
không có dữ liệu hệ thống thứ hai — đây là giới hạn dữ liệu, không phải lỗi phương pháp luận.
Muốn giải RQ6 cần: (1) thu thập/tải bộ dữ liệu benchmark thứ hai có cấu trúc topology tương tự,
(2) chạy lại pipeline RQ1/RQ2/RQ4 không đổi code trên bộ dữ liệu đó, (3) so sánh độ suy giảm
hiệu năng.

---
**Khuyến nghị khi viết bài**: nêu RQ5/RQ6 là "hướng phát triển tương lai" (future work), không
đưa số liệu chưa có vào phần kết quả.
