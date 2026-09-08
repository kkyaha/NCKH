# RQ5 — Trạng thái: CHƯA kiểm chứng

> **RQ5:** Kiến trúc đa tác tử (Parser Agent + Capacity Agent, ReAct) có vượt trội một lời gọi
> LLM đơn lẻ (single-LLM-call) không?

**Lý do**: chưa có baseline "single-LLM-call" (một lần gọi LLM duy nhất thay cho toàn bộ pipeline
ReAct 2-agent) để so sánh F1/MAPE/thời gian/chi phí. Cần xây dựng thêm 1 baseline runner
(tương tự cách `parser_benchmark_suite.py` đã làm cho RQ3) trước khi có thể tuyên bố bất kỳ kết
luận nào. Đây là công việc thuộc kiến trúc MAS, không phải lõi mô hình nhân quả SCM.

**Khuyến nghị khi viết bài**: nêu RQ5 là "hướng phát triển tương lai" (future work), không đưa
số liệu chưa có vào phần kết quả.

---

# RQ6 — Đã kiểm chứng (Train Ticket)

> **RQ6:** Kiến trúc MAS-SCM đề xuất có tổng quát hoá (Plug-and-Play) sang các hệ Microservices
> khác (Online Boutique, Train Ticket, ...) mà không cần hardcode không?

**Cập nhật 2026-09-08**: RQ6 đã được kiểm chứng trên **Train Ticket** (28 microservices, RCAEval),
dùng đúng pipeline đánh giá đã áp dụng cho SockShop (full-resolution MAPE, 4-model comparison,
kiểm định thống kê tách theo metric, ground-truth direct match). Xem chi tiết đầy đủ tại
[`RQ6_TRAINTICKET_GENERALIZATION_REPORT.md`](RQ6_TRAINTICKET_GENERALIZATION_REPORT.md).

**Tóm tắt trung thực**: pipeline chạy được trên hệ thống thứ hai mà không cần sửa code lõi (chỉ
cần cấu hình topology/service list mới) — về mặt kỹ thuật, tính "Plug-and-Play" được xác nhận.
Nhưng **độ chính xác điểm của SCM không tổng quát hoá đồng đều**: SCM đồng hạng nhất ở SockShop
(7/21 lần thắng) nhưng xếp cuối ở Train Ticket (4/56 lần thắng, thua cả 3 baseline ML). Đây là
phát hiện quan trọng cần trình bày trung thực trong bài báo, không nên diễn giải là "RQ6 thất
bại" — mà là bằng chứng cho thấy giá trị của SCM nằm ở khả năng can thiệp do(x) chứ không phải độ
chính xác dự báo thô, và độ chính xác thô vốn dĩ nhạy với đặc điểm từng hệ thống (điều baseline ML
cũng gặp phải tương tự, không riêng SCM).

**Giới hạn còn lại**: mới chạy 30/90 kịch bản khả dụng của Train Ticket (do phạm vi được chọn để
có kết quả nhanh) — có thể mở rộng lên 90 kịch bản nếu cần độ tin cậy cao hơn cho bản nộp cuối.
