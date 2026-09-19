# Báo cáo tiến độ (bản dễ hiểu)

**Ngày:** 16/09/2026

---

## 1. Đang nghiên cứu cái gì?

Hãy tưởng tượng: một khách hàng gõ một câu bình thường, kiểu *"thêm tính năng cho khách đánh giá sao sản phẩm"*. Hệ thống nhóm xây dựng sẽ:

1. Nhờ một AI (giống ChatGPT) đọc câu đó và đoán xem tính năng này sẽ làm tăng bao nhiêu phần trăm lượng truy cập vào hệ thống.
2. Từ con số đó, tính tiếp xem CPU, bộ nhớ, độ trễ của từng dịch vụ trong hệ thống sẽ tăng lên bao nhiêu.
3. Trả lời: tính năng này an toàn, hay sẽ làm sập một chỗ nào đó — **tất cả trước khi lập trình viên viết một dòng code nào**.

Cái khó: AI đôi khi "ảo giác" — bịa ra tên dịch vụ không có thật, hoặc phán một con số vô lý (kiểu "tải tăng 50.000%"). Nếu tin thẳng vào AI, hệ thống sẽ đưa ra lời khuyên sai mà không ai biết.

## 2. Ý tưởng cốt lõi: một "người gác cổng" luôn đúng

Nhóm xây một lớp kiểm tra đứng ngay sau AI, đóng vai trò như **người gác cổng**: dù AI có trả lời bậy đến đâu, lớp này luôn sửa lại thành một câu trả lời nằm trong giới hạn hợp lý — không bao giờ để lọt cái sai ra ngoài.

Điểm đặc biệt: đây không phải "thường thì đúng", mà là **chứng minh được bằng toán** rằng nó **luôn luôn đúng**, với mọi câu trả lời AI có thể đưa ra, kể cả những câu điên rồ nhất. Giống như một cái van an toàn trên nồi áp suất — không quan trọng bên trong sôi mạnh cỡ nào, van vẫn xả đúng lúc.

## 3. Đã phát hiện vấn đề gì khi rà soát lại?

**Vấn đề 1 — bằng chứng "AI giỏi hơn thì hết cần gác cổng" còn quá mỏng.** Nhóm mới thử nghiệm trên 2 loại AI, một loại chỉ thử 1 lần (do giới hạn số lượt dùng miễn phí). Cần thử thêm để chắc chắn.

**Vấn đề 2 — một phần kết quả bị "vô nghĩa lặp lại".** Vì người gác cổng luôn sửa đúng theo định nghĩa, nên đo "guard có sửa đúng không" trên đúng cấu hình đó thì tất nhiên kết quả luôn là 100% — không nói lên điều gì mới. Cần đo bằng cách khác (đưa input cố tình khó để xem guard có thực sự "ra tay" hay không).

**Vấn đề 3 — không có bộ dữ liệu công khai nào "vừa đủ tên thật, vừa đủ biến động tải".** Để dự đoán được, hệ thống cần hai thứ: (a) tên dịch vụ thật, dễ hiểu, để hiểu khách hàng đang nói về cái gì; (b) dữ liệu có tải lên xuống thật nhiều, để học được quy luật "tải tăng thì tài nguyên tăng bao nhiêu". Trớ trêu là hai bộ dữ liệu công khai sẵn có lại chia đôi ngã:

- **Bộ RCAEval** (dùng để giả lập lỗi hệ thống): có tên dịch vụ thật (`front-end`, `orders`, `payment`...), nhưng tải được giữ **cố định, ổn định** — vì mục đích ban đầu của bộ này là để một lỗi tiêm vào nổi bật lên trên nền ổn định, không phải để học quy luật tải.
- **Bộ Alibaba** (dữ liệu vận hành thật của một công ty lớn): tải biến động thật, nhiều, đúng thứ cần để học quy luật — nhưng tên mọi dịch vụ đều bị **mã hoá thành chuỗi ký tự vô nghĩa**, không thể biết dịch vụ nào làm gì.

Không bộ nào một mình đủ dùng. Giải pháp: dùng RCAEval cho phần "hiểu ngôn ngữ khách hàng, biết chặn AI ảo giác", dùng Alibaba cho phần "học quy luật tải → tài nguyên". Đây là một giới hạn thật của dữ liệu công khai hiện có, không phải sai sót của nhóm.

**Vấn đề 4 — một bộ dữ liệu tưởng đo được nhưng thực chất không đo được gì.** Vì RCAEval không có biến động tải thật (Vấn đề 3), nếu dùng nó để thử dự đoán RAM sẽ ra sai số trông rất đẹp (chỉ 2–4%) một cách giả tạo: bộ dữ liệu gần như không có gì để dự đoán, nên đoán bừa cũng ra sai số thấp — giống nhiệt kế hỏng luôn chỉ đúng 25°C, nếu phòng thật sự luôn ở 25°C thì trông vẫn có vẻ "chính xác". Phải có thêm một bước kiểm tra riêng — đo trước xem dữ liệu có tín hiệu thật hay không — để không bị chính con số sai số thấp này đánh lừa.

## 4. Đã làm được gì?

### 4.1. Không chỉ chặn đầu vào của AI — tính luôn được cam kết cho kết quả cuối

Trước đây, "người gác cổng" chỉ đảm bảo con số AI đưa vào nằm trong khoảng an toàn (ví dụ 5%–50%). Nhưng con số kỹ sư thực sự đọc là ở **cuối chuỗi tính toán** (CPU, độ trễ...), không phải con số đầu vào đó.

Nhóm chứng minh được: vì mọi công thức tính toán bên trong hệ thống đều có tính chất "**tăng thì chỉ tăng, không bao giờ giảm ngược**" (tải vào tăng thì CPU/độ trễ ra chỉ có thể tăng theo, không thể tự giảm), nên chỉ cần tính **đúng 2 kịch bản** — "trường hợp thấp nhất" và "trường hợp cao nhất" — là chắc chắn mọi kịch bản ở giữa đều nằm gọn trong khoảng đó. Giống như biết nhiệt độ lúc 6h sáng và 6h chiều là đủ để cam kết "cả ngày không lạnh hơn 6h sáng và không nóng hơn 6h chiều" — miễn nhiệt độ chỉ tăng dần trong ngày.

Kết quả: hệ thống giờ có thể đưa ra một **"tờ cam kết"**: *"nếu tính năng này được duyệt, CPU dịch vụ X chắc chắn nằm trong khoảng 12%–34%"* — thay vì chỉ đoán một con số vu vơ.

### 4.2. Kiểm tra lại trên hệ thống thật — và tìm ra một lỗi có thật

Thay vì chỉ tin vào chứng minh trên giấy, nhóm đã **chạy thử trực tiếp trên code thật** để xem lời hứa toán học ở trên có thực sự đúng không.

Kết quả: phát hiện công thức tính **độ trễ (latency)** của hệ thống bị lỗi ở một điểm rất tinh vi — nó vi phạm đúng cái tính chất "chỉ tăng không giảm" nói trên. Có trường hợp cụ thể: công thức dự đoán tải tăng lên 50% thì độ trễ lại **thấp hơn** so với khi tải chỉ tăng 5% — ngược hẳn logic thông thường. Lỗi này tồn tại trên cả hai hệ thống thử nghiệm (không phải trùng hợp riêng một hệ).

Sửa lỗi này (bằng đúng một kỹ thuật đã dùng ở chỗ khác trong hệ thống) không chỉ khôi phục logic đúng, mà **độ chính xác dự đoán còn tốt lên** ở phần lớn trường hợp — không phải đánh đổi, mà là sửa một lỗi từ trước đến giờ chưa ai để ý.

### 4.3. Câu hỏi mới: hai tính năng ship cùng lúc thì sao?

Thực tế, khách hàng thường duyệt nhiều tính năng cùng một đợt, không phải từng cái một. Câu hỏi đặt ra: nếu tính năng A một mình thì an toàn, tính năng B một mình cũng an toàn, thì **A + B cùng lúc có chắc vẫn an toàn không**?

Nhóm chứng minh: **không hẳn**. Nếu hai tính năng đó cùng đi qua một "nút thắt cổ chai" nào đó trong hệ thống (giống như hai dòng xe cùng đổ về một ngã tư), thì độ trễ khi cả hai xảy ra cùng lúc sẽ **nặng hơn** tổng của hai cái tính riêng lẻ cộng lại — không phải cộng dồn đơn giản, mà là hiệu ứng dồn ứ kiểu tắc đường giờ cao điểm. Đây là điều kiểm được từng con số cụ thể, không phải phỏng đoán.

Nhóm đã kiểm chứng điều này theo hai cách: trên một hệ thống dựng sẵn để test (khớp đúng dự đoán), và trên **dữ liệu vận hành thật của một hệ thống lớn thật** (dữ liệu Alibaba) — tìm được 2 trường hợp thật, cả hai đều cho kết quả đúng như dự đoán, không có trường hợp nào ngược lại. Đây mới là bước đầu (2 trường hợp), chưa đủ để nói "luôn luôn đúng với mọi hệ thống", nhưng là bằng chứng thật đầu tiên, không phải chỉ trên dữ liệu tự dựng.

### 4.4. Đi theo đúng sơ đồ kết nối thật thì có ích — nhưng một kỹ thuật "truy tìm nguyên nhân" phức tạp hơn thì không cần

Để dự đoán tải lan truyền từ dịch vụ này sang dịch vụ khác, có hai cách:

- **Cách ngây thơ:** giả định "tải vào tăng bao nhiêu % thì mọi dịch vụ phía sau cũng tăng đúng bấy nhiêu %" — không quan tâm ai thật sự gọi ai.
- **Cách nhóm dùng:** đi theo **đúng sơ đồ kết nối thật** giữa các dịch vụ (dịch vụ A thật sự gọi dịch vụ B bao nhiêu, B gọi C bao nhiêu...) để tính lan truyền chính xác hơn.

Kiểm tra trên dữ liệu Alibaba thật: cách đi theo sơ đồ thật **thắng** cách ngây thơ trên phần lớn trường hợp — đúng như kỳ vọng, việc hiểu cấu trúc thật của hệ thống là có giá trị.

Riêng một câu hỏi khác thì kết quả bất ngờ. Có một kỹ thuật khoa học nổi tiếng (thuộc lĩnh vực "suy luận nhân quả") chuyên dùng để tách bạch "nguyên nhân thật" khỏi "trùng hợp giả" — ví dụ kinh điển: doanh số kem và số vụ đuối nước cùng tăng vào mùa hè, không phải vì ăn kem gây đuối nước, mà vì cả hai đều do trời nóng — một nguyên nhân ẩn giấu phía sau. Kỹ thuật này giúp "lọc" đúng nguyên nhân ẩn giấu đó ra khỏi phép tính.

Kỹ thuật này thường được dùng để **truy tìm nguyên nhân một sự cố đã xảy ra** (kiểu thám tử điều tra hiện trường). Nhóm thử mang nó sang bài toán ngược: **dự đoán một tính năng chưa xảy ra**. Kết quả: kỹ thuật này **không giúp thêm được gì** ở đây — vì mọi tính năng khách hàng đều "vào" hệ thống từ đúng một cửa chính (cổng vào duy nhất), không hề có nguyên nhân ẩn giấu nào đứng trước cửa đó cần phải lọc bỏ. Giống như đang đứng ngay tại cửa để xem trời có mưa không — không cần kỹ thuật thám tử phức tạp để loại trừ khả năng khác, vì đứng tại cửa thì thấy trực tiếp luôn rồi. Nhóm đã kiểm tra kết luận này bằng 4 cách độc lập khác nhau, tất cả đều cho cùng một câu trả lời.

**Tóm lại:** hiểu đúng sơ đồ kết nối thật giữa các dịch vụ — có ích thật. Kỹ thuật thám tử nhân quả phức tạp — không cần thiết trong đúng bài toán này (dù rất hữu ích ở bài toán khác, kiểu tìm nguyên nhân sự cố đã xảy ra).

## 5. Đang gặp khó khăn ở đâu?

- **Chưa kịp chạy thử với AI mắc tiền, xịn hơn.** Đã chuẩn bị xong hạ tầng, chỉ còn quyết định chi phí trước khi bấm chạy.
- **Bằng chứng "hai tính năng cùng ship" trên dữ liệu thật đã bị rút lại.** Ban đầu tìm được 2 trường hợp thật khớp đúng dự đoán, nhưng khi lấy thêm dữ liệu để kiểm tra lại cho chắc, cả hai biến mất hoàn toàn. Nguyên nhân: cách hệ thống ước lượng "trần năng lực tối đa" của một dịch vụ dựa trên **con số lớn nhất từng thấy được** — có thêm dữ liệu thì thấy một con số cao hơn, cái trần bị đẩy lên, khiến toàn bộ dữ liệu cũ tự nhiên "trông còn xa trần" hơn hẳn, và phần công thức mô phỏng tắc nghẽn (vốn là thứ gây ra hiệu ứng "cộng dồn nặng hơn") bị hệ thống coi là không cần thiết nữa. Đây là một lỗi thật trong cách đo "trần năng lực" — không phải ý tưởng ban đầu sai, mà là thước đo dùng để kiểm nó chưa đủ ổn định. Hiện chỉ còn giữ bằng chứng trên dữ liệu tự dựng (nơi trần năng lực được đặt cố định từ đầu, không phụ thuộc vào việc quan sát được bao nhiêu).

## 6. Sắp tới làm gì?

1. Chạy thử với AI xịn hơn, xem lớp gác cổng còn cần thiết như thế nào.
2. Sửa lại cách đo "trần năng lực" cho ổn định hơn (không dựa vào mỗi con số lớn nhất từng thấy), rồi thử lại xem hiện tượng "hai tính năng cùng ship nặng hơn cộng dồn" có xuất hiện lại trên dữ liệu thật không.

---

*Tóm gọn một câu: nhóm không chỉ chứng minh trên giấy rằng hệ thống an toàn, mà còn thực sự đi kiểm tra trên code chạy thật — và trong lúc kiểm tra, tìm ra một lỗi có thật, sửa được, và mở rộng thêm một câu hỏi mới (nhiều tính năng cùng lúc) mà trước đó chưa ai đặt ra.*
