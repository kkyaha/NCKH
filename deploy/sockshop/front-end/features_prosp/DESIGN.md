# DESIGN: GET /catalogue/browse (REQ-07)

## Tóm tắt

Route `GET /catalogue/browse` phục vụ trang "tất thể thao": lấy các sản phẩm
có tag `sport`, sau đó lọc thêm theo màu sắc và "kích cỡ" nếu người dùng có
truyền. Route công khai, không yêu cầu đăng nhập, không đụng tới session.

## Query params do route này tự định nghĩa

| Param | Bắt buộc | Mô tả |
|---|---|---|
| `color` | không | Danh sách màu, phân tách bởi dấu phẩy (`?color=blue,red`) hoặc lặp key (`?color=blue&color=red`). Khớp theo kiểu OR: sản phẩm chỉ cần có MỘT trong các màu được liệt kê. |
| `size` | không | Cùng cú pháp như `color`. Khớp theo kiểu OR trong nhóm size. |
| `page` | không | Số trang, 1-based. Mặc định `1`. Chỉ có tác dụng khi có `pageSize`. |
| `pageSize` | không | Số sản phẩm mỗi trang. Nếu không truyền, trả về TOÀN BỘ danh sách đã lọc (không phân trang) — vì REQ chỉ yêu cầu "danh sách sản phẩm khớp bộ lọc", không yêu cầu phân trang. |

`color` và `size` kết hợp với nhau theo kiểu AND (sản phẩm phải khớp cả hai
nhóm, nếu nhóm đó có được truyền); trong cùng một nhóm thì là OR (chọn nhiều
màu/size = "màu này hoặc màu kia"). Đây là hành vi filter UX thông thường của
các trang catalogue, không phải điều REQ nói rõ — xem mục "Điểm chưa chắc
chắn" bên dưới.

## Các bước xử lý (đúng thứ tự)

1. Nhận request `GET /catalogue/browse`, log lại bằng `console.log` (query
   params) theo phong cách route hiện có (`api/catalogue/index.js`).
2. Parse `color` và `size` thành mảng token đã trim + lowercase (hỗ trợ cả
   dạng `a,b` và dạng lặp key).
3. Parse `page`/`pageSize` thành số nguyên, có fallback an toàn nếu thiếu
   hoặc không hợp lệ.
4. Gọi backend **một lần duy nhất**:
   `GET http://catalogue/catalogue?tags=sport`
   (dùng `endpoints.catalogueUrl` sẵn có, giống cách `api/catalogue` đang
   gọi catalogue service).
5. Nếu request lỗi (network) hoặc catalogue trả về status khác 200, tạo
   `Error` (kèm `status`) và gọi `next(err)` — lỗi sẽ được xử lý bởi
   `helpers.errorHandler` mà front-end gắn ở tầng ngoài cùng, đúng pattern
   hiện tại (route `/catalogue` không tự xử lý lỗi, chỉ forward qua `next`).
6. `JSON.parse` body trả về; nếu parse lỗi hoặc kết quả không phải mảng,
   cũng coi là lỗi và `next(err)`.
7. Lọc mảng sản phẩm trong bộ nhớ: giữ lại sản phẩm mà `product.tag` chứa ít
   nhất một giá trị trong `color` (nếu có truyền) VÀ chứa ít nhất một giá
   trị trong `size` (nếu có truyền). So khớp không phân biệt hoa/thường.
8. Nếu có `pageSize` hợp lệ, cắt mảng đã lọc theo `page`/`pageSize`
   (`Array.prototype.slice`, tính toán ở front-end, không gọi lại backend).
9. Trả về `200` với body là `JSON.stringify(results)` thông qua
   `helpers.respondSuccessBody`, giống cách các route khác trong
   `helpers/index.js` trả response.

## Lời gọi backend, theo đúng thứ tự thực hiện

1. `GET http://catalogue/catalogue?tags=sport` — **lời gọi backend duy nhất**.

Không có lời gọi thứ hai. Lý do: `tags` param của catalogue service là
OR-match trên toàn bộ danh sách tag được truyền (API_SURFACE.md). Nếu gộp
`tags=sport,blue,red` trong một lời gọi, catalogue sẽ trả về cả sản phẩm màu
blue/red KHÔNG thuộc "sport" (vì đó là OR, không phải AND). Vì route cần
"sport AND (blue OR red)", nên chỉ có thể nhờ backend lọc phần chắc chắn
đúng là AND (tag `sport`), còn lại lọc `color`/`size` phải làm ở tầng
front-end sau khi có dữ liệu về. Đây là lựa chọn đánh đổi chấp nhận được vì
catalogue demo có số lượng sản phẩm nhỏ.

## Điểm chưa chắc chắn / giả định cần lưu ý

0. **Thứ tự mount rất quan trọng — cần app.use() TRƯỚC catalogue hiện có.**
   `api/catalogue/index.js` hiện có route `app.get("/catalogue*", ...)`
   (wildcard) dùng để proxy thẳng mọi request `/catalogue/...` sang catalogue
   service. Nếu `server.js` mount `features_prosp` SAU `catalogue`
   (`app.use(catalogue)` rồi mới `app.use(require("./api/features_prosp"))`),
   thì mọi request tới `/catalogue/browse` sẽ bị route wildcard `/catalogue*`
   của module catalogue nuốt mất trước (vì Express khớp middleware theo thứ
   tự `app.use`), request sẽ bị forward thẳng tới
   `http://catalogue/catalogue/browse` trên catalogue service — service này
   không có endpoint đó nên sẽ trả lỗi/404, và route mới của mình sẽ KHÔNG
   BAO GIỜ được gọi tới. Vì yêu cầu đề bài là "KHÔNG sửa file nào khác" nên
   file này không thể tự sửa `server.js` để đảm bảo thứ tự — người tích hợp
   PHẢI đặt `app.use(require("./api/features_prosp"))` TRƯỚC
   `app.use(catalogue)` trong `server.js`. Đây là điểm rủi ro lớn nhất của
   việc bàn giao này, cần xác nhận lại khi ráp vào `server.js` thật.

1. **Không có trường `size` thật trong dữ liệu.** API_SURFACE.md xác nhận
   sản phẩm chỉ có mảng `tag`, không có field `size` riêng, và ví dụ danh
   sách tag hiện có (`brown, geek, formal, blue, skin, red, action, sport,
   black, magic, green`) không chứa giá trị nào giống size (small/medium/
   large...). Quyết định ở đây là: coi `size` như một filter cùng cơ chế với
   `color` — tức cũng so khớp vào `product.tag`. Đây là cách diễn giải hợp lý
   nhất có thể làm được với dữ liệu hiện tại, nhưng nó có nghĩa là **filter
   `size` sẽ luôn trả về rỗng cho tới khi catalogue service thực sự gắn tag
   kiểu size (vd "small", "large") cho sản phẩm**. Nếu backend dự định biểu
   diễn size theo cách khác (vd trong `description`, hoặc một service khác),
   cần xác nhận lại với team backend — route này không đoán thêm ngoài những
   gì API_SURFACE.md cho biết.
2. **Tag `sport` được hard-code làm bộ lọc gốc**, vì REQ-07 nói rõ tính năng
   là "duyệt danh mục sản phẩm với tất thể thao". Route không nhận tham số
   để đổi category khác — nếu sau này cần route tổng quát hơn (browse theo
   category bất kỳ), sẽ cần thiết kế lại (thêm param `category`/`tag`).
3. **Ngữ nghĩa AND giữa color/size, OR trong từng nhóm** là suy luận theo UX
   filter catalogue thông thường, REQ không nói rõ chi tiết này.
4. **Phân trang (`page`/`pageSize`) là phần thêm vào cho hợp lý**, không có
   trong REQ gốc; nếu không truyền, hành vi coi như không phân trang (trả về
   toàn bộ list đã lọc) để không phá vỡ kỳ vọng "trả về danh sách sản phẩm
   khớp bộ lọc" của REQ.
5. **Không gọi `GET /tags`** để validate giá trị `color`/`size` người dùng
   truyền vào có tồn tại hay không — route chỉ so khớp trực tiếp, giá trị lạ
   sẽ đơn giản không khớp sản phẩm nào (trả về mảng rỗng), không phải lỗi.
   Không gọi vì API_SURFACE.md nói rõ không có backend nào khác cần gọi cho
   tính năng này ngoài catalogue.
