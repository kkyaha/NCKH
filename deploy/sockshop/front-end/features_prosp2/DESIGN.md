# DESIGN

Quy ước: `CAT`=http://catalogue, `CARTS`=http://carts/carts, `USER`=http://user, `ORD`=http://orders.
Lỗi backend/mạng -> 502 JSON. Thiếu tham số -> 400. Chưa đăng nhập (`req.session.customerId` rỗng) -> 401.

## REQ-11 POST /login/quick
Bước: kiểm tra username/password -> gọi login Basic -> 401 nếu backend trả 401 -> gán `req.session.customerId`, gộp giỏ, đặt cookie `logged_in` -> 200 `{id, username}`.
Lời gọi backend theo thứ tự:
1. GET USER/login (header Basic)
2. GET CARTS/{customerId}/merge?sessionId={session.id} (lỗi chỉ log, không chặn đăng nhập)
Giả định: username lấy từ `user.username`; khi mã khác 200/401 -> 502.

## REQ-12 POST /register/quick
Bước: kiểm tra 3 trường -> đăng ký -> thành công thì đăng nhập luôn (session, gộp giỏ, cookie) -> 201 `{id, username}`. Thất bại: phân biệt trùng tên với lỗi hạ tầng.
Lời gọi:
1. POST USER/register `{username,password,email}`
2. (chỉ khi thất bại) GET USER/customers — tìm username trùng; có -> 409; không có mà backend trả 4xx -> 409; còn lại -> 502
3. (chỉ khi thành công) GET CARTS/{id}/merge?sessionId={session.id}
Giả định/chưa chắc: `/customers` có thể phân trang nên tên trùng ngoài trang đầu chỉ bắt được nếu backend trả 4xx.

## REQ-13 POST /wishlist (cần đăng nhập)
Không có dịch vụ wishlist; dùng dịch vụ carts với khoá riêng `wishlist-{customerId}` (tách hẳn giỏ thật, bền vững qua khởi động lại front-end). Kiểm tra tồn tại trước khi thêm để không nhân đôi (POST items sẽ tăng số lượng).
Lời gọi:
1. GET CAT/catalogue/{id} (404 -> 404)
2. GET CARTS/wishlist-{cid}/items
3. POST CARTS/wishlist-{cid}/items `{itemId, unitPrice}` (bỏ qua nếu đã có)
4. GET CARTS/wishlist-{cid}/items (đếm itemId khác nhau -> `wishlistCount`)
Giả định: check-then-add không nguyên tử (hai request đồng thời có thể tăng quantity, nhưng số itemId khác nhau vẫn đúng). Khoá `wishlist-...` không xung đột với customerId thật.

## REQ-14 GET /catalogue/search?q=
Bước: trim `q` (rỗng -> 400) -> lấy toàn bộ sản phẩm -> lọc substring không phân biệt hoa/thường trên name, description, từng tag -> sắp giá tăng -> trả `{query,count,results}`.
Lời gọi:
1. GET CAT/catalogue
Giả định: danh mục nhỏ (~9) nên lọc ở front-end; `imageUrl` và `tag` giữ nguyên dạng mảng từ catalogue; `query` là giá trị đã trim.

## REQ-15 GET /account/overview (cần đăng nhập)
Bước: bốn lời gọi song song -> ghép. Đơn sắp xếp theo ngày giảm dần ở front-end, lấy 3; thẻ chỉ giữ 4 số cuối (không lộ longNum/ccv); 404 -> mảng rỗng.
Lời gọi (song song, không có thứ tự cố định):
1. GET USER/customers/{id}
2. GET USER/customers/{id}/addresses
3. GET USER/customers/{id}/cards
4. GET ORD/orders/search/customerId?sort=date&custId={id}
Giả định: `customer` gồm id, username, firstName, lastName.

## REQ-16 GET /checkout/preview (cần đăng nhập)
Bước: lấy dòng giỏ, địa chỉ, danh mục (để có tên) -> tính thành tiền = quantity x unitPrice (đơn giá của giỏ), subtotal, shipping 4.99 nếu giỏ không rỗng, total (làm tròn 2 số lẻ) -> địa chỉ đầu tiên hoặc null. Chỉ đọc, không gọi POST /orders, không sửa giỏ/kho.
Lời gọi (song song):
1. GET CARTS/{customerId}/items (404 -> rỗng)
2. GET USER/customers/{id}/addresses
3. GET CAT/catalogue
Giả định: tên sản phẩm không có trong catalogue -> `name: null`; đơn giá lấy từ dòng giỏ chứ không phải giá catalogue hiện tại.

## Chung
- `req.body` do tầng trên parse (như các module hiện có); thiếu body coi như `{}`.
- Không chạy được mã; chưa được kiểm thử.
- `require` trỏ `../frontend_src/...` theo bố cục sandbox; khi gắn vào ứng dụng thật cần đổi thành `../endpoints` và `../../helpers` (helpers hiện chưa dùng nên có thể bỏ).
