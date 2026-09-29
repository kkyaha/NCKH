# DESIGN — REQ-19 và REQ-20

Module Express (`module.exports = app`) theo phong cách `frontend_src/api/*`: IIFE + `'use strict'`,
dùng `express`, `request`, `async` và `../endpoints`. Chỉ thêm route mới, không sửa file hiện có.
Giả định vị trí đặt file: `api/<tên thư mục>/index.js` trong cây front-end (vì `require("../endpoints")`),
và module được `app.use` **trước** `api/catalogue` và **trước** `api/orders` — bắt buộc, vì `api/orders`
có route bắt tất cả `GET /orders/*` sẽ nuốt mất `GET /orders/detailed`.

Quy ước chung đã cài: thiếu `req.session.customerId` → `401`; thiếu/sai tham số bắt buộc → `400`;
mọi lỗi từ backend (lỗi mạng, mã trạng thái lạ, JSON không parse được) → `502`; mọi phản hồi là JSON.

---

## REQ-19 — `GET /orders/detailed?limit=<n>`

### Thứ tự lời gọi backend

1. `GET http://orders/orders/search/customerId?custId=<custId>&sort=date`
   — 404 coi như khách chưa có đơn (`orders: []`, không gọi thêm gì nữa).
2. `GET http://catalogue/catalogue/{itemId}` — **một lần cho mỗi `itemId` khác nhau** xuất hiện trong các
   đơn đã chọn, chạy song song (`async.map`), theo thứ tự id xuất hiện lần đầu.
   Không có đơn nào hoặc đơn không có dòng hàng nào → bước này không có lời gọi.

Số lời gọi: `1 + số itemId khác nhau` (danh mục chỉ ~9 sản phẩm nên trần rất thấp; các đơn dùng chung
một sản phẩm chỉ tốn một lời gọi).

### Quyết định và giả định

- **`limit`**: mặc định 3; `> 10` bị kẹp về 10; không phải số nguyên ≥ 1 → `400`. `limit` không bắt buộc.
- **Chọn “đơn gần nhất”**: không tin vào chiều sắp xếp của `sort=date`, module tự sắp xếp giảm dần theo
  `Date.parse(order.date)` rồi cắt `limit` đơn đầu. `date` không parse được → coi như cũ nhất (giá trị 0).
- **`id` của đơn**: dịch vụ `orders` không trả trường `id` trong kết quả tìm kiếm, nên id được lấy từ
  đoạn cuối của `_links.self.href` (vẫn ưu tiên `order.id` nếu có).
- **`pricePaid`** = `unitPrice` của dòng hàng trong đơn; **`priceNow`** = `price` hiện tại trong danh mục.
- **`changed`** = `parseFloat(priceNow) !== parseFloat(pricePaid)` (so sánh theo số, tránh lệch kiểu
  chuỗi/số). Sản phẩm không còn trong danh mục (404) → `name: null`, `priceNow: null`, `changed: false`.
- **`imageUrl`**: danh mục trả mảng `imageUrl[]`, yêu cầu lại là một ảnh → lấy phần tử đầu tiên, không có
  thì `null`.
- **`description`**: schema trong REQ-19 không liệt kê trường này nhưng phần mô tả yêu cầu “mô tả ngắn”,
  nên mỗi dòng hàng có thêm `description` (`null` khi sản phẩm không còn trong danh mục). Các trường bắt
  buộc vẫn đúng tên và đúng nghĩa.
- Không gọi `carts`, `user`, `payment`: mọi dữ liệu giá đã trả nằm sẵn trong đơn.

### Phản hồi

`200 {count, orders: [{id, date, total, items: [{itemId, name, description, imageUrl, pricePaid, priceNow, changed}]}]}`
với `count` là số đơn trả về. Khách chưa có đơn → `200 {count: 0, orders: []}`.

---

## REQ-20 — `POST /orders/reorder` body `{ "orderId": "<id đơn cũ>" }`

### Thứ tự lời gọi backend

1. `GET http://orders/orders/{orderId}` — 404 → trả `404`; đơn có `customerId` khác khách đang đăng nhập
   → cũng trả `404` (không tiết lộ đơn của người khác). Dừng tại đây, không gọi tiếp.
2. `GET http://catalogue/catalogue/{itemId}` — một lần cho mỗi `itemId` khác nhau trong đơn cũ, song song.
   Sản phẩm 404 hoặc `count < 1` → bị loại. Loại hết → trả `409`, dừng (chưa hề ghi vào giỏ nào).
3. Với từng sản phẩm còn lại, **tuần tự** (`async.eachSeries`), vào **giỏ tạm** `<custId>-reorder-<timestamp>-<rand>`:
   - `POST http://carts/carts/<tempCartId>/items` body `{itemId, unitPrice}`
   - `PATCH http://carts/carts/<tempCartId>/items` body `{itemId, quantity, unitPrice}` — **chỉ khi**
     số lượng cần > 1 (POST đã tạo dòng với số lượng 1).
4. `GET http://user/customers/{custId}` — lấy `_links.customer`, `_links.addresses`, `_links.cards`.
5. Song song:
   - `GET <_links.addresses.href>` (`http://user/customers/{custId}/addresses`)
   - `GET <_links.cards.href>` (`http://user/customers/{custId}/cards`)
6. `POST http://orders/orders` body `{customer, address, card, items: "http://carts/carts/<tempCartId>/items"}`
   — `406` → trả `402`; `201` → thành công.
7. `DELETE http://carts/carts/<tempCartId>` — dọn giỏ tạm, chạy **trên mọi nhánh kết thúc** (thành công,
   402 hay 502) miễn là đã có lần ghi vào giỏ tạm; lỗi khi xoá chỉ ghi log, không đổi mã trạng thái.

Các bước 4–5 sao chép đúng cách `api/orders` (`POST /orders`) đang dựng đơn hàng, nên hành vi khi khách
chưa có địa chỉ/thẻ giống hệt route cũ (`address`/`card` để `null`).

### Quyết định và giả định

- **Không đụng giỏ thật**: đây là lý do dùng giỏ tạm. Theo `API_SURFACE.md`, `cartId` là “một khoá bất kỳ”
  và giỏ tự tạo khi ghi lần đầu, nên một khoá ngẫu nhiên gắn với `custId` cho ta một giỏ riêng. Giỏ của
  khách (`cartId == custId`) không hề bị đọc, ghi hay xoá. Giỏ tạm được xoá sau khi đơn đã tạo xong —
  dịch vụ `orders` đã đọc giỏ trong bước 6 nên xoá sau đó là an toàn.
- **Giá dùng lại**: `unitPrice` lấy theo **giá hiện tại trong danh mục**, không phải giá cũ trong đơn
  (mua lại là mua hôm nay). Điều này cũng khớp cách `api/cart` thêm hàng vào giỏ.
- **Số lượng**: giữ nguyên số lượng của đơn cũ, các dòng trùng `itemId` được cộng dồn, và bị kẹp theo tồn
  kho hiện có (`Math.min(quantity, item.count)`) để không đặt nhiều hơn số hàng đang có. Điều kiện loại
  sản phẩm vẫn đúng như yêu cầu: `count >= 1`.
- **`itemCount`** = số dòng hàng (số sản phẩm khác nhau) thực sự được đặt lại, tức số sản phẩm còn hàng;
  các sản phẩm hết hàng bị bỏ qua và không tính.
- **`total`** lấy từ phản hồi của `POST /orders` (dịch vụ orders tự tính); nếu phản hồi không có `total`
  thì tính dự phòng bằng `Σ unitPrice × quantity`.
- **`orderId`** lấy từ `id` của đơn mới, nếu không có thì từ đoạn cuối `_links.self.href`.
- **Không tự gọi `payment` hay `shipping`**: `POST /orders` đã tự thanh toán và tự lo vận chuyển; gọi thêm
  `/paymentAuth` sẽ là một lần ủy quyền thừa. `406` của orders chính là tín hiệu thanh toán bị từ chối → `402`.
- Thiếu/rỗng `orderId` trong body → `400` (giả định front-end đã gắn body-parser JSON như các route hiện có).

### Phản hồi

`201 {orderId, itemCount, total}` — `orderId` là id đơn **mới**.
`400` thiếu `orderId`; `401` chưa đăng nhập; `404` đơn cũ không tồn tại hoặc không thuộc khách này;
`409` mọi sản phẩm trong đơn cũ đều hết hàng; `402` thanh toán bị từ chối; `502` lỗi backend khác.
