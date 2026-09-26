# DESIGN

## REQ-17 `GET /orders/history?limit=<n>`

Backend calls, in order:
1. `GET http://orders/orders/search/customerId?sort=date&custId={req.session.customerId}` (only this one call).

Logic: 401 if no `req.session.customerId` (no backend call). `limit` defaults to 5, clamped to 1..20. Orders are re-sorted by date descending in the front-end and cut to `limit`. `itemCount` = sum of `items[].quantity`. `status` = `shipment.status` if present, else `null`. Backend 404 -> `orders: []`; other non-200 or network error -> 502.

Assumptions:
- The search result already embeds `items[]` (with `quantity`) and `shipment`, so no per-order `GET /orders/{id}` is needed.
- The shipment status, if any, is in `shipment.status`; the stock shipment record has no such field, so `status` will usually be `null`.
- Orders have no top-level `id`; `id` is taken from the last path segment of `_links.self.href` (fallback to `o.id` if present).
- Sort direction of the backend is not relied upon.
- `index.js` is placed as `api/<name>/index.js` so that `../endpoints` resolves.

## REQ-18 `GET /catalogue/related?id=<id>`

Backend calls, in order:
1. `GET http://catalogue/catalogue/{id}` (404 -> 404; other failure -> 502).
2. `GET http://catalogue/catalogue?size=100&tags={tags of the product, comma separated}` (skipped if the product has no tags; returns `related: []`).

Logic: 400 if `id` missing (no backend call). Candidates exclude the product itself; score = number of tags shared with the product; sort by score descending, then price ascending; keep 4.

Assumptions:
- The catalogue is small (~9 products), so `size=100` returns all matches in one page.
- Tag filtering in the catalogue is OR-matching, so it returns exactly the products with at least one shared tag.
- `imageUrl` and `tag` are passed through as arrays, as in the catalogue.
