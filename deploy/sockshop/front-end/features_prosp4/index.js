(function (){
  'use strict';

  var async     = require("async")
    , express   = require("express")
    , request   = require("request")
    , endpoints = require("../endpoints")
    , app       = express()

  var DEFAULT_LIMIT = 3
    , MAX_LIMIT     = 10

  /* Trả JSON kèm mã trạng thái. */
  function respondJson(res, statusCode, body) {
    res.status(statusCode).json(body);
  }

  /* Hai route đều cần đăng nhập: chỉ chấp nhận req.session.customerId. */
  function loggedInCustomer(req, res) {
    var custId = req.session != null ? req.session.customerId : null;
    if (custId == null || custId === "") {
      respondJson(res, 401, {message: "User not logged in."});
      return null;
    }
    return custId;
  }

  function parseJson(body) {
    try {
      return JSON.parse(body);
    } catch (e) {
      return null;
    }
  }

  /* Dịch vụ orders không trả trường `id`; id nằm cuối _links.self.href. */
  function orderIdOf(order) {
    if (order == null) {
      return null;
    }
    if (order.id != null) {
      return order.id;
    }
    if (order._links != null && order._links.self != null && order._links.self.href != null) {
      var href = order._links.self.href.toString();
      return href.substring(href.lastIndexOf("/") + 1);
    }
    return null;
  }

  function dateValue(order) {
    var t = Date.parse(order != null ? order.date : null);
    return isNaN(t) ? 0 : t;
  }

  function samePrice(a, b) {
    return parseFloat(a) === parseFloat(b);
  }

  /* GET /catalogue/{id}. callback(null, null) khi sản phẩm không còn trong danh mục. */
  function fetchCatalogueItem(itemId, callback) {
    var url = endpoints.catalogueUrl + "/catalogue/" + itemId;
    console.log("GET Request to: " + url);
    request.get(url, function (error, response, body) {
      if (error) {
        return callback(error);
      }
      if (response.statusCode == 404) {
        return callback(null, null);
      }
      if (response.statusCode != 200) {
        return callback(new Error("Catalogue returned status " + response.statusCode + " for item " + itemId));
      }
      var item = parseJson(body);
      if (item == null || item.id == null) {
        return callback(null, null);
      }
      callback(null, item);
    });
  }

  // REQ-19: lịch sử mua hàng đầy đủ, kèm giá đã trả và giá hiện tại.
  // Phải được gắn TRƯỚC api/orders vì ở đó có route bắt tất cả GET /orders/*.
  app.get("/orders/detailed", function (req, res, next) {
    console.log("Request received: " + req.url);

    var custId = loggedInCustomer(req, res);
    if (custId == null) {
      return;
    }

    var limit = DEFAULT_LIMIT;
    if (req.query.limit != null && req.query.limit !== "") {
      limit = parseInt(req.query.limit, 10);
      if (isNaN(limit) || limit < 1) {
        return respondJson(res, 400, {message: "limit must be a positive integer"});
      }
      if (limit > MAX_LIMIT) {
        limit = MAX_LIMIT;
      }
    }

    async.waterfall([
        // 1. Các đơn của khách.
        function (callback) {
          var url = endpoints.ordersUrl + "/orders/search/customerId?custId=" + custId + "&sort=date";
          console.log("GET Request to: " + url);
          request.get(url, function (error, response, body) {
            if (error) {
              return callback(error);
            }
            if (response.statusCode == 404) {
              console.log("No orders found for user: " + custId);
              return callback(null, []);
            }
            if (response.statusCode != 200) {
              return callback(new Error("Orders returned status " + response.statusCode));
            }
            var json = parseJson(body);
            if (json == null) {
              return callback(new Error("Could not parse orders response"));
            }
            var orders = json._embedded != null ? json._embedded.customerOrders : null;
            callback(null, orders || []);
          });
        },
        // 2. Giữ lại các đơn gần nhất.
        function (orders, callback) {
          orders.sort(function (a, b) {
            return dateValue(b) - dateValue(a);
          });
          callback(null, orders.slice(0, limit));
        },
        // 3. Tra danh mục, mỗi itemId đúng một lần.
        function (orders, callback) {
          var itemIds = [];
          orders.forEach(function (order) {
            (order.items || []).forEach(function (line) {
              if (line != null && line.itemId != null && itemIds.indexOf(line.itemId) === -1) {
                itemIds.push(line.itemId);
              }
            });
          });
          async.map(itemIds, fetchCatalogueItem, function (err, items) {
            if (err) {
              return callback(err);
            }
            var catalogue = {};
            itemIds.forEach(function (id, i) {
              catalogue[id] = items[i];
            });
            callback(null, orders, catalogue);
          });
        }
    ],
    function (err, orders, catalogue) {
      if (err) {
        console.log("Error building detailed order history: " + err);
        return respondJson(res, 502, {message: "Could not read order history."});
      }

      var detailed = orders.map(function (order) {
        return {
          id: orderIdOf(order),
          date: order.date,
          total: order.total,
          items: (order.items || []).map(function (line) {
            var item      = catalogue[line.itemId] || null
              , pricePaid = line.unitPrice
              , priceNow  = item != null ? item.price : null

            return {
              itemId:      line.itemId,
              name:        item != null ? item.name : null,
              description: item != null ? item.description : null,
              imageUrl:    item != null && item.imageUrl != null ? (item.imageUrl[0] || null) : null,
              pricePaid:   pricePaid,
              priceNow:    priceNow,
              changed:     item != null ? !samePrice(priceNow, pricePaid) : false
            };
          })
        };
      });

      respondJson(res, 200, {count: detailed.length, orders: detailed});
    });
  });

  // REQ-20: mua lại một đơn cũ, không đụng tới giỏ hàng thật của khách.
  app.post("/orders/reorder", function (req, res, next) {
    console.log("Request received with body: " + JSON.stringify(req.body));

    var custId = loggedInCustomer(req, res);
    if (custId == null) {
      return;
    }

    var orderId = req.body != null ? req.body.orderId : null;
    if (orderId == null || orderId === "") {
      return respondJson(res, 400, {message: "Must pass orderId of the order to reorder"});
    }

    // Giỏ tạm: khoá giỏ là chuỗi bất kỳ nên giỏ thật của khách không bị ảnh hưởng.
    var tempCartId    = custId + "-reorder-" + Date.now() + "-" + Math.floor(Math.random() * 100000)
      , tempCartUrl   = endpoints.cartsUrl + "/" + tempCartId
      , tempItemsUrl  = tempCartUrl + "/items"
      , tempCartUsed  = false

    function cleanupThen(done) {
      if (!tempCartUsed) {
        return done();
      }
      console.log("Deleting temporary cart: " + tempCartUrl);
      request({uri: tempCartUrl, method: 'DELETE'}, function (error) {
        if (error) {
          console.log("Could not delete temporary cart " + tempCartId + ": " + error);
        }
        done();
      });
    }

    function finish(statusCode, body) {
      cleanupThen(function () {
        respondJson(res, statusCode, body);
      });
    }

    async.waterfall([
        // 1. Đơn cũ, và đơn đó phải thuộc về khách này.
        function (callback) {
          var url = endpoints.ordersUrl + "/orders/" + orderId;
          console.log("GET Request to: " + url);
          request.get(url, function (error, response, body) {
            if (error) {
              return callback({status: 502, message: "Could not read the original order."});
            }
            if (response.statusCode == 404) {
              return callback({status: 404, message: "Order not found."});
            }
            if (response.statusCode != 200) {
              return callback({status: 502, message: "Could not read the original order."});
            }
            var order = parseJson(body);
            if (order == null) {
              return callback({status: 502, message: "Could not read the original order."});
            }
            var owner = order.customerId != null ? order.customerId
                                                 : (order.customer != null ? order.customer.id : null);
            if (owner == null || owner.toString() !== custId.toString()) {
              return callback({status: 404, message: "Order not found."});
            }
            callback(null, order.items || []);
          });
        },
        // 2. Sản phẩm nào còn trong danh mục và còn hàng.
        function (lines, callback) {
          var wanted = [];
          lines.forEach(function (line) {
            if (line == null || line.itemId == null) {
              return;
            }
            var seen = null;
            wanted.forEach(function (w) {
              if (w.itemId === line.itemId) {
                seen = w;
              }
            });
            if (seen != null) {
              seen.quantity += (line.quantity || 1);
            } else {
              wanted.push({itemId: line.itemId, quantity: line.quantity || 1});
            }
          });

          async.map(wanted, function (line, cb) {
            fetchCatalogueItem(line.itemId, function (err, item) {
              if (err) {
                return cb(err);
              }
              if (item == null || !(item.count >= 1)) {
                console.log("Skipping unavailable item: " + line.itemId);
                return cb(null, null);
              }
              cb(null, {
                itemId:    item.id,
                unitPrice: item.price,
                quantity:  Math.min(line.quantity, item.count)
              });
            });
          }, function (err, results) {
            if (err) {
              return callback({status: 502, message: "Could not check the catalogue."});
            }
            var available = results.filter(function (r) { return r != null; });
            if (available.length === 0) {
              return callback({status: 409, message: "No item of this order is available any more."});
            }
            callback(null, available);
          });
        },
        // 3. Đưa vào giỏ tạm.
        function (available, callback) {
          async.eachSeries(available, function (line, cb) {
            var options = {
              uri: tempItemsUrl,
              method: 'POST',
              json: true,
              body: {itemId: line.itemId, unitPrice: line.unitPrice}
            };
            console.log("POST to carts: " + options.uri + " body: " + JSON.stringify(options.body));
            tempCartUsed = true;
            request(options, function (error, response) {
              if (error) {
                return cb(error);
              }
              if (response.statusCode != 201) {
                return cb(new Error("Carts returned status " + response.statusCode));
              }
              if (line.quantity <= 1) {
                return cb();
              }
              var patch = {
                uri: tempItemsUrl,
                method: 'PATCH',
                json: true,
                body: {itemId: line.itemId, quantity: line.quantity, unitPrice: line.unitPrice}
              };
              console.log("PATCH to carts: " + patch.uri + " body: " + JSON.stringify(patch.body));
              request(patch, function (patchError, patchResponse) {
                if (patchError) {
                  return cb(patchError);
                }
                if (patchResponse.statusCode >= 400) {
                  return cb(new Error("Carts returned status " + patchResponse.statusCode));
                }
                cb();
              });
            });
          }, function (err) {
            if (err) {
              console.log("Could not fill the temporary cart: " + err);
              return callback({status: 502, message: "Could not prepare the new order."});
            }
            callback(null, available);
          });
        },
        // 4. Khách hàng, để lấy link customer/address/card.
        function (available, callback) {
          var url = endpoints.customersUrl + "/" + custId;
          console.log("GET Request to: " + url);
          request.get(url, function (error, response, body) {
            if (error || response.statusCode != 200) {
              return callback({status: 502, message: "Could not read the customer."});
            }
            var jsonBody = parseJson(body);
            if (jsonBody == null || jsonBody._links == null) {
              return callback({status: 502, message: "Could not read the customer."});
            }
            var links = jsonBody._links
              , order = {
                  "customer": (links.customer != null ? links.customer : links.self).href,
                  "address": null,
                  "card": null,
                  "items": tempItemsUrl
                }

            callback(null, available, order, links.addresses.href, links.cards.href);
          });
        },
        // 5. Địa chỉ và thẻ.
        function (available, order, addressLink, cardLink, callback) {
          async.parallel([
              function (cb) {
                console.log("GET Request to: " + addressLink);
                request.get(addressLink, function (error, response, body) {
                  if (error) {
                    return cb(error);
                  }
                  var jsonBody = parseJson(body);
                  if (jsonBody != null && jsonBody._embedded != null && jsonBody._embedded.address[0] != null) {
                    order.address = jsonBody._embedded.address[0]._links.self.href;
                  }
                  cb();
                });
              },
              function (cb) {
                console.log("GET Request to: " + cardLink);
                request.get(cardLink, function (error, response, body) {
                  if (error) {
                    return cb(error);
                  }
                  var jsonBody = parseJson(body);
                  if (jsonBody != null && jsonBody._embedded != null && jsonBody._embedded.card[0] != null) {
                    order.card = jsonBody._embedded.card[0]._links.self.href;
                  }
                  cb();
                });
              }
          ], function (err) {
            if (err) {
              return callback({status: 502, message: "Could not read address or card."});
            }
            callback(null, available, order);
          });
        },
        // 6. Tạo đơn mới từ giỏ tạm.
        function (available, order, callback) {
          var options = {
            uri: endpoints.ordersUrl + '/orders',
            method: 'POST',
            json: true,
            body: order
          };
          console.log("Posting Order: " + JSON.stringify(order));
          request(options, function (error, response, body) {
            if (error) {
              return callback({status: 502, message: "Could not create the new order."});
            }
            if (response.statusCode == 406) {
              return callback({status: 402, message: "Payment declined."});
            }
            if (response.statusCode != 201 && response.statusCode != 200) {
              return callback({status: 502, message: "Could not create the new order."});
            }
            callback(null, available, body);
          });
        }
    ],
    function (err, available, newOrder) {
      if (err) {
        return finish(err.status || 502, {message: err.message || "Could not reorder."});
      }

      var total = newOrder != null && newOrder.total != null
        ? newOrder.total
        : available.reduce(function (sum, line) { return sum + line.unitPrice * line.quantity; }, 0);

      finish(201, {
        orderId:   orderIdOf(newOrder),
        itemCount: available.length,
        total:     total
      });
    });
  });

  module.exports = app;
}());
