(function (){
  'use strict';

  /* Bon tinh nang huong nguoi dung moi cho Sock Shop (nguon: data/benchmark/
   * parser_benchmark_prompts.json, REQ-01..04). Moi route goi truc tiep cac
   * backend THAT theo cach lap trinh vien se viet tu nhien; chuoi dich vu thuc
   * te duoc DO tu /metrics roi doi chieu voi taxonomy, khong ep cho khop.
   *
   *   POST /promo            APPLY_PROMO_CODE     carts -> orders -> payment
   *   GET  /recommendations  RECOMMEND_PRODUCTS   user -> orders -> catalogue
   *   GET  /track            TRACK_PACKAGE        orders -> shipping
   *   POST /reviews          WRITE_PRODUCT_REVIEW user -> catalogue
   */

  var async     = require("async")
    , express   = require("express")
    , request   = require("request")
    , endpoints = require("../endpoints")
    , helpers   = require("../../helpers")
    , app       = express()

  var paymentUrl  = "http://payment/paymentAuth";
  var shippingUrl = "http://shipping/shipping";
  var VOUCHER_PCT = 20;
  var reviews     = {};           // productId -> [review], giu trong bo nho front-end
  var MAX_REVIEWS = 50;

  function loggedInCustomer(req) {
    if (!req.cookies.logged_in) {
      throw new Error("User not logged in.");
    }
    return req.session.customerId;
  }

  function pastOrders(custId, callback) {
    request(endpoints.ordersUrl + "/orders/search/customerId?sort=date&custId=" + custId,
      function (error, response, body) {
        if (error) {
          return callback(error);
        }
        if (response.statusCode == 404) {
          return callback(null, []);
        }
        callback(null, JSON.parse(body)._embedded.customerOrders);
      });
  }

  // Ap dung ma giam gia: tinh tam tinh tu gio hang, kiem tra khach quay lai
  // (voucher chi danh cho khach da tung mua), pre-authorise so tien sau giam.
  app.post("/promo", function (req, res, next) {
    console.log("Promo request received with body: " + JSON.stringify(req.body));
    var custId = loggedInCustomer(req);

    async.waterfall([
      function (callback) {
        request(endpoints.cartsUrl + "/" + custId + "/items", function (error, response, body) {
          if (error) {
            return callback(error);
          }
          var subtotal = 0;
          JSON.parse(body).forEach(function (item) {
            subtotal += item.unitPrice * item.quantity;
          });
          callback(null, subtotal);
        });
      },
      function (subtotal, callback) {
        pastOrders(custId, function (error, orders) {
          callback(error, subtotal, orders && orders.length > 0);
        });
      },
      function (subtotal, eligible, callback) {
        var total = eligible ? subtotal * (100 - VOUCHER_PCT) / 100 : subtotal;
        var options = {
          uri: paymentUrl,
          method: 'POST',
          json: true,
          body: { amount: total }
        };
        request(options, function (error, response, body) {
          if (error) {
            return callback(error);
          }
          callback(null, {
            code: req.body.code,
            eligible: eligible,
            subtotal: subtotal,
            total: total,
            authorised: body && body.authorised
          });
        });
      }
    ],
    function (err, result) {
      if (err) {
        return next(err);
      }
      helpers.respondStatusBody(res, 200, JSON.stringify(result));
    });
  });

  // Goi y san pham: hoi nguoi dung, doc lich su mua, lay the cua mon da mua roi
  // tim mon cung the trong catalogue (loai tru mon da mua).
  app.get("/recommendations", function (req, res, next) {
    console.log("Recommendations request received");
    var custId = loggedInCustomer(req);

    async.waterfall([
      function (callback) {
        request(endpoints.customersUrl + "/" + custId, function (error, response, body) {
          callback(error);
        });
      },
      function (callback) {
        pastOrders(custId, callback);
      },
      function (orders, callback) {
        var bought = [];
        orders.forEach(function (order) {
          (order.items || []).forEach(function (item) {
            if (bought.indexOf(item.itemId) < 0) {
              bought.push(item.itemId);
            }
          });
        });
        if (bought.length === 0) {
          return callback(null, bought, []);
        }
        request(endpoints.catalogueUrl + "/catalogue/" + bought[0], function (error, response, body) {
          if (error) {
            return callback(error);
          }
          callback(null, bought, JSON.parse(body).tag || []);
        });
      },
      function (bought, tags, callback) {
        request(endpoints.catalogueUrl + "/catalogue?size=8&tags=" + tags.join(","),
          function (error, response, body) {
            if (error) {
              return callback(error);
            }
            var picks = JSON.parse(body).filter(function (product) {
              return bought.indexOf(product.id) < 0;
            });
            callback(null, picks.slice(0, 4));
          });
      }
    ],
    function (err, result) {
      if (err) {
        return next(err);
      }
      helpers.respondStatusBody(res, 200, JSON.stringify(result));
    });
  });

  // Theo doi goi hang: don gan nhat cua khach, roi hoi shipping ve van don do.
  app.get("/track", function (req, res, next) {
    console.log("Track request received");
    var custId = loggedInCustomer(req);

    async.waterfall([
      function (callback) {
        pastOrders(custId, callback);
      },
      function (orders, callback) {
        if (orders.length === 0) {
          return callback(null, { orders: 0 });
        }
        var latest = orders.reduce(function (a, b) {
          return a.date > b.date ? a : b;
        });
        var orderId = latest._links.self.href.split("/").pop();
        request(shippingUrl + "/" + orderId, function (error, response, body) {
          if (error) {
            return callback(error);
          }
          callback(null, { orders: orders.length, orderId: orderId, shipment: body });
        });
      }
    ],
    function (err, result) {
      if (err) {
        return next(err);
      }
      helpers.respondStatusBody(res, 200, JSON.stringify(result));
    });
  });

  // Danh gia san pham: xac minh tac gia (user) va san pham (catalogue), roi luu.
  app.post("/reviews", function (req, res, next) {
    console.log("Review request received with body: " + JSON.stringify(req.body));
    var custId = loggedInCustomer(req);
    var review = req.body;

    async.waterfall([
      function (callback) {
        request(endpoints.customersUrl + "/" + custId, function (error, response, body) {
          if (error) {
            return callback(error);
          }
          callback(null, JSON.parse(body).username);
        });
      },
      function (author, callback) {
        request(endpoints.catalogueUrl + "/catalogue/" + review.productId, function (error, response, body) {
          if (error) {
            return callback(error);
          }
          if (response.statusCode != 200) {
            return callback(null, author, false);
          }
          callback(null, author, true);
        });
      },
      function (author, exists, callback) {
        if (!exists) {
          return callback(null, 404, { error: "unknown product" });
        }
        var list = reviews[review.productId] = reviews[review.productId] || [];
        list.push({ author: author, stars: review.stars, text: review.text, date: Date.now() });
        if (list.length > MAX_REVIEWS) {
          list.shift();
        }
        callback(null, 201, { productId: review.productId, count: list.length });
      }
    ],
    function (err, status, result) {
      if (err) {
        return next(err);
      }
      helpers.respondStatusBody(res, status, JSON.stringify(result));
    });
  });

  module.exports = app;
}());
