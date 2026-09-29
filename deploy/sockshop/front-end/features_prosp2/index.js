(function (){
  'use strict';

  var async     = require("async")
    , express   = require("express")
    , request   = require("request")
    , endpoints = require("../endpoints")
    , helpers   = require("../../helpers")
    , app       = express()
    , cookie_name = "logged_in"
    , SHIPPING_FEE = 4.99

  // ---- tien ich chung ----------------------------------------------------

  function isStr(v) {
    return typeof v === "string" && v.trim() !== "";
  }

  function bad(res, msg) {
    return res.status(400).json({error: msg});
  }

  function backendFail(res, err) {
    console.log("Backend error: " + (err && err.message ? err.message : err));
    return res.status(502).json({error: "backend error"});
  }

  // Yeu cau dang nhap: tra ve customerId, hoac gui 401 va tra ve null.
  function requireLogin(req, res) {
    var custId = req.session && req.session.customerId;
    if (!custId) {
      res.status(401).json({error: "not logged in"});
      return null;
    }
    return custId;
  }

  function round2(n) {
    return Math.round(n * 100) / 100;
  }

  function parseJson(body) {
    if (body && typeof body === "object") return body;
    try { return JSON.parse(body); } catch (e) { return null; }
  }

  // GET JSON; callback(err, statusCode, data)
  function getJson(url, callback) {
    request.get(url, function (error, response, body) {
      if (error) return callback(error);
      callback(null, response.statusCode, parseJson(body));
    });
  }

  // Gop gio khoa phien vao gio khach; loi gop gio khong chan dang nhap.
  function mergeCarts(req, custId, callback) {
    var uri = endpoints.cartsUrl + "/" + custId + "/merge?sessionId=" + req.session.id;
    request({uri: uri, method: 'GET'}, function (error) {
      if (error) console.log("Cart merge failed: " + error);
      callback();
    });
  }

  function startSession(req, res, custId, callback) {
    req.session.customerId = custId;
    mergeCarts(req, custId, function () {
      res.cookie(cookie_name, req.session.id, {maxAge: 3600000});
      callback();
    });
  }

  function embedded(data, key) {
    if (data && data._embedded && Array.isArray(data._embedded[key])) {
      return data._embedded[key];
    }
    return [];
  }

  // Lay danh sach HAL (dia chi / the / don) cua khach; 404 -> [].
  function getEmbedded(url, key, callback) {
    getJson(url, function (err, status, data) {
      if (err) return callback(err);
      if (status === 404) return callback(null, []);
      if (status !== 200) return callback(new Error("status " + status + " from " + url));
      callback(null, embedded(data, key));
    });
  }

  function getCartItems(cartId, callback) {
    getJson(endpoints.cartsUrl + "/" + cartId + "/items", function (err, status, data) {
      if (err) return callback(err);
      if (status === 404) return callback(null, []);
      if (status !== 200 || !Array.isArray(data)) {
        return callback(new Error("cart status " + status));
      }
      callback(null, data);
    });
  }

  // ---- REQ-11 ------------------------------------------------------------
  app.post("/login/quick", function (req, res) {
    var body = req.body || {};
    if (!isStr(body.username) || !isStr(body.password)) {
      return bad(res, "username and password required");
    }
    var auth = "Basic " + Buffer.from(body.username + ":" + body.password).toString("base64");

    request({uri: endpoints.loginUrl, headers: {'Authorization': auth}}, function (error, response, rbody) {
      if (error) return backendFail(res, error);
      if (response.statusCode === 401) {
        return res.status(401).json({error: "invalid credentials"});
      }
      var data = parseJson(rbody);
      if (response.statusCode !== 200 || !data || !data.user || !data.user.id) {
        return backendFail(res, new Error("login status " + response.statusCode));
      }
      var user = data.user;
      startSession(req, res, user.id, function () {
        res.status(200).json({id: user.id, username: user.username || body.username});
      });
    });
  });

  // ---- REQ-12 ------------------------------------------------------------
  app.post("/register/quick", function (req, res) {
    var body = req.body || {};
    if (!isStr(body.username) || !isStr(body.password) || !isStr(body.email)) {
      return bad(res, "username, password and email required");
    }
    var payload = {username: body.username, password: body.password, email: body.email};

    request({uri: endpoints.registerUrl, method: 'POST', json: true, body: payload}, function (error, response, rbody) {
      if (error) return backendFail(res, error);

      var ok = response.statusCode === 200 && rbody && !rbody.error && rbody.id;
      if (ok) {
        return startSession(req, res, rbody.id, function () {
          res.status(201).json({id: rbody.id, username: body.username});
        });
      }

      // That bai: phan biet "ten da ton tai" (409) voi loi ha tang (502).
      getJson(endpoints.customersUrl, function (err, status, data) {
        var exists = !err && status === 200 && embedded(data, "customer").some(function (c) {
          return c.username === body.username;
        });
        if (exists || (response.statusCode >= 400 && response.statusCode < 500)) {
          return res.status(409).json({error: "username already exists"});
        }
        return backendFail(res, new Error("register status " + response.statusCode));
      });
    });
  });

  // ---- REQ-13 ------------------------------------------------------------
  // Danh sach yeu thich luu trong dich vu carts duoi khoa rieng "wishlist-<customerId>",
  // tach biet voi gio that (khoa = customerId).
  app.post("/wishlist", function (req, res) {
    var custId = requireLogin(req, res);
    if (!custId) return;
    var body = req.body || {};
    if (body.id == null || String(body.id).trim() === "") {
      return bad(res, "id required");
    }
    var listId = "wishlist-" + custId;
    var itemsUrl = endpoints.cartsUrl + "/" + listId + "/items";

    async.waterfall([
      function (callback) {
        getJson(endpoints.catalogueUrl + "/catalogue/" + encodeURIComponent(String(body.id)), function (err, status, item) {
          if (err) return callback(err);
          if (status === 404) return callback(null, null);
          if (status !== 200 || !item || item.id == null) {
            return callback(new Error("catalogue status " + status));
          }
          callback(null, item);
        });
      },
      function (item, callback) {
        if (!item) return callback(null, null);
        getCartItems(listId, function (err, items) {
          if (err) return callback(err);
          var has = items.some(function (i) { return String(i.itemId) === String(item.id); });
          if (has) return callback(null, item);
          request({uri: itemsUrl, method: 'POST', json: true, body: {itemId: item.id, unitPrice: item.price}},
            function (error, response) {
              if (error) return callback(error);
              if (response.statusCode !== 201) return callback(new Error("wishlist add status " + response.statusCode));
              callback(null, item);
            });
        });
      },
      function (item, callback) {
        if (!item) return callback(null, null, 0);
        getCartItems(listId, function (err, items) {
          if (err) return callback(err);
          var seen = {};
          items.forEach(function (i) { seen[String(i.itemId)] = true; });
          callback(null, item, Object.keys(seen).length);
        });
      }
    ], function (err, item, count) {
      if (err) return backendFail(res, err);
      if (!item) return res.status(404).json({error: "product not found"});
      res.status(201).json({
        added: {id: item.id, name: item.name, price: item.price},
        wishlistCount: count
      });
    });
  });

  // ---- REQ-14 ------------------------------------------------------------
  app.get("/catalogue/search", function (req, res) {
    var q = req.query.q;
    if (typeof q !== "string" || q.trim() === "") {
      return bad(res, "q required");
    }
    q = q.trim();
    var needle = q.toLowerCase();

    getJson(endpoints.catalogueUrl + "/catalogue", function (err, status, data) {
      if (err) return backendFail(res, err);
      if (status !== 200 || !Array.isArray(data)) {
        return backendFail(res, new Error("catalogue status " + status));
      }
      function has(s) {
        return typeof s === "string" && s.toLowerCase().indexOf(needle) !== -1;
      }
      var results = data.filter(function (p) {
        return has(p.name) || has(p.description) || (Array.isArray(p.tag) && p.tag.some(has));
      }).sort(function (a, b) {
        return a.price - b.price;
      }).map(function (p) {
        return {id: p.id, name: p.name, price: p.price, imageUrl: p.imageUrl, tag: p.tag};
      });
      res.status(200).json({query: q, count: results.length, results: results});
    });
  });

  // ---- REQ-15 ------------------------------------------------------------
  app.get("/account/overview", function (req, res) {
    var custId = requireLogin(req, res);
    if (!custId) return;
    var base = endpoints.customersUrl + "/" + custId;

    async.parallel({
      customer: function (callback) {
        getJson(base, function (err, status, data) {
          if (err) return callback(err);
          if (status !== 200 || !data) return callback(new Error("customer status " + status));
          callback(null, data);
        });
      },
      addresses: function (callback) {
        getEmbedded(base + "/addresses", "address", callback);
      },
      cards: function (callback) {
        getEmbedded(base + "/cards", "card", callback);
      },
      orders: function (callback) {
        getEmbedded(endpoints.ordersUrl + "/orders/search/customerId?sort=date&custId=" + encodeURIComponent(custId),
          "customerOrders", callback);
      }
    }, function (err, r) {
      if (err) return backendFail(res, err);
      var c = r.customer;
      var recent = r.orders.slice().sort(function (a, b) {
        return new Date(b.date) - new Date(a.date);
      }).slice(0, 3).map(function (o) {
        return {id: o.id, date: o.date, total: o.total};
      });
      res.status(200).json({
        customer: {id: c.id || custId, username: c.username, firstName: c.firstName, lastName: c.lastName},
        addresses: r.addresses.map(function (a) {
          return {id: a.id, street: a.street, number: a.number, city: a.city, postcode: a.postcode, country: a.country};
        }),
        cards: r.cards.map(function (k) {
          return {last4: String(k.longNum || "").slice(-4), expires: k.expires};
        }),
        recentOrders: recent
      });
    });
  });

  // ---- REQ-16 ------------------------------------------------------------
  app.get("/checkout/preview", function (req, res) {
    var custId = requireLogin(req, res);
    if (!custId) return;

    async.parallel({
      items: function (callback) {
        getCartItems(custId, callback);
      },
      addresses: function (callback) {
        getEmbedded(endpoints.customersUrl + "/" + custId + "/addresses", "address", callback);
      },
      catalogue: function (callback) {
        getJson(endpoints.catalogueUrl + "/catalogue", function (err, status, data) {
          if (err) return callback(err);
          if (status !== 200 || !Array.isArray(data)) return callback(new Error("catalogue status " + status));
          callback(null, data);
        });
      }
    }, function (err, r) {
      if (err) return backendFail(res, err);
      var names = {};
      r.catalogue.forEach(function (p) { names[String(p.id)] = p.name; });

      var subtotal = 0;
      var items = r.items.map(function (i) {
        var lineTotal = round2(i.quantity * i.unitPrice);
        subtotal += lineTotal;
        return {
          id: i.itemId,
          name: names[String(i.itemId)] || null,
          quantity: i.quantity,
          unitPrice: i.unitPrice,
          lineTotal: lineTotal
        };
      });
      subtotal = round2(subtotal);
      var shipping = items.length > 0 ? SHIPPING_FEE : 0;
      var a = r.addresses[0];
      res.status(200).json({
        items: items,
        subtotal: subtotal,
        shipping: shipping,
        total: round2(subtotal + shipping),
        address: a ? {id: a.id, street: a.street, number: a.number, city: a.city, postcode: a.postcode, country: a.country} : null
      });
    });
  });

  module.exports = app;
}());
