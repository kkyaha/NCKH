(function (){
  'use strict';

  var async     = require("async")
    , express   = require("express")
    , request   = require("request")
    , helpers   = require("../../helpers")
    , endpoints = require("../endpoints")
    , app       = express()

  var BACKEND_TIMEOUT = 15000;

  // ---------------------------------------------------------------- utilities

  function httpError(status, message) {
    var err = new Error(message);
    err.status = status;
    return err;
  }

  // Calls a backend service. Network errors become 502 errors, so that the
  // callback always receives either an error carrying an HTTP status or a response.
  function callBackend(options, callback) {
    options.timeout = BACKEND_TIMEOUT;
    request(options, function (error, response, body) {
      if (error) {
        console.log("Backend call failed: " + options.method + " " + options.uri + ": " + error);
        return callback(httpError(502, "Backend unavailable: " + options.uri));
      }
      callback(null, response, body);
    });
  }

  // Parses a JSON body if it is still a string; returns undefined on failure.
  function parseJson(body) {
    if (typeof body !== "string") {
      return body;
    }
    try {
      return JSON.parse(body);
    } catch (e) {
      return undefined;
    }
  }

  // Product id coming from the request body: accepts a non-empty string or a number.
  function getProductId(req) {
    var id = req.body ? req.body.id : null;
    if (typeof id === "number") {
      id = id.toString();
    }
    if (typeof id !== "string" || id.length === 0 || id.length > 200) {
      return null;
    }
    return id;
  }

  // Customer id for the cart routes (same rules as the existing /cart routes).
  function getCartOwner(req) {
    try {
      var custId = helpers.getCustomerId(req, app.get("env"));
      if (custId == null) {
        return null;
      }
      return custId;
    } catch (e) {
      return null;
    }
  }

  // Loads a product from the catalogue. Calls back with (err, product).
  function fetchProduct(productId, callback) {
    var options = {
      uri: endpoints.catalogueUrl + "/catalogue/" + encodeURIComponent(productId),
      method: 'GET'
    };
    callBackend(options, function (err, response, body) {
      if (err) {
        return callback(err);
      }
      if (response.statusCode == 404) {
        return callback(httpError(404, "Product not found: " + productId));
      }
      if (response.statusCode != 200) {
        return callback(httpError(502, "Catalogue returned status " + response.statusCode));
      }
      var item = parseJson(body);
      if (!item || item.id == null || typeof item.price !== "number") {
        return callback(httpError(502, "Catalogue returned an invalid product"));
      }
      callback(null, item);
    });
  }

  // Loads the items of a cart. Calls back with (err, itemsArray). A missing cart is empty.
  function fetchCartItems(cartId, callback) {
    var options = {
      uri: endpoints.cartsUrl + "/" + encodeURIComponent(cartId) + "/items",
      method: 'GET'
    };
    callBackend(options, function (err, response, body) {
      if (err) {
        return callback(err);
      }
      if (response.statusCode == 404) {
        return callback(null, []);
      }
      if (response.statusCode != 200) {
        return callback(httpError(502, "Carts returned status " + response.statusCode));
      }
      var items = parseJson(body);
      if (!Array.isArray(items)) {
        return callback(httpError(502, "Carts returned an invalid item list"));
      }
      callback(null, items);
    });
  }

  // Adds one unit of a product to a cart. Calls back with (err).
  function addToCart(cartId, item, callback) {
    var options = {
      uri: endpoints.cartsUrl + "/" + encodeURIComponent(cartId) + "/items",
      method: 'POST',
      json: true,
      body: {itemId: item.id, unitPrice: item.price}
    };
    console.log("POST to carts: " + options.uri + " body: " + JSON.stringify(options.body));
    callBackend(options, function (err, response, body) {
      if (err) {
        return callback(err);
      }
      if (response.statusCode != 201) {
        return callback(httpError(502, "Unable to add to cart. Status code: " + response.statusCode));
      }
      callback(null);
    });
  }

  // Deletes a whole cart. Errors are only logged.
  function deleteCart(cartId, callback) {
    var options = {
      uri: endpoints.cartsUrl + "/" + encodeURIComponent(cartId),
      method: 'DELETE'
    };
    callBackend(options, function (err, response, body) {
      if (err) {
        console.log("Unable to delete cart " + cartId + ": " + err.message);
      } else {
        console.log("Cart " + cartId + " deleted with status: " + response.statusCode);
      }
      callback();
    });
  }

  function toCents(price) {
    return Math.round(Number(price) * 100);
  }

  // ---------------------------------------------------------------- REQ-06

  // Cart contents (with product details) and provisional total.
  app.get("/cart/summary", function (req, res, next) {
    console.log("Request received: " + req.url);
    var custId = getCartOwner(req);
    if (custId == null) {
      return next(httpError(401, "User not logged in."));
    }
    console.log("Customer ID: " + custId);

    async.waterfall([
        function (callback) {
          fetchCartItems(custId, callback);
        },
        function (cartItems, callback) {
          // Product details for every line, in parallel. A failing lookup does not
          // hide the line: the cart data itself is enough to price it.
          async.map(cartItems, function (line, done) {
            var quantity = parseInt(line.quantity, 10);
            if (isNaN(quantity) || quantity < 0) {
              quantity = 0;
            }
            var unitPrice = Number(line.unitPrice);
            if (isNaN(unitPrice)) {
              unitPrice = 0;
            }
            var entry = {
              itemId: line.itemId,
              name: null,
              imageUrl: null,
              quantity: quantity,
              unitPrice: unitPrice,
              lineTotal: (toCents(unitPrice) * quantity) / 100
            };
            fetchProduct(String(line.itemId), function (err, product) {
              if (err) {
                console.log("No product details for item " + line.itemId + ": " + err.message);
              } else {
                entry.name = product.name;
                entry.description = product.description;
                entry.imageUrl = (product.imageUrl && product.imageUrl.length > 0) ? product.imageUrl[0] : null;
                entry.tag = product.tag;
              }
              done(null, entry);
            });
          }, callback);
        }
    ], function (err, entries) {
      if (err) {
        return next(err);
      }
      var cents = 0;
      var totalQuantity = 0;
      entries.forEach(function (entry) {
        cents += toCents(entry.unitPrice) * entry.quantity;
        totalQuantity += entry.quantity;
      });
      res.status(200).json({
        items: entries,
        itemCount: entries.length,
        totalQuantity: totalQuantity,
        subtotal: cents / 100
      });
    });
  });

  // ---------------------------------------------------------------- REQ-05

  // Quick add: one unit of the product goes into the customer's cart.
  app.post("/cart/quick", function (req, res, next) {
    console.log("Attempting to quick add to cart: " + JSON.stringify(req.body));

    var productId = getProductId(req);
    if (productId == null) {
      return next(httpError(400, "Must pass id of item to add"));
    }
    var custId = getCartOwner(req);
    if (custId == null) {
      return next(httpError(401, "User not logged in."));
    }

    async.waterfall([
        function (callback) {
          fetchProduct(productId, callback);
        },
        function (item, callback) {
          addToCart(custId, item, function (err) {
            callback(err, item);
          });
        },
        function (item, callback) {
          // Report the new state of the cart; the add itself already succeeded.
          fetchCartItems(custId, function (err, cartItems) {
            if (err) {
              console.log("Item added but cart could not be read back: " + err.message);
              return callback(null, item, null);
            }
            callback(null, item, cartItems);
          });
        }
    ], function (err, item, cartItems) {
      if (err) {
        return next(err);
      }
      var result = {
        added: {itemId: item.id, name: item.name, unitPrice: item.price, quantity: 1},
        itemCount: null,
        totalQuantity: null
      };
      if (cartItems) {
        var totalQuantity = 0;
        cartItems.forEach(function (line) {
          var q = parseInt(line.quantity, 10);
          totalQuantity += isNaN(q) ? 0 : q;
        });
        result.itemCount = cartItems.length;
        result.totalQuantity = totalQuantity;
      }
      res.status(201).json(result);
    });
  });

  // ---------------------------------------------------------------- REQ-10

  // Express checkout: buy exactly one product with the customer's saved address and card.
  // The purchase is made from a private, temporary cart, so the customer's own cart is
  // neither bought by accident nor polluted with the express item.
  app.post("/checkout/express", function (req, res, next) {
    console.log("Express checkout request received with body: " + JSON.stringify(req.body));

    var productId = getProductId(req);
    if (productId == null) {
      return next(httpError(400, "Must pass id of item to buy"));
    }
    if (!req.cookies || !req.cookies.logged_in || req.session.customerId == null) {
      return next(httpError(401, "User not logged in."));
    }
    var custId = req.session.customerId;

    var tempCartId = "express-" + custId + "-" + Date.now() + "-" + Math.floor(Math.random() * 1000000);
    var tempCartUsed = false;

    async.waterfall([
        // 1. Product must exist and be in stock.
        function (callback) {
          fetchProduct(productId, function (err, item) {
            if (err) {
              return callback(err);
            }
            if (typeof item.count === "number" && item.count < 1) {
              return callback(httpError(409, "Product out of stock: " + productId));
            }
            callback(null, item);
          });
        },
        // 2. Customer record: links to customer, addresses and cards.
        function (item, callback) {
          var options = {
            uri: endpoints.customersUrl + "/" + encodeURIComponent(custId),
            method: 'GET'
          };
          callBackend(options, function (err, response, body) {
            if (err) {
              return callback(err);
            }
            if (response.statusCode == 404) {
              return callback(httpError(404, "Customer not found"));
            }
            var customer = parseJson(body);
            if (response.statusCode != 200 || !customer || !customer._links ||
                !customer._links.customer || !customer._links.addresses || !customer._links.cards) {
              return callback(httpError(502, "User service returned an invalid customer"));
            }
            callback(null, item, {
              customer: customer._links.customer.href,
              addresses: customer._links.addresses.href,
              cards: customer._links.cards.href
            });
          });
        },
        // 3. Address and card of the customer, in parallel.
        function (item, links, callback) {
          async.parallel({
            address: function (done) {
              console.log("GET Request to: " + links.addresses);
              callBackend({uri: links.addresses, method: 'GET'}, function (err, response, body) {
                if (err) {
                  return done(err);
                }
                var data = parseJson(body);
                if (response.statusCode != 200 || !data || !data._embedded || !Array.isArray(data._embedded.address)) {
                  return done(httpError(502, "User service returned invalid addresses"));
                }
                var first = data._embedded.address[0];
                if (first == null || !first._links || !first._links.self) {
                  return done(httpError(422, "Customer has no address"));
                }
                done(null, first._links.self.href);
              });
            },
            card: function (done) {
              console.log("GET Request to: " + links.cards);
              callBackend({uri: links.cards, method: 'GET'}, function (err, response, body) {
                if (err) {
                  return done(err);
                }
                var data = parseJson(body);
                if (response.statusCode != 200 || !data || !data._embedded || !Array.isArray(data._embedded.card)) {
                  return done(httpError(502, "User service returned invalid cards"));
                }
                var first = data._embedded.card[0];
                if (first == null || !first._links || !first._links.self) {
                  return done(httpError(422, "Customer has no payment card"));
                }
                done(null, first._links.self.href);
              });
            }
          }, function (err, found) {
            if (err) {
              return callback(err);
            }
            callback(null, item, {
              "customer": links.customer,
              "address": found.address,
              "card": found.card,
              "items": endpoints.cartsUrl + "/" + encodeURIComponent(tempCartId) + "/items"
            });
          });
        },
        // 4. Temporary cart holding only the purchased product.
        function (item, order, callback) {
          tempCartUsed = true;
          addToCart(tempCartId, item, function (err) {
            callback(err, order);
          });
        },
        // 5. The order service takes the items from the temporary cart, authorises the
        //    payment, adds the shipping fee and queues the shipment.
        function (order, callback) {
          var options = {
            uri: endpoints.ordersUrl + '/orders',
            method: 'POST',
            json: true,
            body: order
          };
          console.log("Posting Order: " + JSON.stringify(order));
          callBackend(options, function (err, response, body) {
            if (err) {
              return callback(err);
            }
            console.log("Order response status: " + response.statusCode);
            if (response.statusCode == 406) {
              return callback(httpError(402, "Payment declined"));
            }
            if (response.statusCode != 201) {
              return callback(httpError(502, "Order service returned status " + response.statusCode));
            }
            callback(null, body);
          });
        }
    ], function (err, order) {
      // Whatever happened, the temporary cart must not survive the request.
      var finish = function () {
        if (err) {
          return next(err);
        }
        res.status(201).json(order);
      };
      if (tempCartUsed) {
        deleteCart(tempCartId, finish);
      } else {
        finish();
      }
    });
  });

  // Errors of the routes above: JSON body with the proper HTTP status, never a crash.
  app.use(function (err, req, res, next) {
    var status = err && err.status ? err.status : 500;
    console.log("Request failed: " + req.url + ": " + (err && err.message ? err.message : err));
    if (res.headersSent) {
      return next(err);
    }
    res.status(status).json({message: (err && err.message) ? err.message : "Internal error"});
  });

  module.exports = app;
}());
