(function (){
  'use strict';

  var express   = require("express")
    , request   = require("request")
    , endpoints = require("../endpoints")
    , app       = express()

  function parseJson(body) {
    if (typeof body === "string") {
      return JSON.parse(body);
    }
    return body;
  }

  // REQ-17: order history (login required)
  app.get("/orders/history", function (req, res, next) {
    var custId = req.session && req.session.customerId;
    if (!custId) {
      return res.status(401).json({error: "User not logged in."});
    }

    var limit = parseInt(req.query.limit, 10);
    if (isNaN(limit) || limit < 1) {
      limit = 5;
    }
    if (limit > 20) {
      limit = 20;
    }

    var url = endpoints.ordersUrl + "/orders/search/customerId?sort=date&custId=" + encodeURIComponent(custId);
    request.get(url, function (error, response, body) {
      if (error) {
        return res.status(502).json({error: "Orders service unavailable."});
      }
      if (response.statusCode == 404) {
        return res.status(200).json({count: 0, orders: []});
      }
      if (response.statusCode != 200) {
        return res.status(502).json({error: "Orders service error."});
      }

      var list;
      try {
        var json = parseJson(body);
        list = (json && json._embedded && json._embedded.customerOrders) || [];
      } catch (e) {
        return res.status(502).json({error: "Invalid response from orders service."});
      }

      // Newest first, regardless of the service's sort direction.
      list = list.slice().sort(function (a, b) {
        return new Date(b.date).getTime() - new Date(a.date).getTime();
      }).slice(0, limit);

      var orders = list.map(function (o) {
        var itemCount = (o.items || []).reduce(function (sum, it) {
          return sum + (Number(it.quantity) || 0);
        }, 0);
        var status = null;
        if (o.shipment && o.shipment.status != null) {
          status = o.shipment.status;
        }
        var id = o.id;
        if (id == null) {
          var href = o._links && o._links.self && o._links.self.href;
          if (href) {
            id = href.split("?")[0].replace(/\/+$/, "").split("/").pop();
          }
        }
        return {
          id: id,
          date: o.date,
          total: o.total,
          itemCount: itemCount,
          status: status
        };
      });

      res.status(200).json({count: orders.length, orders: orders});
    });
  });

  // REQ-18: related products (public)
  app.get("/catalogue/related", function (req, res, next) {
    var id = req.query.id;
    if (!id || typeof id !== "string") {
      return res.status(400).json({error: "Missing id."});
    }

    request.get(endpoints.catalogueUrl + "/catalogue/" + encodeURIComponent(id), function (error, response, body) {
      if (error) {
        return res.status(502).json({error: "Catalogue service unavailable."});
      }
      if (response.statusCode == 404) {
        return res.status(404).json({error: "Product not found."});
      }
      if (response.statusCode != 200) {
        return res.status(502).json({error: "Catalogue service error."});
      }

      var product;
      try {
        product = parseJson(body);
      } catch (e) {
        return res.status(502).json({error: "Invalid response from catalogue service."});
      }
      var tags = (product && product.tag) || [];
      if (tags.length === 0) {
        return res.status(200).json({id: id, related: []});
      }

      var listUrl = endpoints.catalogueUrl + "/catalogue?size=100&tags=" +
        tags.map(encodeURIComponent).join(",");
      request.get(listUrl, function (error, response, body) {
        if (error) {
          return res.status(502).json({error: "Catalogue service unavailable."});
        }
        if (response.statusCode != 200) {
          return res.status(502).json({error: "Catalogue service error."});
        }

        var items;
        try {
          items = parseJson(body);
        } catch (e) {
          return res.status(502).json({error: "Invalid response from catalogue service."});
        }
        if (!Array.isArray(items)) {
          return res.status(502).json({error: "Invalid response from catalogue service."});
        }

        var scored = [];
        items.forEach(function (p) {
          if (p.id === product.id || p.id === id) {
            return;
          }
          var shared = (p.tag || []).filter(function (t, i, arr) {
            return tags.indexOf(t) !== -1 && arr.indexOf(t) === i;
          }).length;
          if (shared > 0) {
            scored.push({p: p, shared: shared});
          }
        });

        scored.sort(function (a, b) {
          if (b.shared !== a.shared) {
            return b.shared - a.shared;
          }
          return a.p.price - b.p.price;
        });

        var related = scored.slice(0, 4).map(function (s) {
          return {
            id: s.p.id,
            name: s.p.name,
            price: s.p.price,
            imageUrl: s.p.imageUrl,
            tag: s.p.tag
          };
        });

        res.status(200).json({id: product.id, related: related});
      });
    });
  });

  module.exports = app;
}());
