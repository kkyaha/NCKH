(function (){
  'use strict';

  var express   = require("express")
    , request   = require("request")
    , endpoints = require("../endpoints")
    , helpers   = require("../../helpers")
    , app       = express()

  /* This module backs REQ-07: browsing the "sport" socks section of the
   * catalogue, with optional filtering by color and by "size".
   *
   * The catalogue service has no dedicated `size` field (see API_SURFACE.md);
   * the only categorical attribute a product carries is its `tag` array
   * (e.g. ["blue","sport","action"]). So both `color` and `size` query
   * values are matched the same way, against that same `tag` array. This is
   * the most defensible reading available without a real `size` field to
   * anchor on -- see DESIGN.md for the reasoning and the risk this carries.
   */

  var CATEGORY_TAG = "sport";

  /* parseCsvParam("blue, Red") -> ["blue","red"]
   * Accepts either a single "a,b,c" query value or Express's array form
   * (repeated query keys, e.g. ?color=blue&color=red). Missing/blank input
   * yields an empty array, which callers treat as "no filter on this facet".
   */
  function parseCsvParam(raw) {
    var tokens = [];
    if (!raw) return tokens;

    var values = Array.isArray(raw) ? raw : [raw];
    values.forEach(function (value) {
      String(value).split(",").forEach(function (token) {
        var trimmed = token.trim().toLowerCase();
        if (trimmed.length > 0) {
          tokens.push(trimmed);
        }
      });
    });

    return tokens;
  }

  /* True if `productTags` contains at least one of the values in `wanted`
   * (case-insensitive). An empty `wanted` list means "filter not in use",
   * so every product passes it.
   */
  function matchesAny(productTags, wanted) {
    if (wanted.length === 0) return true;
    if (!Array.isArray(productTags)) return false;

    var lowerTags = productTags.map(function (tag) {
      return String(tag).toLowerCase();
    });

    return wanted.some(function (want) {
      return lowerTags.indexOf(want) !== -1;
    });
  }

  app.get("/catalogue/browse", function (req, res, next) {
    console.log("GET /catalogue/browse - query: %j", req.query);

    var colors = parseCsvParam(req.query.color);
    var sizes  = parseCsvParam(req.query.size);

    var page = parseInt(req.query.page, 10);
    if (isNaN(page) || page < 1) {
      page = 1;
    }

    var pageSize = parseInt(req.query.pageSize, 10);
    if (isNaN(pageSize) || pageSize < 1) {
      pageSize = null; // no pagination requested -> return the full filtered list
    }

    // The catalogue service's `tags` filter is an OR-match across all the
    // tags you give it, so we can only safely ask it to narrow down to the
    // "sport" category here. Combining "sport,blue,red" in one call would
    // also return non-sport blue/red products, which is not what a color
    // filter on the sport-socks page should do. So color/size narrowing
    // happens locally, below, once we have the "sport" set back.
    var url = endpoints.catalogueUrl + "/catalogue?tags=" + encodeURIComponent(CATEGORY_TAG);

    request.get(url, function (error, response, body) {
      if (error) {
        return next(error);
      }

      if (!response || response.statusCode !== 200) {
        var statusErr = new Error(
          "catalogue service responded with status " + (response && response.statusCode)
        );
        statusErr.status = (response && response.statusCode) || 502;
        return next(statusErr);
      }

      var products;
      try {
        products = JSON.parse(body);
      } catch (parseErr) {
        return next(parseErr);
      }

      if (!Array.isArray(products)) {
        return next(new Error("Unexpected response from catalogue service"));
      }

      var filtered = products.filter(function (product) {
        return matchesAny(product.tag, colors) && matchesAny(product.tag, sizes);
      });

      var results = filtered;
      if (pageSize) {
        var start = (page - 1) * pageSize;
        results = filtered.slice(start, start + pageSize);
      }

      helpers.respondSuccessBody(res, JSON.stringify(results));
    });
  });

  module.exports = app;
}());
