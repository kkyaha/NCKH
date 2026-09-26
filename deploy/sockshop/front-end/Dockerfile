# front-end goc (weaveworksdemos/front-end:0.3.12) + BA module them route:
#   api/features        : 4 tinh nang (promo, recs, track, review) do nguoi xay dung cai
#   api/features_indep  : 3 tinh nang (cart/summary, cart/quick, checkout/express) do MOT AGENT DOC LAP cai
#   api/features_prosp2 : 6 tinh nang (login, register, wishlist, catalogue/search, account/overview, checkout/preview; REQ-11..16) do MOT AGENT MU cai -- vong TIEN CUU 2
#   api/features_prosp3 : 2 tinh nang (orders/history, catalogue/related; REQ-17,18) do MOT AGENT MU KHAC cai -- bo sung n=8 cua vong TIEN CUU 2
#   api/features_prosp4 : 2 tinh nang NANG (orders/detailed, orders/reorder; REQ-19,20) do MOT AGENT MU KHAC cai
#                         -- mo rong DAI diem gay de kiem tra mo hinh co phan biet duoc giua cac tinh nang khong (muc 5n)
#   api/features_prosp  : 1 tinh nang (catalogue/browse, REQ-07) do MOT AGENT DOC LAP KHAC cai -- kiem dinh TIEN CUU cho P3
# Chi THEM route, khong doi hanh vi route co san -> baseline chay tren cung image nay
# la tuong duong image goc (route moi khong duoc goi thi khong ton tai chi phi).
#
# QUAN TRONG: features_prosp PHAI gan TRUOC `catalogue` -- api/catalogue/index.js co route dai dien
# `app.get("/catalogue*", ...)` proxy moi /catalogue/... sang catalogue service; neu gan sau, route
# /catalogue/browse se bi no nuot mat truoc khi toi duoc features_prosp (agent doc lap da tu canh bao
# dung diem nay trong DESIGN.md). features/features_indep khong dung tien to /catalogue nen khong bi anh huong.
FROM weaveworksdemos/front-end:0.3.12
USER root
COPY features /usr/src/app/api/features
COPY features_indep /usr/src/app/api/features_indep
COPY features_prosp /usr/src/app/api/features_prosp
COPY features_prosp2 /usr/src/app/api/features_prosp2
COPY features_prosp3 /usr/src/app/api/features_prosp3
COPY features_prosp4 /usr/src/app/api/features_prosp4
RUN sed -i 's#^app.use(cart);#app.use(require("./api/features_prosp4"));\napp.use(require("./api/features_prosp3"));\napp.use(require("./api/features_prosp2"));\napp.use(require("./api/features_prosp"));\napp.use(cart);#' /usr/src/app/server.js \
 && sed -i 's#^app.use(user);#app.use(user);\napp.use(require("./api/features"));\napp.use(require("./api/features_indep"));#' /usr/src/app/server.js \
 && grep -n 'api/features\|require("./api/features_prosp")' /usr/src/app/server.js \
 && chmod -R a+rX /usr/src/app/api/features /usr/src/app/api/features_indep /usr/src/app/api/features_prosp /usr/src/app/api/features_prosp2 /usr/src/app/api/features_prosp3 /usr/src/app/api/features_prosp4
USER myuser
