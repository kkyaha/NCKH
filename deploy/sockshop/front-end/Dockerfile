# front-end goc (weaveworksdemos/front-end:0.3.12) + hai module them route:
#   api/features        : 4 tinh nang (promo, recs, track, review) do nguoi xay dung cai
#   api/features_indep  : 3 tinh nang (cart/summary, cart/quick, checkout/express) do MOT AGENT DOC LAP cai
# Chi THEM route, khong doi hanh vi route co san -> baseline chay tren cung image nay
# la tuong duong image goc (route moi khong duoc goi thi khong ton tai chi phi).
FROM weaveworksdemos/front-end:0.3.12
USER root
COPY features /usr/src/app/api/features
COPY features_indep /usr/src/app/api/features_indep
RUN sed -i 's#^app.use(user);#app.use(user);\napp.use(require("./api/features"));\napp.use(require("./api/features_indep"));#' /usr/src/app/server.js \
 && grep -n 'api/features' /usr/src/app/server.js \
 && chmod -R a+rX /usr/src/app/api/features /usr/src/app/api/features_indep
USER myuser
