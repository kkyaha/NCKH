# front-end goc (weaveworksdemos/front-end:0.3.12) + module api/features (4 tinh nang moi).
# Chi THEM route, khong doi hanh vi route co san -> baseline chay tren cung image nay
# la tuong duong image goc (route moi khong duoc goi thi khong ton tai chi phi).
FROM weaveworksdemos/front-end:0.3.12
USER root
COPY features /usr/src/app/api/features
RUN sed -i 's#^app.use(user);#app.use(user);\napp.use(require("./api/features"));#' /usr/src/app/server.js \
 && grep -n 'api/features' /usr/src/app/server.js \
 && chmod -R a+rX /usr/src/app/api/features
USER myuser
