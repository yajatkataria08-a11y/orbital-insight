FROM nginx:1.27-alpine
COPY frontend/ /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Replace backend URL with Railway internal reference
ENV BACKEND_URL=http://backend.railway.internal:8000
CMD ["/bin/sh", "-c", "sed -i 's|http://backend:8000|${BACKEND_URL}|g' /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
