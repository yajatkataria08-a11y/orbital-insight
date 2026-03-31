FROM nginx:1.27-alpine

# Copy the static frontend files
COPY frontend/ /usr/share/nginx/html/

# Copy nginx config from root (where it actually exists)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Set default backend URL (Railway will override this)
ENV BACKEND_URL=http://backend.railway.internal:8000

# Replace the placeholder in nginx.conf with the actual backend URL at runtime
CMD ["/bin/sh", "-c", "sed -i 's|http://backend:8000|${BACKEND_URL}|g' /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
