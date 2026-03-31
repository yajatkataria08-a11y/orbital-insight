FROM nginx:1.27-alpine

# Copy the frontend static files
COPY frontend/ /usr/share/nginx/html/

# Generate a clean nginx config with dynamic backend resolution
RUN printf '\
server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
\n\
    # Important: Enable resolver for Railway internal DNS\n\
    resolver 127.0.0.11 valid=30s ipv6=off;\n\
\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
\n\
    # Proxy API calls to backend service\n\
    location /api/ {\n\
        # Use variable so Nginx re-resolves the hostname on every request\n\
        set $backend http://backend.railway.internal:8000;\n\
        proxy_pass $backend/api/;\n\
\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
        proxy_set_header X-Forwarded-Proto $scheme;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

# Environment variable (for reference)
ENV BACKEND_URL=http://backend.railway.internal:8000

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
