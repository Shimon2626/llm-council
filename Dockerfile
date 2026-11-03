# Two runtime images from one file: `api` (FastAPI backend) and `web` (nginx
# serving the built UI and proxying /api/ to the backend).

FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim AS api
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/ backend/
ENV PATH="/app/.venv/bin:$PATH" COUNCIL_HOST=0.0.0.0 COUNCIL_PORT=8001 COUNCIL_RELOAD=0
EXPOSE 8001
CMD ["python", "-m", "backend.main"]

FROM nginx:1.27-alpine AS web
COPY deploy/nginx-container.conf /etc/nginx/conf.d/default.conf
COPY --from=ui /ui/dist /usr/share/nginx/html
