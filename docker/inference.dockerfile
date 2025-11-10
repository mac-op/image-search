ARG PROJECT_ID
FROM us-central1-docker.pkg.dev/${PROJECT_ID}/inference-service/base:latest

WORKDIR /app
COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
