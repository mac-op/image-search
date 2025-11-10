ARG PROJECT_ID
FROM gcr.io/${PROJECT_ID}/base:latest

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]