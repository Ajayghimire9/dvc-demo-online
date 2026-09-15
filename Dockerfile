FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app

COPY pyproject.toml requirements-portfolio.txt ./
RUN pip install --no-cache-dir -r requirements-portfolio.txt uvicorn
COPY src ./src

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "customerflow.api:app", "--host", "0.0.0.0", "--port", "8000"]
