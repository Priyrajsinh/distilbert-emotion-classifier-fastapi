# Stage 1 — builder: install all Python dependencies
FROM python:3.12-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt


# Stage 2 — runtime: lean image with only what the app needs
FROM python:3.12-slim AS runtime

# Copy installed packages from builder
COPY --from=builder /deps /usr/local

WORKDIR /app

# Copy application code and assets
COPY src/ src/
COPY models/ models/
COPY config/ config/
COPY reports/ reports/

# Non-root user for security
RUN adduser --disabled-password appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
