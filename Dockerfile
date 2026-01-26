# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-prod.txt .

# Install Python dependencies to /install
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements-prod.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ /app/
COPY artifacts/models/ /app/models/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


















# # ----------------------------------------------
# # Base image
# # ----------------------------------------------
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Avoid Python buffering
# ENV PYTHONUNBUFFERED=1

# # ----------------------------------------------
# # Install OS dependencies (for building some packages)
# # ----------------------------------------------
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         gcc \
#         g++ \
#         curl \
#         && rm -rf /var/lib/apt/lists/*

# # ----------------------------------------------
# # Copy requirements and install
# # ----------------------------------------------
# COPY requirements-prod.txt .
# RUN pip install --no-cache-dir --upgrade pip \
#     && pip install --no-cache-dir -r requirements-prod.txt

# # ----------------------------------------------
# # Copy app source code
# # ----------------------------------------------
# COPY ./app ./app

# # ----------------------------------------------
# # Expose port for FastAPI
# # ----------------------------------------------
# EXPOSE 8000

# # ----------------------------------------------
# # Run FastAPI with Uvicorn
# # ----------------------------------------------
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
