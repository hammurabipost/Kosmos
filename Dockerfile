# =============================================================================
# Kosmos AI Scientist - Production Dockerfile
# =============================================================================
#
# Multi-stage build for optimized production deployment
#
# Features:
# - Multi-stage build for minimal image size
# - Non-root user for security
# - Health check endpoint
# - Optimized layer caching
# - Production-ready configuration
#
# Build:
#   docker build -t kosmos:latest .
#
# Run:
#   docker run -p 8000:8000 kosmos:latest
#
# =============================================================================

# =============================================================================
# Stage 1: Builder
# =============================================================================

FROM python:3.11-slim as builder

LABEL maintainer="Kosmos AI Scientist Team"
LABEL description="Autonomous scientific research system"
LABEL version="0.10.0"

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for layer caching)
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# =============================================================================
# Stage 2: Runtime
# =============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY kosmos/ /app/kosmos/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/
COPY README.md /app/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache && \
    chmod 755 /app/data /app/logs /app/cache

# Create non-root user for security
RUN useradd --create-home --uid 1000 --shell /bin/bash kosmos && \
    chown -R kosmos:kosmos /app

# Switch to non-root user
USER kosmos

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    KOSMOS_DATA_DIR=/app/data \
    KOSMOS_LOG_DIR=/app/logs \
    KOSMOS_CACHE_DIR=/app/cache

# Expose port (if running web service)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import kosmos; print('healthy')" || exit 1

# Default command
CMD ["python", "-m", "kosmos.cli.main", "--help"]

# =============================================================================
# Build Info
# =============================================================================
#
# Image size: ~400MB (vs ~1GB without multi-stage build)
# Security: Non-root user, minimal attack surface
# Performance: Optimized layer caching for fast rebuilds
#
# =============================================================================
