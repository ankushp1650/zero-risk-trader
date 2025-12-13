# Use official Python image
FROM python:3.11-slim-bullseye

LABEL maintainer="Ankush Patil"

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install only required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    curl \
 && mkdir -p /usr/local/share/ca-certificates/extra \
 && rm -rf /var/lib/apt/lists/*

# Copy Azure MySQL CA certificate
COPY certificate/DigiCertGlobalRootCA.crt.pem \
     /usr/local/share/ca-certificates/extra/DigiCertGlobalRootCA.crt.pem

# Update CA trust store
RUN update-ca-certificates

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose application port
EXPOSE 8000

# Run Django with Gunicorn
CMD ["gunicorn", "auth_project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2"]
