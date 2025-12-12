# Use an official, slim Python 3.11 image (Debian bullseye)
FROM python:3.11-slim-bullseye

# metadata
LABEL maintainer="Ankush Patil"

# prevent Python from writing .pyc files and ensure stdout/stderr are not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# set workdir
WORKDIR /app

# install system deps + add CA cert in one layer (small & efficient)
# install system deps + CA cert + pkg-config + mysql client dev headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    python3-dev \
    default-libmysqlclient-dev \
    libmariadb-dev-compat \
    libmariadb-dev \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    curl \
 && mkdir -p /usr/local/share/ca-certificates/extra \
 && rm -rf /var/lib/apt/lists/*


# copy the certificate from repo and add to system trust store
# make sure you have certificate/DigiCertGlobalRootCA.crt.pem in your repo
COPY certificate/DigiCertGlobalRootCA.crt.pem /usr/local/share/ca-certificates/extra/DigiCertGlobalRootCA.crt.pem
RUN update-ca-certificates

# copy only requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# install python dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy project files (but DO NOT copy .env into image)
COPY . /app

# create a non-root user and give ownership of app dir
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# expose port and set CMD (gunicorn)
EXPOSE 8000

CMD ["gunicorn", "auth_project.wsgi:application", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8000"]
