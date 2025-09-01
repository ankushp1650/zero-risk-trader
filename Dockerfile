# Use a more complete Python 3.11 image
FROM python:3.11-bullseye

# Install system dependencies using MariaDB packages instead
RUN apt-get update -y && apt-get install -y \
    libmariadb-dev-compat \
    libmariadb-dev \
    build-essential \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .
COPY .env .env

# TensorFlow performance/logging settings
ENV TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    KMP_AFFINITY=disabled

# Expose port 8000 (Django default port)
EXPOSE 8000

# Command to run your application
CMD ["gunicorn", "auth_project.wsgi:application", "--workers=3", "--threads=2", "--timeout=120", "--bind", "0.0.0.0:8000"]
