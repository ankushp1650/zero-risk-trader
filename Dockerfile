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

# Expose port 8000 (Django default port)
EXPOSE 8000

# Command to run your application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
