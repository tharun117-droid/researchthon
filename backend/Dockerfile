# 1. Base Image
# Using python:3.10-slim based on your runtime.txt
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# 2. Install System Dependencies
# Install system libraries required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Python Dependencies
# Copy requirements file first (it's in the same dir as the Dockerfile)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Application Code
# Copy everything from the current directory (the backend dir)
# into the container's /app directory
COPY . .

# expose Port
EXPOSE 8000

#  Run Application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]