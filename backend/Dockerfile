# 1. Base Image
# Using python:3.10-slim based on your runtime.txt
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app



# 3. Install Python Dependencies
# Copy requirements file first (it's in the same dir as the Dockerfile)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# expose Port
EXPOSE 8000

#  Run Application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]