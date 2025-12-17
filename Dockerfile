FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY Pipfile Pipfile.lock ./
COPY requirements.txt ./

# Install dependencies from requirements.txt (fallback to avoid pipenv lock issues)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
