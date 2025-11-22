# Docker Setup Guide

## Quick Start

### Build and Run with Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Build and Run with Docker

```bash
# Build the image
docker build -t nostradamus-api .

# Run the container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/all_sim_input_data.v2.json:/app/all_sim_input_data.v2.json:ro \
  -v $(pwd)/all_sim_input_data_with_params.v2.json:/app/all_sim_input_data_with_params.v2.json:ro \
  --name nostradamus-api \
  nostradamus-api

# View logs
docker logs -f nostradamus-api

# Stop the container
docker stop nostradamus-api

# Remove the container
docker rm nostradamus-api
```

## Testing the API

Once the container is running, test the endpoints:

### Health Check
```bash
curl http://localhost:8000/health
```

### Full Simulation
```bash
curl -X POST http://localhost:8000/api/v1/simulation/simulate \
  -H "Content-Type: application/json" \
  -d @all_sim_input_data_with_params.v2.json
```

### Histogram Only
```bash
curl -X POST http://localhost:8000/api/v1/simulation/histo_buy \
  -H "Content-Type: application/json" \
  -d @all_sim_input_data_with_params.v2.json
```

### API Documentation
Open in your browser:
- http://localhost:8000/docs
- http://localhost:8000/redoc

## Development with Docker

### Run with live code reload (mount source code)

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd):/app \
  --name nostradamus-api-dev \
  nostradamus-api \
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Execute commands inside the container

```bash
# Open a shell
docker exec -it nostradamus-api bash

# Run Python commands
docker exec nostradamus-api python -c "import pandas; print(pandas.__version__)"
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs nostradamus-api

# Check if port is already in use
lsof -i :8000

# Remove old containers
docker rm -f nostradamus-api
```

### Rebuild after code changes
```bash
# With docker-compose
docker-compose up --build

# With docker
docker build -t nostradamus-api . && docker run -d -p 8000:8000 nostradamus-api
```

### View resource usage
```bash
docker stats nostradamus-api
```

## Production Deployment

For production, consider:

1. **Use environment variables for configuration**
2. **Add health checks**
3. **Set resource limits**
4. **Use multi-stage builds to reduce image size**

Example production docker-compose.yml:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=info
    volumes:
      - ./data:/app/data:ro
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```
