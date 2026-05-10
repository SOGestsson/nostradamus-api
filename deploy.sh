#!/bin/bash
set -e

IMAGE="sogestsson/nostradamus-api:latest"
PI_HOST="100.108.73.62"
PI_USER="siggi"
CONTAINER="nostradamus-api"

if ! ssh -o BatchMode=yes -o ConnectTimeout=3 "$PI_USER@$PI_HOST" true 2>/dev/null; then
  echo "==> Setting up SSH key on $PI_HOST (one-time)..."
  ssh-copy-id "$PI_USER@$PI_HOST"
fi

echo "==> Building for linux/arm64..."
docker buildx build --platform linux/arm64 -t "$IMAGE" --push .

echo "==> Deploying on $PI_HOST..."
ssh "$PI_USER@$PI_HOST" "
  docker pull $IMAGE &&
  docker stop $CONTAINER 2>/dev/null || true &&
  docker rm $CONTAINER 2>/dev/null || true &&
  docker run -d -p 8000:8000 --name $CONTAINER \
    --network nostradamus-net \
    -e REDIS_URL=redis://redis:6379/0 \
    -e SANDBOX_DB_HOST=192.168.1.50 \
    -e SANDBOX_DB_PORT=4406 \
    -e SANDBOX_DB_USER=root \
    -e SANDBOX_DB_PASSWORD=Superman \
    $IMAGE
"

echo "==> Done. Nostradamus API running on $PI_HOST:8000"
