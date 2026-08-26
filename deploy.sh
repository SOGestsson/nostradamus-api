#!/bin/bash
set -e

IMAGE="sogestsson/nostradamus-api:latest"
PI_HOST="${DEPLOY_HOST:-100.108.73.62}"
PI_USER="siggi"
CONTAINER="nostradamus-api"
REDIS_CONTAINER="redis"
NETWORK="nostradamus-net"

# The API refuses every /api/v1 request when API_KEY is unset, so fail here
# rather than deploying a container that returns 503 to every caller.
: "${API_KEY:?API_KEY must be set. Same value as SIM_API_KEY on the pipeline gateway and the frontend.}"
: "${SANDBOX_DB_PASSWORD:?SANDBOX_DB_PASSWORD must be set}"

if ! ssh -o BatchMode=yes -o ConnectTimeout=3 "$PI_USER@$PI_HOST" true 2>/dev/null; then
  echo "==> Setting up SSH key on $PI_HOST (one-time)..."
  ssh-copy-id "$PI_USER@$PI_HOST"
fi

echo "==> Building for linux/arm64..."
docker buildx build --platform linux/arm64 -t "$IMAGE" --push .

echo "==> Deploying on $PI_HOST..."
ssh "$PI_USER@$PI_HOST" "
  docker network inspect $NETWORK >/dev/null 2>&1 || docker network create $NETWORK

  if ! docker ps --format '{{.Names}}' | grep -qx $REDIS_CONTAINER; then
    docker rm -f $REDIS_CONTAINER 2>/dev/null || true
    docker run -d --name $REDIS_CONTAINER --restart unless-stopped \
      --network $NETWORK \
      -p 6379:6379 \
      redis:7-alpine
  fi

  docker pull $IMAGE &&
  docker stop $CONTAINER 2>/dev/null || true &&
  docker rm $CONTAINER 2>/dev/null || true &&
  docker run -d -p 8000:8000 --name $CONTAINER --restart unless-stopped \
    --network $NETWORK \
    -e REDIS_URL=redis://${REDIS_CONTAINER}:6379/0 \
    -e API_KEY='$API_KEY' \
    -e SANDBOX_DB_HOST=192.168.1.50 \
    -e SANDBOX_DB_PORT=4406 \
    -e SANDBOX_DB_USER='${SANDBOX_DB_USER:-root}' \
    -e SANDBOX_DB_PASSWORD='$SANDBOX_DB_PASSWORD' \
    $IMAGE &&

  ok=0 &&
  for i in \$(seq 1 20); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null; then
      echo \"nostradamus-api ok (attempt \$i)\"
      ok=1
      break
    fi
    sleep 2
  done &&
  if [ \"\$ok\" -ne 1 ]; then
    echo 'nostradamus-api health check failed' >&2
    docker logs --tail 50 $CONTAINER >&2 || true
    exit 1
  fi
"

echo "==> Done. Nostradamus API running on $PI_HOST:8000"
