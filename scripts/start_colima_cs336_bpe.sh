#!/usr/bin/env bash
set -euo pipefail

# Colima runtime profile for CS336 Assignment 1 OWT BPE training.
# Defaults are tuned for this host: arm64, 10 CPU cores, 16 GiB RAM.

PROFILE="${COLIMA_PROFILE:-cs336-bpe}"
CPUS="${COLIMA_CPUS:-4}"
MEMORY_GB="${COLIMA_MEMORY_GB:-12}"
DISK_GB="${COLIMA_DISK_GB:-30}"
WORKSPACE="${CS336_WORKSPACE:-/Users/daniel/projs/cs336homework/assignment1-basics}"

if ! command -v colima >/dev/null 2>&1; then
  echo "colima is not installed. Install it first with: brew install colima"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker CLI is not installed. Install it first with: brew install docker"
  exit 1
fi

if [ ! -d "$WORKSPACE" ]; then
  echo "workspace does not exist: $WORKSPACE"
  exit 1
fi

echo "Starting Colima profile: $PROFILE"
echo "  cpus:      $CPUS"
echo "  memory:    ${MEMORY_GB}GiB"
echo "  disk:      ${DISK_GB}GiB"
echo "  workspace: $WORKSPACE"

colima start "$PROFILE" \
  --runtime docker \
  --arch aarch64 \
  --vm-type vz \
  --mount-type virtiofs \
  --cpus "$CPUS" \
  --memory "$MEMORY_GB" \
  --disk "$DISK_GB" \
  --mount "$WORKSPACE:w" \
  --activate

echo
echo "Active Docker context:"
docker context show

echo
echo "Docker VM resources:"
docker info --format 'OS={{.OperatingSystem}} CPUs={{.NCPU}} MemBytes={{.MemTotal}}'

cat <<'EOF'

Run OWT BPE inside this capped Colima profile:

docker run --rm -it --name cs336-bpe-owt \
  --cpus=3 \
  --memory=9g \
  --memory-swap=9g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e UV_CACHE_DIR=/work/.uv-cache \
  -e UV_PROJECT_ENVIRONMENT=/work/.docker-venv \
  -e BPE_NUM_PROCESSES=3 \
  -e BPE_NUM_CHUNKS=32 \
  -v /Users/daniel/projs/cs336homework/assignment1-basics:/work \
  -w /work \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -lc 'uv run python -u cs336_basics/train_bpe.py > logs/owt-bpe-$(date +%Y%m%d-%H%M%S).log 2>&1'

Useful controls:
  colima status cs336-bpe
  colima stop cs336-bpe
  docker logs -f cs336-bpe-owt
EOF
