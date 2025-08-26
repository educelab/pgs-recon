#! /bin/bash

set -e

echo "!!!WARNING!!!"
echo "This script will push new tags to the GitHub Container Registry. If "
echo "you're *really* sure you want to do that, edit this file to update the "
echo "REF and VER_* variables and to comment out this message."
echo "!!!WARNING!!!"
exit

REPO=https://github.com/educelab/pgs-recon
GHCR=ghcr.io/educelab/pgs-recon
REF=v1.7.1
VER_MAJOR=1
VER_MINOR=7
VER_PATCH=1
VER_EXTRA=
VER_FULL=${VER_MAJOR}.${VER_MINOR}.${VER_PATCH}${VER_EXTRA}
VER_SHORT=${VER_MAJOR}.${VER_MINOR}
REV=$(git rev-list -n 1 ${REF})

CFGS=(\
#  "ubuntu:22.04 OFF " \ # GitHub Actions handles CPU-only builds
  "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 ON -cuda12.4" \
  "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 ON -cuda12.8" \
)

timestamp() {
  date -u "+%FT%TZ"
}

labels() {
    SUFFIX=$1
    echo --label org.opencontainers.image.created=$(timestamp) \
    --label org.opencontainers.image.licenses=AGPL-3.0 \
    --label org.opencontainers.image.revision=${REV} \
    --label org.opencontainers.image.url=${REPO} \
    --label org.opencontainers.image.source=${REPO} \
    --label org.opencontainers.image.version=${VER_FULL}${SUFFIX}
}

tags() {
  SUFFIX=$1

  # if we're on main, this is an edge tag only
  if [[ $REF == "main" ]]; then
    echo "--tag ${GHCR}:edge${SUFFIX}"
    return
  fi

  # not main, so a versioned release
  TAGS="--tag ${GHCR}:${VER_FULL}${SUFFIX}"

  # not an alpha/beta/etc. release, so update the minor 
  # version and latest tags
  if [[ -z "$VER_EXTRA" ]]; then
    TAGS="${TAGS} \
          --tag ${GHCR}:${VER_SHORT}${SUFFIX} \
          --tag ${GHCR}:latest${SUFFIX}"
  fi
  echo "${TAGS}"
}

# do the builds
for cfg in "${CFGS[@]}"; do
  cfg=($cfg)
  BASE_IMAGE=${cfg[0]}
  USE_CUDA=${cfg[1]}
  SUFFIX=${cfg[2]}
  echo ===========================================
  echo Building "${VER_FULL}${SUFFIX} (ref: ${REF})"
  echo ===========================================
  docker buildx build \
    --platform=linux/amd64,linux/arm64 \
      $(labels ${SUFFIX}) \
      $(tags ${SUFFIX}) \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg USE_CUDA=${USE_CUDA} \
    --push \
    https://github.com/educelab/pgs-recon.git#${REV}
    echo
done