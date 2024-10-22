#!/bin/bash
##############################################################################
##                            Run the container                             ##
##############################################################################
SRC_CONTAINER=/home/philipp/workspace/src
SRC_HOST="$(pwd)"/src
ASSETS_CONTAINER=/home/philipp/workspace/assets
ASSETS_HOST="$(pwd)"/assets

docker run \
  --name ur10e-cell-bullet \
  --privileged \
  --rm \
  -it \
  --net=host \
  -v "$SRC_HOST":"$SRC_CONTAINER":rw \
  -v "$ASSETS_HOST":"$ASSETS_CONTAINER":rw \
  -e DISPLAY="$DISPLAY" \
  --gpus all \
 rp2024/bullet
