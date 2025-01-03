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
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
  -e DISPLAY="$DISPLAY" -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e PULSE_SERVER=$PULSE_SERVER \
  --gpus all \
 rp2024/bullet
