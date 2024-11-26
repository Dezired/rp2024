#!/bin/bash
##############################################################################
##                            Run the container                             ##
##############################################################################
SRC_CONTAINER=/home/philipp/workspace/src
SRC_HOST="$(pwd)"/src
DATA_CONTAINER=/home/philipp/data
DATA_HOST="$(pwd)"/data

docker run \
  --name rp2024-bullet \
  --privileged \
  --rm \
  -it \
  --net=host \
  -v "$SRC_HOST":"$SRC_CONTAINER":rw \
  -v "$DATA_HOST":"$DATA_CONTAINER":rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
  -e DISPLAY="$DISPLAY" -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e PULSE_SERVER=$PULSE_SERVER \
  --gpus all \
 rp2024/bullet
