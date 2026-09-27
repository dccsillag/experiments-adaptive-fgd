#!/usr/bin/env bash

set -e

CUDA_TOOLKIT_BINARY="cuda_12.6.0_560.28.03_linux.run"
INSTALL_DIR="$( realpath "$( echo "$CUDA_TOOLKIT_BINARY" | awk -F. '{print $1}' )" )"


# install CUDA
if [ ! -f "$CUDA_TOOLKIT_BINARY" ] ; then
    wget "https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/$CUDA_TOOLKIT_BINARY"
else
    echo "$CUDA_TOOLKIT_BINARY found, skipping download..."
fi

chmod +x "$CUDA_TOOLKIT_BINARY"

if [ ! -d "$INSTALL_DIR/bin" ] ; then
    "./$CUDA_TOOLKIT_BINARY" --silent --toolkit --installpath="$INSTALL_DIR" --override
else
    echo "$INSTALL_DIR found, skipping install..."
fi
