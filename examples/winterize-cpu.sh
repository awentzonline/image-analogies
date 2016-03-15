#!/bin/bash

TARGET=$1
OUTPUT_PREFIX=$2
VGG_WEIGHTS=${3-vgg16_weights.h5}
HEIGHT=${4-512}
PATCH_SIZE=1  # try 3 for better-looking but slower-rendering results

make_image_analogy.py \
  images/season-xfer-A.jpg images/season-xfer-Ap.jpg \
  $TARGET out/$OUTPUT_PREFIX-winterized-cpu/$OUTPUT_PREFIX-Bp \
  --analogy-layers='conv1_1,conv2_1,conv3_1,conv4_1' \
  --scales=5 --contrast=0.1 \
  --model=patchmatch --patch-size=$PATCH_SIZE \
  --height=$HEIGHT \
  --vgg-weights=$VGG_WEIGHTS --output-full
