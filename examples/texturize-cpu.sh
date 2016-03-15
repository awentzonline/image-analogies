#!/bin/bash

TARGET=$1
OUTPUT_PREFIX=$2
VGG_WEIGHTS=${3-vgg16_weights.h5}
WIDTH=${4-512}
PATCH_SIZE=${5-3} # try 1 for faster but less interesting patterns

echo "Making a texture (local coherence loss only)"
make_image_analogy.py \
  $TARGET $TARGET $TARGET\
  out/$OUTPUT_PREFIX-texturized-cpu/$OUTPUT_PREFIX-Bp \
  --analogy-layers='conv3_1,conv4_1' \
  --scales=3 --analogy-w=0 \
  --mode=patchmatch --patch-size=$PATCH_SIZE \
  --width=$WIDTH \
  --vgg-weights=$VGG_WEIGHTS --output-full
