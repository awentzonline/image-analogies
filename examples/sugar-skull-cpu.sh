#!/bin/bash

IMAGE_B=$1
PREFIX=$2
VGG_WEIGHTS=${3-vgg16_weights.h5}
HEIGHT=512
PATCH_SIZE=3  # try 3 for more interesting, but slow-rendering effects
SKULL_IMAGE_A=images/sugarskull-A.jpg
SKULL_IMAGE_AP=images/sugarskull-Ap.jpg

echo "Make a sugar skull (CPU)"
make_image_analogy.py \
  $SKULL_IMAGE_A $SKULL_IMAGE_AP \
  $IMAGE_B \
  out/$PREFIX-sugarskull-cpu/$PREFIX-Bp  \
  --height=$HEIGHT \
  --mrf-w=1.5 \
  --a-scale-mode=match \
  --model=patchmatch --patch-size=$PATCH_SIZE \
  --contrast=1 \
  --vgg-weights=$VGG_WEIGHTS --output-full
