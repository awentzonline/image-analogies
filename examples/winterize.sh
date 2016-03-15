#!/bin/bash

TARGET=$1
OUTPUT_PREFIX=$2
VGG_WEIGHTS=${3-vgg16_weights.h5}
WIDTH=${4-512}

make_image_analogy.py \
  images/season-xfer-A.jpg images/season-xfer-Ap.jpg \
  $TARGET out/$OUTPUT_PREFIX-winterized/$OUTPUT_PREFIX-Bp \
  --analogy-layers='conv3_1,conv4_1' \
  --scales=5 --contrast=0.1 \
  --mode=brute --patch-size=3 \
  --width=$WIDTH \
  --vgg-weights=$VGG_WEIGHTS --output-full
