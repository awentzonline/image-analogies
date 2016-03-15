#!/bin/bash

PREFIX=$1
VGG_WEIGHTS=${2-vgg16_weights.h5}
MRF_VALS=(0.0 0.5 1.0 1.5)
HEIGHT=${3-320}

for ((i=0; i < ${#MRF_VALS[@]}; i++)); do
  MRF_VAL=${MRF_VALS[i]}
  make_image_analogy.py \
    images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
    images/$PREFIX-B.jpg out/$PREFIX-mrf-${MRF_VAL}-cpu/$PREFIX-Bp \
    --mrf-w=${MRF_VAL} --patch-size=3 --height=$HEIGHT \
    --model=patchmatch \
    --vgg-weights=$VGG_WEIGHTS --output-full
done
