#!/bin/bash

PREFIX=$1
VGG_WEIGHTS=${2-vgg16_weights.h5}
HEIGHT=512
PATCH_SIZE=3

echo "Only using analogy loss"
make_image_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-analogy-only/$PREFIX-Bp \
  --mrf-w=0 --height=$HEIGHT \
  --model=brute --patch-size=$PATCH_SIZE \
  --vgg-weights=$VGG_WEIGHTS --output-full

echo "Stock output (analogy and local coherence loss)"
make_image_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-blend/$PREFIX-Bp \
  --height=$HEIGHT \
  --model=brute --patch-size=$PATCH_SIZE \
  --vgg-weights=$VGG_WEIGHTS --output-full

echo "Style transfer (content loss and local coherence loss)"
make_image_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-style-xfer/$PREFIX-Bp \
  --analogy-w=0 --b-content-w=1 --mrf-w=1 --height=$HEIGHT \
  --model=brute --patch-size=$PATCH_SIZE \
  --vgg-weights=$VGG_WEIGHTS --output-full

echo "Texture generator (local coherence only)"
make_image_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-texture/$PREFIX-Bp \
  --analogy-w=0 --height=$HEIGHT \
  --model=brute --patch-size=$PATCH_SIZE \
  --vgg-weights=$VGG_WEIGHTS --output-full
