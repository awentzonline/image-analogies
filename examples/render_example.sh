#!/bin/bash

PREFIX=$1
VGG_WEIGHTS=${2-vgg16_weights.h5}
HEIGHT=320

make_image_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-analogy-only/$PREFIX-Bp \
  --mrf-w=0 --height=$HEIGHT \
  --vgg-weights=$VGG_WEIGHTS --output-full

make_image_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-blend/$PREFIX-Bp \
  --mrf-w=1 --height=$HEIGHT \
  --vgg-weights=$VGG_WEIGHTS --output-full

make_image_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-style-xfer/$PREFIX-Bp \
  --analogy-w=0 --b-content-w=1 --mrf-w=1 --height=$HEIGHT \
  --vgg-weights=$VGG_WEIGHTS --output-full

make_image_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-texture/$PREFIX-Bp \
  --analogy-w=0 --height=$HEIGHT \
  --vgg-weights=$VGG_WEIGHTS --output-full
