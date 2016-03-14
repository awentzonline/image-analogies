#!/bin/bash

PREFIX=$1
VGG_WEIGHTS=${2-vgg16_weights.h5}
HEIGHT=512

make_nnf_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-analogy-only/$PREFIX-Bp \
  --mrf-w=0 --height=$HEIGHT \
  --vgg-weights=$VGG_WEIGHTS --output-full

make_nnf_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-blend/$PREFIX-Bp \
  --mrf-w=1.0 --height=$HEIGHT \
  --vgg-weights=$VGG_WEIGHTS --output-full

make_nnf_analogy.py \
  images/$PREFIX-A.jpg images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-mrf-only/$PREFIX-Bp \
  --analogy-w=0 --height=$HEIGHT \
  --vgg-weights=$VGG_WEIGHTS --output-full
