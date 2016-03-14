#!/bin/bash

GIFSRC=$1
IMAGEA=$2
IMAGEAP=$3
PREFIX=$4
VGG_WEIGHTS=${5-vgg16_weights.h5}
TMPDIR=/tmp/img-analogy
# chop up gif into frames
mkdir $TMPDIR
convert --coalesce $GIFSRC ./
make_image_analogy.py \
  $IMAGEA $IMAGEAP images/$PREFIX-Ap.jpg \
  images/$PREFIX-B.jpg out/$PREFIX-fast/$PREFIX-Bp \
  --mrf-w=0.5 --patch-size=1 --height=$HEIGHT \
  --vgg-weights=$VGG_WEIGHTS --output-full
