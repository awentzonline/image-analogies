#!/bin/bash
# Styles a gif by splitting out the frames and processing them individually.
# Uses `convert` tool from ImageMagick http://www.imagemagick.org/script/binary-releases.php
GIF=$1
IMAGE_A=$2
IMAGE_AP=$3
PREFIX=$4
VGG_WEIGHTS=${5-vgg16_weights.h5}
WIDTH=256
PATCH_SIZE=3  # try 1 for less interesting, but faster-rendering effects
MODEL=brute
CONTRAST=0.5
MRFW=1.5
CONTENTW=0
ANALOGYW=1
SCALES=1
ITERATIONS=2
FRAMES_PATH=$PREFIX/frames
WORK_PATH=$PREFIX/work
PROCESSED_PATH=$PREFIX/processed
DELAY=5

echo "Styling a gif."
echo "Splitting $GIF..."
mkdir -p $FRAMES_PATH
mkdir -p $WORK_PATH
mkdir -p $PROCESSED_PATH
convert -alpha Remove -coalesce $GIF $FRAMES_PATH/%04d.png

echo "Optimizing frames..."
for frame in $FRAMES_PATH/*.png
do
  echo "processing $frame"
  make_image_analogy.py \
    $IMAGE_A $IMAGE_AP \
    $frame \
    $WORK_PATH/out  \
    --width=$WIDTH \
    --mrf-w=$MRFW \
    --a-scale-mode=match \
    --b-content-w=$CONTENTW \
    --analogy-w=$ANALOGYW \
    --scales=$SCALES --min-scale=0.5 --iters=$ITERATIONS \
    --model=$MODEL --patch-size=$PATCH_SIZE \
    --contrast=$CONTRAST \
    --vgg-weights=$VGG_WEIGHTS --output-full
  LAST_FILE=`ls -1 $WORK_PATH | tail -n 1`
  cp $WORK_PATH/$LAST_FILE $PROCESSED_PATH/$(basename $frame)
done

echo "Combining new frames..."
convert -delay $DELAY -loop 0 $PROCESSED_PATH/*.png $PREFIX/result.gif
