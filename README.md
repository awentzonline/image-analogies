neural image analogies
----------------------
![Image of arch](https://raw.githubusercontent.com/awentzonline/image-analogies/master/images/image-analogy-explanation.jpg)
![Image of Trump](https://raw.githubusercontent.com/awentzonline/image-analogies/master/images/trump-image-analogy.jpg)
![Image of Sugar Steve](https://raw.githubusercontent.com/awentzonline/image-analogies/master/images/sugarskull-analogy.jpg)

This is basically an implementation of this "Image Analogies" paper http://www.mrl.nyu.edu/projects/image-analogies/index.html In our case, we use feature maps from VGG16. The patch matching and blending is done with a method described in "Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis" http://arxiv.org/abs/1601.04589  Effects similar to that paper can be achieved by
turning off the analogy loss (or leave it on!) and turning on the B/B' content weighting
via the `--b-content-w` parameter.

The initial code was adapted from the Keras "neural style transfer" example.

The example arch images are from the "Image Analogies" website at http://www.mrl.nyu.edu/projects/image-analogies/tbn.html
They have some other good examples from their own implementation which
are worth a look. Their paper discusses the various applications of image
analogies so you might want to take a look for inspiration.

Installation
------------
You'll want to run this on a GPU. [Here are the docs](http://deeplearning.net/software/theano/tutorial/using_gpu.html) for getting
up and running.

[Install latest keras and theano](http://keras.io/#installation) (requires theano, no tensorflow atm).

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in this script matches the location of the file or use the `--vgg-weights` parameter

Example script usage:
`python image_analogy.py path_to_A path_to_A_prime path_to_B prefix_for_B_prime`

e.g.:

`python image_analogy.py images/arch-mask.jpg images/arch.jpg images/arch-newmask.jpg out/arch`

Currently, the images are all assumed to be the same size. Output size is the same
as the new mask image, unless specified otherwise.

Parameters
----------

 * --width Sets image output max width
 * --height Sets image output max height
 * --scales Run at N different scales
 * --iters Number of iterations per scale
 * --min-scale Smallest scale to iterate
 * --mrf-w Weight for MRF loss between A' and B'
 * --analogy-w Weight for analogy loss
 * --b-content-w Weight for content loss between B and B'
 * --tv-w Weight for total variation loss
 * --vgg-weights Path to VGG16 weights
 * --a-scale-mode Method of scaling A and A' relative to B
 * * 'match': force A to be the same size as B regardless of aspect ratio (current default for backwards compatibility)
 * * 'ratio': apply scale imposed by width/height params on B to A
 * * 'none': leave A/A' alone
 * --a-scale Additional scale factor for A and A'
 * --pool-mode Pooling style used by VGG
 * * 'avg': average pooling - generally smoother results
 * * 'max': max pooling - more noisy but maybe that's what you want (original default)
 * --contrast adjust the contrast of the output by removing the bottom x percentile
    and scaling by the (100 - x)th percentile. Defaults to 0.02
 * --output-full Output all intermediate images at full size regardless of actual scale
 * --analogy-layers Comma-separated list of layer names to be used for the analogy loss (default: "conv3_1,conv_4_1")
 * --mrf-layers Comma-separated list of layer names to be used for the MRF loss (default: "conv3_1,conv_4_1")
 * --content-layers Comma-separated list of layer names to be used for the content loss (default: "conv3_1,conv_4_1")
 * --patch-size Patch size used for matching (default: 3)
The analogy loss is the amount of influence of B -> A -> A' -> B'
It should be set a lot higher than the MRF loss (default is 9:1)

The MRF loss is the influence of B' -> A' -> B'

The B/B' content loss is set to 0.0 by default. You can get effects similar
to CNNMRF by turning this up and setting analogy weight to zero. Or leave the
analogy loss on for some extra style guidance.

If you'd like to only visualize the analogy target to see what's happening,
set the MRF and content loss to zero: `--mrf-w=0 --content-w=0`
