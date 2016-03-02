neural image analogies
----------------------
![Image of arch](https://raw.githubusercontent.com/awentzonline/image-analogies/master/images/image-analogy-explanation.jpg)
![Image of Trump](https://raw.githubusercontent.com/awentzonline/image-analogies/master/images/trump-image-analogy.jpg)

This is basically an implementation of "texture by numbers" from this "Image Analogies" paper http://www.mrl.nyu.edu/projects/image-analogies/index.html In our case, we use feature maps from VGG16. The patch matching and blending is done with a method described in "Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis" http://arxiv.org/abs/1601.04589

The code is adapted from the Keras "neural style transfer" example.

The example arch images are from the "Image Analogies" website at http://www.mrl.nyu.edu/projects/image-analogies/tbn.html
They have some other good examples from their own implementation which
are worth a look.

Installation
------------
You'll want to run this on a GPU.

Install latest keras and theano (requires theano, no tensorflow atm).

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in this script matches the location of the file or use the `--vgg-weights` parameter

Example script usage:
`python image_analogy.py path_to_your_base_mask.jpg path_to_your_reference.jpg path_to_new_mask prefix_for_results`

e.g.:

`python image_analogy.py images/arch-mask.jpg images/arch.jpg images/arch-newmask.jpg out/arch`

Currently, the images are all assumed to be the same size. Output size is the same
as the new mask image, unless specified otherwise.

Parameters
----------

 * --width Set image output max width.
 * --height Set image output max height.
 * --scales Run at N different scales
 * --iters Number of iterations per scale.
 * --min-scale Smallest scale to iterate.
 * --style-w Weight for MRF loss.
 * --analogy-w Weight for analogy loss.
 * --tv-w Weight for TV loss.
 * --vgg-weights Path to VGG16 weights.

The analogy loss is the amount of influence of B -> A -> A' -> B'
The style loss is the influence of B' -> A' -> B
