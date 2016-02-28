neural image analogies
----------------------
This is basically an implementation of "Image Analogies" which is described
here: http://www.mrl.nyu.edu/projects/image-analogies/index.html In this case,
we use feature maps from VGG16. The patch matching and blending is done with a
method described in http://arxiv.org/abs/1601.04589

The code is adapted from the Keras "neural style transfer" example.
The example arch images are from the "Image Analogies" website at http://www.mrl.nyu.edu/projects/image-analogies/index.html

Usage
-----
You'll want to run this on a GPU.

Install latest keras and theano (requires theano, no tensorflow atm).

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in this script matches the location of the file.

`python image_analogy.py path_to_your_base_mask.jpg path_to_your_reference.jpg path_to_new_mask prefix_for_results`

e.g.:

`python image_analogy.py images/arch-mask.jpg images/arch.jpg images/arch-newmask.jpg out/arch`

Example
-------
These images are from the original "Image Analogies" website:

source image (B):

![Image of arch](https://raw.githubusercontent.com/awentzonline/image-analogies/master/images/arch.jpg)

color-coded mask of the source image (A):

![Image of arch mask](https://raw.githubusercontent.com/awentzonline/image-analogies/master/images/arch-mask.jpg)

new color-coded mask describing the desired image (A')

![Image of new arch mask](https://raw.githubusercontent.com/awentzonline/image-analogies/master/images/arch-newmask.jpg)

the synthesized image (B')

![Image of resulting synthesis](https://raw.githubusercontent.com/awentzonline/image-analogies/master/images/arch-result.png)
