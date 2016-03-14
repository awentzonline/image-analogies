#!/usr/bin/env python
'''Neural Image Analogies with Keras

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the parameter `vgg_weights` matches the location of the file.

It is preferrable to run this script on GPU, for speed.
'''
import image_analogy.argparser
import image_analogy.main
from image_analogy.models.nnf_cpu import NNFModel


if __name__ == '__main__':
    args = image_analogy.argparser.parse_args()
    if args:
        try:
            image_analogy.main.main(args, NNFModel)
        except KeyboardInterrupt:
            print('Shutting down...')
