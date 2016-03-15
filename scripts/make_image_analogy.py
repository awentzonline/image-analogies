#!/usr/bin/env python
'''Neural Image Analogies with Keras

Before running this script, download the weights for the convolutional layers of
the VGG16 model at:
https://github.com/awentzonline/image-analogies/releases/download/v0.0.5/vgg16_weights.h5
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the parameter `vgg_weights` matches the location of the file.
'''
import time

import image_analogy.argparser
import image_analogy.main


if __name__ == '__main__':
    args = image_analogy.argparser.parse_args()
    if args:
        if args.match_model == 'patchmatch':
            print('Using PatchMatch model')
            from image_analogy.models.nnf import NNFModel as model_class
        else:
            print('Using brute-force model')
            from image_analogy.models.analogy import AnalogyModel as model_class
        start_time = time.time()
        try:
            image_analogy.main.main(args, model_class)
        except KeyboardInterrupt:
            print('Shutting down...')
        print('Done after {:.2f} seconds'.format(time.time() - start_time))
