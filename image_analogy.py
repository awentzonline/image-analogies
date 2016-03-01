'''Image Analogies with Keras

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in this script matches the location of the file.

This is adapted from the Keras "neural style transfer" example code.

Run the script with:
```
python image_analogy.py path_to_your_base_image_mask.jpg path_to_your_reference.jpg path_to_new_mask prefix_for_results
```
e.g.:
```
python image_analogy.py images/arch-mask.jpg images/arch.jpg images/arch-newmask.jpg
```

It is preferrable to run this script on GPU, for speed.
'''

from __future__ import print_function
import scipy.ndimage
from scipy.misc import imread, imresize, imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import os
import argparse
import h5py

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import backend as K


parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_mask_image_path', metavar='ref', type=str,
                    help='Path to the reference image mask.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the source image.')
parser.add_argument('new_mask_image_path', metavar='ref', type=str,
                    help='Path to the new mask for generation.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

args = parser.parse_args()
base_image_path = args.base_image_path
base_mask_image_path = args.base_mask_image_path
new_mask_image_path = args.new_mask_image_path
result_prefix = args.result_prefix
weights_path = 'vgg16_weights.h5'

# these are the weights of the different loss components
total_variation_weight = 1.0
analogy_weight = 1.0
style_weight = 1.0
patch_size = 3
patch_stride = 1

analogy_layers = ['conv3_1', 'conv4_1']
mrf_layers = ['conv3_1', 'conv4_1']

# dimensions of the generated picture.
full_img_width = 450
full_img_height = 450

num_iterations_per_scale = 7
num_scales = 3  # run the algorithm at a few different sizes
min_scale_factor = 0.25
if num_scales > 1:
    step_scale_factor = (1 - min_scale_factor) / (num_scales - 1)
else:
    step_scale_factor = 0.0
    min_scale_factor = 1.0

# util function to open, resize and format pictures into appropriate tensors
def load_and_preprocess_image(image_path, img_width, img_height):
    img = preprocess_image(imread(image_path), img_width, img_height)
    return img

def add_vgg_mean(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    return x

def sub_vgg_mean(x):
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(x, img_width, img_height):
    img = imresize(x, (img_height, img_width)).astype('float64')
    img = img[:,:,::-1]  # I think this uses BGR instead of RGB
    img = sub_vgg_mean(img)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x.transpose((1, 2, 0))
    x = add_vgg_mean(x)
    x = x[:,:,::-1]  # back to RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# loss functions

def make_patches(x, patch_size, patch_stride):
    from theano.tensor.nnet.neighbours import images2neibs
    x = K.expand_dims(x, 0)
    patches = images2neibs(x,
        (patch_size, patch_size), (patch_stride, patch_stride),
        mode='valid')
    # neibs are sorted per-channel
    patches = K.reshape(patches, (K.shape(x)[1], K.shape(patches)[0] // K.shape(x)[1], patch_size, patch_size))
    patches = K.permute_dimensions(patches, (1, 0, 2, 3))
    patches_norm = K.l2_normalize(patches, 1)
    return patches, patches_norm

def find_patch_matches(a, b):
    # find the best matching patches
    # we want cross-correlation here so flip the kernels
    convs = K.conv2d(a, b[:, :, ::-1, ::-1], border_mode='valid')
    argmax = K.argmax(convs, axis=1)
    return argmax

# CNNMRF http://arxiv.org/pdf/1601.04589v1.pdf
def mrf_loss(style, combination, patch_size=3, patch_stride=1):
    # extract patches from feature maps
    combination_patches, combination_patches_norm = make_patches(combination, patch_size, patch_stride)
    style_patches, style_patches_norm = make_patches(style, patch_size, patch_stride)
    # find best patches and calculate loss
    patch_ids = find_patch_matches(combination_patches_norm, style_patches_norm)
    best_style_patches = K.reshape(style_patches[patch_ids], K.shape(combination_patches))
    loss = K.sum(K.square(best_style_patches - combination_patches))
    return loss

# http://www.mrl.nyu.edu/projects/image-analogies/index.html
def analogy_loss(a, b, a_prime, b_prime, patch_size=3, patch_stride=1):
    # extract patches from feature maps
    a_patches, a_patches_norm = make_patches(a, patch_size, patch_stride)
    a_prime_patches, a_prime_patches_norm = make_patches(a_prime, patch_size, patch_stride)
    b_patches, b_patches_norm = make_patches(b, patch_size, patch_stride)
    b_prime_patches, b_prime_patches_norm = make_patches(b_prime, patch_size, patch_stride)
    # find best patches and calculate loss
    q = find_patch_matches(a_prime_patches_norm, a_patches_norm)
    best_patches = K.reshape(b_patches[q], K.shape(b_prime_patches))
    loss = K.sum(K.square(best_patches - b_prime_patches))
    return loss

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x, img_width, img_height):
    assert K.ndim(x) == 4
    a = K.square(x[:, :, 1:, :img_width-1] - x[:, :, :img_height-1, :img_width-1])
    b = K.square(x[:, :, :img_height-1, 1:] - x[:, :, :img_height-1, :img_width-1])
    return K.sum(K.pow(a + b, 1.25))

full_base_image = imread(base_image_path)
full_base_mask_image = imread(base_mask_image_path)
full_new_mask_image = imread(new_mask_image_path)

x = None
for scale_i in range(num_scales):
    scale_factor = (scale_i * step_scale_factor) + min_scale_factor
    img_width = int(round(full_img_width * scale_factor))
    img_height = int(round(full_img_height * scale_factor))
    if x is None:
        x = np.random.uniform(0, 255, (img_height, img_width, 3))
        x = x[:,:,::-1]  # to BGR
        x = sub_vgg_mean(x)
        x = x.transpose(2, 0, 1)
    else:  # resize the last state
        zoom_ratio = img_width / float(x.shape[-1])
        x = scipy.ndimage.zoom(x, (1, zoom_ratio, zoom_ratio), order=1)
        img_height, img_width = x.shape[-2:]
    print(scale_factor, x.shape)

    # get tensor representations of our images
    base_image = preprocess_image(full_base_image, img_width, img_height)
    base_mask_image = preprocess_image(full_base_mask_image, img_width, img_height)
    new_mask_image = preprocess_image(full_new_mask_image, img_width, img_height)

    # this will contain our generated image
    vgg_input = K.placeholder((1, 3, img_height, img_width))

    # build the VGG16 network
    first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_height, img_width))
    first_layer.input = vgg_input

    model = Sequential()
    model.add(first_layer)
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        layer = model.layers[k]
        if isinstance(layer, Convolution2D):
            layer.W = layer.W[:, :, ::-1, ::-1]
    f.close()
    print('Model loaded.')

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.get_output()) for layer in model.layers])

    def get_features(x, layers):
        features = {}
        for layer_name in layers:
            f = K.function([vgg_input], outputs_dict[layer_name])
            features[layer_name] = f([x])
        return features

    print('Precomputing static features...')
    all_base_mask_features = get_features(base_mask_image, set(analogy_layers + mrf_layers))
    all_base_image_features = get_features(base_image, set(analogy_layers + mrf_layers))
    all_new_mask_features = get_features(new_mask_image, set(analogy_layers + mrf_layers))

    # combine the loss functions into a single scalar
    print('Building loss function...')
    loss = K.variable(0.)

    for layer_name in analogy_layers:
        layer_features = outputs_dict[layer_name]
        base_mask_features = K.variable(all_base_mask_features[layer_name][0])
        base_image_features = K.variable(all_base_image_features[layer_name][0])
        new_mask_features = K.variable(all_new_mask_features[layer_name][0])
        combination_features = layer_features[0, :, :, :]
        al = analogy_loss(base_mask_features, base_image_features,
            new_mask_features, combination_features)
        loss += (analogy_weight / len(analogy_layers)) * al

    for layer_name in mrf_layers:
        layer_features = outputs_dict[layer_name]
        base_image_features = K.variable(all_base_image_features[layer_name][0])
        combination_features = layer_features[0, :, :, :]
        sl = mrf_loss(base_image_features, combination_features,
            patch_size=patch_size, patch_stride=patch_stride)
        loss += (style_weight / len(mrf_layers)) * sl

    loss += total_variation_weight * total_variation_loss(vgg_input, img_width, img_height)

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, vgg_input)

    outputs = [loss]
    if type(grads) in {list, tuple}:
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([vgg_input], outputs)
    def eval_loss_and_grads(x):
        x = x.reshape((1, 3, img_height, img_width))
        outs = f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    # this Evaluator class makes it possible
    # to compute loss and gradients in one pass
    # while retrieving them via two separate functions,
    # "loss" and "grads". This is done because scipy.optimize
    # requires separate functions for loss and gradients,
    # but computing them separately would be inefficient.
    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    evaluator = Evaluator()

    # run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the neural style loss
    for i in range(num_iterations_per_scale):
        print('Start of iteration', scale_i, i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # save current generated image
        x = x.reshape((3, img_height, img_width))
        img = deprocess_image(np.copy(x))
        fname = result_prefix + '_at_iteration_%d_%d.png' % (scale_i, i)
        imsave(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
