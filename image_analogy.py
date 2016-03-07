'''Neural Image Analogies with Keras

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in this script matches the location of the file.

This is adapted from the Keras "neural style transfer" example code.

Run the script with:
```
python image_analogy.py path_to_your_ap_image_mask.jpg path_to_your_reference.jpg path_to_b prefix_for_results
```
e.g.:
```
python image_analogy.py images/arch-mask.jpg images/arch.jpg images/arch-newmask.jpg
```

It is preferrable to run this script on GPU, for speed.
'''

from __future__ import print_function
import argparse
import time

import numpy as np
import scipy.ndimage
from scipy.misc import imread, imresize, imsave
from scipy.optimize import fmin_l_bfgs_b
from keras import backend as K

import losses
import vgg16


parser = argparse.ArgumentParser(description='Neural image analogies with Keras.')
parser.add_argument('a_image_path', metavar='ref', type=str,
                    help='Path to the reference image mask (A)')
parser.add_argument('ap_image_path', metavar='base', type=str,
                    help='Path to the source image (A\')')
parser.add_argument('b_image_path', metavar='ref', type=str,
                    help='Path to the new mask for generation (B)')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results (B\')')
parser.add_argument('--width', dest='out_width', type=int,
                    default=0, help='Set output width')
parser.add_argument('--height', dest='out_height', type=int,
                    default=0, help='Set output height')
parser.add_argument('--scales', dest='num_scales', type=int,
                    default=3, help='Run at N different scales')
parser.add_argument('--iters', dest='num_iterations', type=int,
                    default=5, help='Number of iterations per scale')
parser.add_argument('--min-scale', dest='min_scale', type=float,
                    default=0.25, help='Smallest scale to iterate')
parser.add_argument('--mrf-w', dest='mrf_weight', type=float,
                    default=1.0, help='Weight for MRF loss between A\' and B\'')
parser.add_argument('--b-content-w', dest='b_bp_content_weight', type=float,
                    default=0.0, help='Weight for content loss between B and B\'')
parser.add_argument('--analogy-w', dest='analogy_weight', type=float,
                    default=9.0, help='Weight for analogy loss.')
parser.add_argument('--tv-w', dest='tv_weight', type=float,
                    default=1.0, help='Weight for TV loss.')
parser.add_argument('--vgg-weights', dest='vgg_weights', type=str,
                    default='vgg16_weights.h5', help='Path to VGG16 weights.')
parser.add_argument('--a-scale-mode', dest='a_scale_mode', type=str,
                    default='match', help='Method of scaling A and A\' relative to B')
parser.add_argument('--a-scale', dest='a_scale', type=float,
                    default=1.0, help='Additional scale factor for A and A\'')
parser.add_argument('--pool-mode', dest='pool_mode', type=str,
                    default='max', help='Pooling mode for VGG ("avg" or "max")')
parser.add_argument('--jitter', dest='jitter', type=float,
                    default=0, help='Magnitude of random shift at scale x1')
parser.add_argument('--color-jitter', dest='color_jitter', type=float,
                    default=0, help='Magnitude of random jitter to each pixel')
parser.add_argument('--contrast', dest='contrast_percent', type=float,
                    default=0.02, help='Drop the bottom x percentile and scale by the top (100 - x)th percentile')
parser.add_argument('--output-full', dest='output_full_size', action='store_true',
                    help='Output all intermediate images at full size regardless of actual scale.')
parser.add_argument('--analogy-layers', dest='analogy_layers', type=str,
                    default='conv3_1,conv4_1',
                    help='Comma-separated list of layer names to be used for the analogy loss')
parser.add_argument('--mrf-layers', dest='mrf_layers', type=str,
                    default='conv3_1,conv4_1',
                    help='Comma-separated list of layer names to be used for the MRF loss')
parser.add_argument('--content-layers', dest='content_layers', type=str,
                    default='conv3_1,conv4_1',
                    help='Comma-separated list of layer names to be used for the content loss')
parser.add_argument('--patch-size', dest='patch_size', type=int,
                    default=3, help='Patch size used for matching.')

args = parser.parse_args()
a_image_path = args.a_image_path
ap_image_path = args.ap_image_path
b_image_path = args.b_image_path
result_prefix = args.result_prefix
weights_path = args.vgg_weights
a_scale_mode = args.a_scale_mode
assert a_scale_mode in ('ratio', 'none', 'match'), 'a-scale-mode must be set to one of "ratio", "none", or "match"'
# hack for CPU users :(
from keras.backend import theano_backend
if not theano_backend._on_gpu():
    a_scale_mode = 'match'  # prevent conv2d errors when using CPU
    args.a_scale = 1
    print('CPU mode detected. Forcing a-scale-mode to "match"')

# these are the weights of the different loss components
total_variation_weight = args.tv_weight
analogy_weight = args.analogy_weight
b_bp_content_weight = args.b_bp_content_weight
mrf_weight = args.mrf_weight
patch_size = 3
patch_stride = 1

analogy_layers = args.analogy_layers.split(',')
mrf_layers = args.mrf_layers.split(',')
b_content_layers = args.content_layers.split(',')

num_iterations_per_scale = args.num_iterations
num_scales = args.num_scales
min_scale_factor = args.min_scale
if num_scales > 1:
    step_scale_factor = (1 - min_scale_factor) / (num_scales - 1)
else:
    step_scale_factor = 0.0
    min_scale_factor = 1.0

# util function to open, resize and format pictures into appropriate tensors
def load_and_preprocess_image(image_path, img_width, img_height):
    img = preprocess_image(imread(image_path), img_width, img_height)
    return img

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(x, img_width, img_height):
    img = imresize(x, (img_height, img_width), interp='bicubic').astype('float64')
    img = vgg16.img_to_vgg(img)
    img = np.expand_dims(img, axis=0)
    return img

# util function to convert a tensor into a valid image
def deprocess_image(x, contrast_percent=0.0, resize=None):
    x = vgg16.img_from_vgg(x)
    if contrast_percent:
        min_x, max_x = np.percentile(x, (contrast_percent, 100 - contrast_percent))
        x = (x - min_x) * 255.0 / (max_x - min_x)
    x = np.clip(x, 0, 255)
    if resize:
        x = imresize(x, resize, interp='bicubic')
    return x.astype('uint8')

# prepare the input images
full_ap_image = imread(ap_image_path)
full_a_image = imread(a_image_path)
full_b_image = imread(b_image_path)

# dimensions of the generated picture.
# default to the size of the new mask image
full_img_width = full_b_image.shape[1]
full_img_height = full_b_image.shape[0]
if args.out_width or args.out_height:
    if args.out_width and args.out_height:
        full_img_width = args.out_width
        full_img_height = args.out_height
    else:
        if args.out_width:
            full_img_height = int(round(args.out_width / float(full_img_width) * full_img_height))
            full_img_width = args.out_width
        else:
            full_img_width = int(round(args.out_height / float(full_img_height) * full_img_width))
            full_img_height = args.out_height

b_scale_ratio_width = float(full_b_image.shape[1]) / full_img_width
b_scale_ratio_height = float(full_b_image.shape[0]) / full_img_height

x = None
for scale_i in range(num_scales):
    scale_factor = (scale_i * step_scale_factor) + min_scale_factor
    # scale our inputs
    img_width = int(round(full_img_width * scale_factor))
    img_height = int(round(full_img_height * scale_factor))
    img_width, img_height = img_width, img_height
    if x is None:
        x = np.random.uniform(0, 255, (img_height, img_width, 3))
        x = vgg16.img_to_vgg(x)
    else:  # resize the last state
        zoom_ratio = img_width / float(x.shape[-1])
        x = scipy.ndimage.zoom(x, (1, zoom_ratio, zoom_ratio), order=1)
        img_height, img_width = x.shape[-2:]

    if a_scale_mode == 'match':
        a_img_width = img_width
        a_img_height = img_height
    elif a_scale_mode == 'none':
        a_img_width = full_a_image.shape[1] * scale_factor
        a_img_height = full_a_image.shape[0] * scale_factor
    else:  # should just be 'ratio'
        a_img_width = full_a_image.shape[1] * scale_factor * b_scale_ratio_width
        a_img_height = full_a_image.shape[0] * scale_factor * b_scale_ratio_height
    a_img_width = int(round(args.a_scale * a_img_width))
    a_img_height = int(round(args.a_scale * a_img_height))

    a_image = preprocess_image(full_a_image, a_img_width, a_img_height)
    ap_image = preprocess_image(full_ap_image, a_img_width, a_img_height)
    b_image = preprocess_image(full_b_image, img_width, img_height)

    print('Scale factor %s "A" shape %s "B" shape %s' % (scale_factor, a_image.shape, b_image.shape))

    # build the VGG16 network. It seems this needs to be rebuilt at each scale
    # or CPU users get an error from conv2d :(
    model = vgg16.get_model(img_width, img_height, weights_path=weights_path, pool_mode=args.pool_mode)
    first_layer = model.layers[0]
    vgg_input = first_layer.input
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
    all_a_features = get_features(a_image, set(analogy_layers + mrf_layers))
    all_ap_image_features = get_features(ap_image, set(analogy_layers + mrf_layers))
    all_b_features = get_features(b_image, set(analogy_layers + mrf_layers + b_content_layers))

    # combine the loss functions into a single scalar
    print('Building loss function...')
    loss = K.variable(0.)

    if analogy_weight != 0.0:
        for layer_name in analogy_layers:
            a_features = all_a_features[layer_name][0]
            ap_image_features = all_ap_image_features[layer_name][0]
            b_features = all_b_features[layer_name][0]
            layer_features = outputs_dict[layer_name]
            combination_features = layer_features[0, :, :, :]
            al = losses.analogy_loss(a_features, ap_image_features, b_features, combination_features)
            loss += (analogy_weight / len(analogy_layers)) * al

    if mrf_weight != 0.0:
        for layer_name in mrf_layers:
            ap_image_features = K.variable(all_ap_image_features[layer_name][0])
            layer_features = outputs_dict[layer_name]
            combination_features = layer_features[0, :, :, :]
            sl = losses.mrf_loss(ap_image_features, combination_features,
                patch_size=patch_size, patch_stride=patch_stride)
            loss += (mrf_weight / len(mrf_layers)) * sl

    if b_bp_content_weight != 0.0:
        for layer_name in b_content_layers:
            b_features = K.variable(all_b_features[layer_name][0])
            bp_features = outputs_dict[layer_name]
            cl = losses.content_loss(bp_features, b_features)
            loss += b_bp_content_weight / len(b_content_layers) * cl

    loss += total_variation_weight * losses.total_variation_loss(vgg_input, img_width, img_height)

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
        print('Start of iteration %dx%d' % (scale_i, i))
        start_time = time.time()
        if args.color_jitter:
            color_jitter = (args.color_jitter * 2) * (np.random.random((3, img_height, img_width)) - 0.5)
            x += color_jitter
        if args.jitter:
            jitter = args.jitter * scale_factor
            ox, oy = np.random.randint(-jitter, jitter+1, 2)
            x = np.roll(np.roll(x, ox, -1), oy, -2) # apply jitter shift
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # unjitter the image
        x = x.reshape((3, img_height, img_width))
        if args.jitter:
            x = np.roll(np.roll(x, -ox, -1), -oy, -2) # unshift image
        if args.color_jitter:
            x -= color_jitter
        # save the image
        if args.output_full_size:
            out_resize_shape = (full_img_height, full_img_width)
        else:
            out_resize_shape = None
        img = deprocess_image(np.copy(x), contrast_percent=args.contrast_percent,resize=out_resize_shape)
        fname = result_prefix + '_at_iteration_%d_%d.png' % (scale_i, i)
        imsave(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration completed in %ds' % (end_time - start_time,))
