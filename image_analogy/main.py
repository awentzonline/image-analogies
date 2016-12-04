import os
import time

import numpy as np
import scipy.ndimage
from keras import backend as K
from scipy.misc import imsave

from image_analogy import img_utils, vgg16
from image_analogy.optimizer import Optimizer


def main(args, model_class):
    '''The main loop which does the things.'''
    K.set_image_dim_ordering('th')
    # calculate scales
    if args.num_scales > 1:
        step_scale_factor = (1 - args.min_scale) / (args.num_scales - 1)
    else:
        step_scale_factor = 0.0
        args.min_scale = 1.0
    # prepare the input images
    full_ap_image = img_utils.load_image(args.ap_image_path)
    full_a_image = img_utils.load_image(args.a_image_path)
    full_b_image = img_utils.load_image(args.b_image_path)
    # calculate the output size
    full_img_width, full_img_height = calculate_image_dims(args, full_b_image)
    img_num_channels = 3  # TODO: allow alpha
    b_scale_ratio_width = float(full_b_image.shape[1]) / full_img_width
    b_scale_ratio_height = float(full_b_image.shape[0]) / full_img_height
    # ensure the output dir exists
    output_dir = os.path.dirname(args.result_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # multi-scale loop
    x = None  # this is going to hold our output image
    optimizer = Optimizer()
    for scale_i in range(args.num_scales):
        scale_factor = (scale_i * step_scale_factor) + args.min_scale
        # scale our inputs
        img_width = int(round(full_img_width * scale_factor))
        img_height = int(round(full_img_height * scale_factor))
        # prepare the current optimizer state
        if x is None:  # we need to create an initial state
            x = np.random.uniform(0, 255, (img_height, img_width, 3)).astype(np.float32)
            x = vgg16.img_to_vgg(x)
        else:  # resize the last state
            zoom_ratio = img_width / float(x.shape[-1])
            x = scipy.ndimage.zoom(x, (1, zoom_ratio, zoom_ratio), order=1)
            img_height, img_width = x.shape[-2:]
        # determine scaling of "A" images
        if args.a_scale_mode == 'match':
            a_img_width = img_width
            a_img_height = img_height
        elif args.a_scale_mode == 'none':
            a_img_width = full_a_image.shape[1] * scale_factor
            a_img_height = full_a_image.shape[0] * scale_factor
        else:  # should just be 'ratio'
            a_img_width = full_a_image.shape[1] * scale_factor * b_scale_ratio_width
            a_img_height = full_a_image.shape[0] * scale_factor * b_scale_ratio_height
        a_img_width = int(round(args.a_scale * a_img_width))
        a_img_height = int(round(args.a_scale * a_img_height))
        # prepare images for use
        a_image = img_utils.preprocess_image(full_a_image, a_img_width, a_img_height)
        ap_image = img_utils.preprocess_image(full_ap_image, a_img_width, a_img_height)
        b_image = img_utils.preprocess_image(full_b_image, img_width, img_height)
        print('Scale factor {} "A" shape {} "B" shape {}'.format(scale_factor, a_image.shape, b_image.shape))
        # load up the net and create the model
        net = vgg16.get_model(img_width, img_height, weights_path=args.vgg_weights, pool_mode=args.pool_mode)
        model = model_class(net, args)
        model.build(a_image, ap_image, b_image, (1, img_num_channels, img_height, img_width))

        for i in range(args.num_iterations_per_scale):
            print('Start of iteration {} x {}'.format(scale_i, i))
            start_time = time.time()
            if args.color_jitter:
                color_jitter = (args.color_jitter * 2) * (np.random.random((3, img_height, img_width)) - 0.5)
                x += color_jitter
            if args.jitter:
                jitter = args.jitter * scale_factor
                ox, oy = np.random.randint(-jitter, jitter+1, 2)
                x = np.roll(np.roll(x, ox, -1), oy, -2) # apply jitter shift
            # actually run the optimizer
            x, min_val, info = optimizer.optimize(x, model)
            print('Current loss value: {}'.format(min_val))
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
            img = img_utils.deprocess_image(np.copy(x), contrast_percent=args.contrast_percent,resize=out_resize_shape)
            fname = args.result_prefix + '_at_iteration_{}_{}.png'.format(scale_i, i)
            imsave(fname, img)
            end_time = time.time()
            print('Image saved as {}'.format(fname))
            print('Iteration completed in {:.2f} seconds'.format(end_time - start_time,))


def calculate_image_dims(args, full_b_image):
    '''Determine the dimensions of the generated picture.

    Defaults to the size of Image B.
    '''
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
    return full_img_width, full_img_height
