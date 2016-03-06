from itertools import product

import numpy as np
from keras import backend as K


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

def reconstruct_from_patches_2d(patches, image_size):
    '''This is from scikit-learn. I thought it was a little overkill
    to require it just for this function.
    '''
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i:i + p_h, j:j + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            img[i, j] /= float(min(i + 1, p_h, i_h - i) *
                               min(j + 1, p_w, i_w - j))
    return img

def combine_patches(patches, out_shape):
    patches = patches.transpose(0, 2, 3, 1)
    recon = reconstruct_from_patches_2d(patches, out_shape)
    return recon

def find_patch_matches(a, b):
    # for each patch in A, find the best matching patch in B
    # we want cross-correlation here so flip the kernels
    convs = K.conv2d(a, b[:, :, ::-1, ::-1], border_mode='valid')
    argmax = K.argmax(convs, axis=1)
    return argmax

# CNNMRF http://arxiv.org/pdf/1601.04589v1.pdf
def mrf_loss(source, combination, patch_size=3, patch_stride=1):
    # extract patches from feature maps
    combination_patches, combination_patches_norm = make_patches(combination, patch_size, patch_stride)
    source_patches, source_patches_norm = make_patches(source, patch_size, patch_stride)
    # find best patches and calculate loss
    patch_ids = find_patch_matches(combination_patches_norm, source_patches_norm)
    best_source_patches = K.reshape(source_patches[patch_ids], K.shape(combination_patches))
    loss = K.sum(K.square(best_source_patches - combination_patches))
    return loss

def make_b_from_a_prime(a, a_prime, b, patch_size=3, patch_stride=1):
    # extract patches from feature maps
    a_patches, a_patches_norm = make_patches(K.variable(a), patch_size, patch_stride)
    a_prime_patches, a_prime_patches_norm = make_patches(K.variable(a_prime), patch_size, patch_stride)
    b_patches, b_patches_norm = make_patches(K.variable(b), patch_size, patch_stride)
    # find best patches and calculate loss
    p = find_patch_matches(b_patches_norm, a_patches_norm)
    #best_patches = a_prime_patches[p]
    best_patches = K.reshape(a_prime_patches[p], K.shape(a_prime_patches))
    f = K.function([], best_patches)
    best_patches = f([])
    aps = a_prime.shape
    b_analogy = combine_patches(best_patches, (aps[1], aps[2], aps[0]))
    return b_analogy.transpose(2, 0, 1)

# http://www.mrl.nyu.edu/projects/image-analogies/index.html
def analogy_loss(a, a_prime, b, b_prime, patch_size=3, patch_stride=1):
    # since a, a', b never change we can precalculate the patch matching part
    b_analogy = make_b_from_a_prime(a, a_prime, b, patch_size=patch_size, patch_stride=patch_stride)
    loss = content_loss(np.expand_dims(b_analogy, 0), b_prime)
    return loss

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x, img_width, img_height):
    assert K.ndim(x) == 4
    a = K.square(x[:, :, 1:, :img_width-1] - x[:, :, :img_height-1, :img_width-1])
    b = K.square(x[:, :, :img_height-1, 1:] - x[:, :, :img_height-1, :img_width-1])
    return K.sum(K.pow(a + b, 1.25))

def content_loss(a, b):
    return K.sum(K.square(a - b))
