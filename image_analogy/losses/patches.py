from itertools import product

import numpy as np
from keras import backend as K
from sklearn.feature_extraction.image import reconstruct_from_patches_2d


def make_patches(x, patch_size, patch_stride):
    '''Break image `x` up into a bunch of patches.'''
    from theano.tensor.nnet.neighbours import images2neibs
    x = K.expand_dims(x, 0)
    patches = images2neibs(x,
        (patch_size, patch_size), (patch_stride, patch_stride),
        mode='valid')
    # neibs are sorted per-channel
    patches = K.reshape(patches, (K.shape(x)[1], K.shape(patches)[0] // K.shape(x)[1], patch_size, patch_size))
    patches = K.permute_dimensions(patches, (1, 0, 2, 3))
    patches_norm = K.sqrt(K.sum(K.square(patches), axis=(1,2,3), keepdims=True))
    return patches, patches_norm


def reconstruct_from_patches_2d(patches, image_size):
    '''This is from scikit-learn. I thought it was a little overkill
    to require it just for this function.
    '''
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size, dtype=np.float32)
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
    '''Reconstruct an image from these `patches`'''
    patches = patches.transpose(0, 2, 3, 1)
    recon = reconstruct_from_patches_2d(patches, out_shape)
    return recon.transpose(2, 0, 1).astype(np.float32)


def find_patch_matches(a, a_norm, b):
    '''For each patch in A, find the best matching patch in B'''
    convs = None
    if K.backend() == 'theano':
        # HACK: This was not being performed on the GPU for some reason.
        from theano.sandbox.cuda import dnn
        if dnn.dnn_available():
            convs = dnn.dnn_conv(
                img=a, kerns=b[:, :, ::-1, ::-1], border_mode='valid')
    if convs is None:
        convs = K.conv2d(a, b[:, :, ::-1, ::-1], border_mode='valid')
    argmax = K.argmax(convs / a_norm, axis=1)
    return argmax
