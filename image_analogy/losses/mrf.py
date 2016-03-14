from keras import backend as K

from . import patches


def make_patches_grid(x, patch_size, patch_stride):
    '''Break image `x` up into a grid of patches.

    input shape: (channels, rows, cols)
    output shape: (rows, cols, channels, patch_rows, patch_cols)
    '''
    from theano.tensor.nnet.neighbours import images2neibs  # TODO: all K, no T
    x = K.expand_dims(x, 0)
    xs = K.shape(x)
    num_rows = 1 + (xs[-2] - patch_size) // patch_stride
    num_cols = 1 + (xs[-1] - patch_size) // patch_stride
    num_channels = xs[-3]
    patches = images2neibs(x,
        (patch_size, patch_size), (patch_stride, patch_stride),
        mode='valid')
    # neibs are sorted per-channel
    patches = K.reshape(patches, (num_channels, K.shape(patches)[0] // num_channels, patch_size, patch_size))
    patches = K.permute_dimensions(patches, (1, 0, 2, 3))
    # arrange in a 2d-grid (rows, cols, channels, px, py)
    patches = K.reshape(patches, (num_rows, num_cols, num_channels, patch_size, patch_size))
    patches_norm = K.sqrt(K.sum(K.square(patches), axis=(2,3,4), keepdims=True))
    return patches, patches_norm


def mrf_loss(source, combination, patch_size=3, patch_stride=1):
    '''CNNMRF http://arxiv.org/pdf/1601.04589v1.pdf'''
    # extract patches from feature maps
    combination_patches, combination_patches_norm = patches.make_patches(combination, patch_size, patch_stride)
    source_patches, source_patches_norm = patches.make_patches(source, patch_size, patch_stride)
    # find best patches and calculate loss
    patch_ids = patches.find_patch_matches(combination_patches, combination_patches_norm, source_patches / source_patches_norm)
    best_source_patches = K.reshape(source_patches[patch_ids], K.shape(combination_patches))
    loss = K.sum(K.square(best_source_patches - combination_patches)) / patch_size ** 2
    return loss
