import numpy as np
from keras import backend as K

from . import patches
from .core import content_loss


def find_analogy_patches(a, a_prime, b, patch_size=3, patch_stride=1):
    '''This is for precalculating the analogy_loss

    Since A, A', and B never change we only need to calculate the patch matches once.
    '''
    # extract patches from feature maps
    a_patches, a_patches_norm = patches.make_patches(K.variable(a), patch_size, patch_stride)
    a_prime_patches, a_prime_patches_norm = patches.make_patches(K.variable(a_prime), patch_size, patch_stride)
    b_patches, b_patches_norm = patches.make_patches(K.variable(b), patch_size, patch_stride)
    # find best patches and calculate loss
    p = patches.find_patch_matches(b_patches, b_patches_norm, a_patches / a_patches_norm)
    #best_patches = a_prime_patches[p]
    best_patches = K.reshape(a_prime_patches[p], K.shape(b_patches))
    f = K.function([], best_patches)
    best_patches = f([])
    return best_patches


def analogy_loss(a, a_prime, b, b_prime, patch_size=3, patch_stride=1, use_full_analogy=False):
    '''http://www.mrl.nyu.edu/projects/image-analogies/index.html'''
    best_a_prime_patches = find_analogy_patches(a, a_prime, b, patch_size=patch_size, patch_stride=patch_stride)
    if use_full_analogy:  # combine all the patches into a single image
        b_prime_patches, _ = patches.make_patches(b_prime, patch_size, patch_stride)
        loss = content_loss(best_a_prime_patches, b_prime_patches) / patch_size ** 2
    else:
        bs = b.shape
        b_analogy = patches.combine_patches(best_a_prime_patches, (bs[1], bs[2], bs[0]))
        loss = content_loss(np.expand_dims(b_analogy, 0), b_prime)
    return loss
