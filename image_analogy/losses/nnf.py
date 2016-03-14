import numpy as np
from keras import backend as K

from .core import content_loss
from .patch_matcher import PatchMatcher


def nnf_analogy_loss(a, a_prime, b, b_prime, num_steps=5, jump_size=1.0, patch_size=1, patch_stride=1):
    '''image shapes: (channels, rows, cols)
    '''
    bs = b.shape
    matcher = PatchMatcher((bs[2], bs[1], bs[0]), a, jump_size=jump_size, patch_size=patch_size, patch_stride=patch_stride)
    b_patches = matcher.get_patches_for(b)
    b_normed = matcher.normalize_patches(b_patches)
    for i in range(num_steps):
        matcher.update_with_patches(b_normed, reverse_propagation=bool(i % 2))
    target = matcher.get_reconstruction(combined=a_prime)
    loss = content_loss(target, b_prime)
    return loss


class NNFState(object):
    def __init__(self, matcher, f_layer):
        self.matcher = matcher
        mis = matcher.input_shape
        self.placeholder = K.placeholder(mis[::-1])
        self.f_layer = f_layer

    def update(self, x, num_steps=5):
        x_f = self.f_layer([x])[0]
        x_patches = self.matcher.get_patches_for(x_f[0])
        x_normed = self.matcher.normalize_patches(x_patches)
        for i in range(num_steps):
            self.matcher.update_with_patches(x_normed, reverse_propagation=bool(i % 2))
