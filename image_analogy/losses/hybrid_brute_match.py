import numpy as np
from keras import backend as K

from .patches_grid import (
    _calc_patch_grid_dims, combine_patches_grid, extract_patches_2d,
    make_patch_grid, reconstruct_from_patches_2d)


def make_patches(x, patch_size, patch_stride=1):
    '''x shape: (num_channels, rows, cols)'''
    x = x.transpose(2, 1, 0)
    patches = extract_patches_2d(x, (patch_size, patch_size))
    patches = patches.transpose(0, 3, 1, 2) # patches, channels, ph, pw
    return patches


def combine_patches(patches, out_shape):
    '''Reconstruct an image from these `patches`

    input shape: (patches, channels, patch_row, patch_col)
    '''
    num_channels = patches.shape[-3]
    patch_size = patches.shape[-1]
    num_patches = patches.shape[0]
    patches = np.transpose(patches, (0, 2, 3, 1)) # (patches, p, p, channels)
    recon = reconstruct_from_patches_2d(patches, out_shape)
    return recon.transpose(2, 1, 0)  # (channels, rows, cols)


def patches_normal(patches):
    norm = np.sqrt(np.sum(np.square(patches), axis=(1, 2, 3), keepdims=True))
    return norm


class HybridBruteMatcher(object):
    def __init__(self, input_shape, target_img, patch_size=3, patch_stride=1, input_chunk_size=256):
        '''input shape = (num_rows, num_cols, num_channels)'''
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.input_shape = input_shape
        self.input_chunk_size = input_chunk_size
        self.num_input_rows, self.num_input_cols = _calc_patch_grid_dims(input_shape, patch_size, patch_stride)
        self.num_input_channels = self.input_shape[-1]
        self.num_input_patches = self.num_input_rows * self.num_input_cols
        self.target_patches = make_patches(target_img, patch_size)
        self.target_patches_norm = patches_normal(self.target_patches)
        self.target_patched_normed = self.target_patches / self.target_patches_norm
        self.build()
        self.reconstruction = None

    def build(self):
        self.input_placeholder = K.placeholder((self.num_input_patches, self.num_input_channels, self.patch_size, self.patch_size))
        self.input_patches_norm_placeholder = K.placeholder((self.num_input_patches, 1, 1, 1))
        self.chunk_placeholder = K.placeholder(self.target_patches.shape)
        convs = K.conv2d(self.input_placeholder, self.chunk_placeholder[:,:,::-1,::-1], border_mode='valid')
        argmax = K.argmax(convs, axis=1)
        self.f_matches = K.function([self.input_placeholder, self.chunk_placeholder], [argmax])

    def match(self, input_patches, input_patches_norm=None):
        if input_patches_norm is None:
            input_patches_norm = patches_normal(input_patches)
        return self.f_matches([input_patches / input_patches_norm, self.target_patched_normed])[0]

    def reconstruct(self, ids, target=None, target_patches=None):
        if target is not None:
            target_patches = self.get_patches_for(target)
        if target_patches is None:
            target_patches = self.target_patches
        patches = target_patches[ids.flatten()]
        return combine_patches(patches, self.input_shape)

    def get_patches_for(self, img):
        return make_patches(img, self.patch_size);


if __name__ == '__main__':
    import sys
    import time
    from scipy.misc import imsave
    from image_analogy.img_utils import load_image, preprocess_image, deprocess_image

    patch_size = 3

    content_image_path, style_image_path, output_prefix = sys.argv[1:]

    content_img_img = load_image(content_image_path)
    content_n_channels, content_n_rows, content_n_cols = content_img_img.shape[::-1]
    content_img = preprocess_image(content_img_img, content_n_cols, content_n_rows)[0]
    style_img = load_image(style_image_path)
    style_n_channels, style_n_rows, style_n_cols = content_img_img.shape[::-1]
    style_img =  preprocess_image(
        load_image(style_image_path), style_n_cols, style_n_rows)[0]
    # basic patch reconstruction
    pg = make_patches(content_img, patch_size)
    result = combine_patches(pg, content_img.shape[::-1])
    outimg = deprocess_image(result, contrast_percent=0)
    imsave(output_prefix + '_perfect.png', outimg)
    # match-based reconstruction
    matcher = HybridBruteMatcher(
        (content_n_cols, content_n_rows, content_n_channels), style_img,
        patch_size=patch_size)

    content_patches = make_patches(content_img, patch_size)

    for i in range(10):
        start_t = time.time()
        match_ids = matcher.match(content_patches)
        result = matcher.reconstruct(match_ids.flatten())
        print('recons in {:2f} seconds'.format(time.time() - start_t))
    outimg = deprocess_image(result, contrast_percent=0)
    imsave(output_prefix + '_style.png', outimg)
