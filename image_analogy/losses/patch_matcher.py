import numpy as np
import scipy.interpolate
import scipy.ndimage

from .patches_grid import _calc_patch_grid_dims, combine_patches_grid, make_patch_grid, normalize_patches


class PatchMatcher(object):
    '''A matcher of image patches inspired by the PatchMatch algorithm.

    image shape: (width, height, channels)
    '''
    def __init__(self, input_shape, target_img, patch_size=1, patch_stride=1, jump_size=0.5,
            num_propagation_steps=5, num_random_steps=5, random_max_radius=1.0, random_scale=0.5):
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.jump_size = jump_size
        self.num_propagation_steps = num_propagation_steps
        self.num_random_steps = num_random_steps
        self.random_max_radius = random_max_radius
        self.random_scale = random_scale
        self.num_input_rows, self.num_input_cols = _calc_patch_grid_dims(input_shape, patch_size, patch_stride)
        self.target_patches = make_patch_grid(target_img, patch_size)
        self.target_patches_normed = self.normalize_patches(self.target_patches)
        self.coords = np.random.uniform(0.0, 1.0,  # TODO: switch to pixels
            (2, self.num_input_rows, self.num_input_cols))# * [[[self.num_input_rows]],[[self.num_input_cols]]]
        self.similarity = np.zeros(input_shape[:2:-1], dtype ='float32')
        self.min_propagration_row = 1.0 / self.num_input_rows
        self.min_propagration_col = 1.0 / self.num_input_cols
        self.delta_row = np.array([[[self.min_propagration_row]], [[0.0]]])
        self.delta_col = np.array([[[0.0]], [[self.min_propagration_col]]])

    def update(self, input_img, reverse_propagation=False):
        input_patches = self.get_patches_for(input_img)
        self.update_with_patches(self.normalize_patches(input_patches), reverse_propagation=reverse_propagation)

    def update_with_patches(self, input_patches, reverse_propagation=False):
        self._propagate(input_patches, reverse_propagation=reverse_propagation)
        self._random_update(input_patches)

    def get_patches_for(self, img):
        return make_patch_grid(img, self.patch_size);

    def normalize_patches(self, patches):
        return normalize_patches(patches)

    def _propagate(self, input_patches, reverse_propagation=False):
        if reverse_propagation:
            roll_direction = 1
        else:
            roll_direction = -1
        sign = float(roll_direction)
        for step_i in range(self.num_propagation_steps):
            new_coords = self.clip_coords(np.roll(self.coords, roll_direction, 1) + self.delta_row * sign)
            coords_row, similarity_row = self.eval_state(new_coords, input_patches)
            new_coords = self.clip_coords(np.roll(self.coords, roll_direction, 2) + self.delta_col * sign)
            coords_col, similarity_col = self.eval_state(new_coords, input_patches)
            self.coords, self.similarity = self.take_best(coords_row, similarity_row, coords_col, similarity_col)

    def _random_update(self, input_patches):
        for alpha in range(1, self.num_random_steps + 1):  # NOTE this should actually stop when the move is < 1
            new_coords = self.clip_coords(self.coords + np.random.uniform(-self.random_max_radius, self.random_max_radius, self.coords.shape) * self.random_scale ** alpha)
            self.coords, self.similarity = self.eval_state(new_coords, input_patches)

    def eval_state(self, new_coords, input_patches):
        new_similarity = self.patch_similarity(input_patches, new_coords)
        delta_similarity = new_similarity - self.similarity
        coords = np.where(delta_similarity > 0, new_coords, self.coords)
        best_similarity = np.where(delta_similarity > 0, new_similarity, self.similarity)
        return coords, best_similarity

    def take_best(self, coords_a, similarity_a, coords_b, similarity_b):
        delta_similarity = similarity_a - similarity_b
        best_coords = np.where(delta_similarity > 0, coords_a, coords_b)
        best_similarity = np.where(delta_similarity > 0, similarity_a, similarity_b)
        return best_coords, best_similarity

    def patch_similarity(self, source, coords):
        '''Check the similarity of the patches specified in coords.'''
        target_vals = self.lookup_coords(self.target_patches_normed, coords)
        err = source * target_vals
        return np.sum(err, axis=(2, 3, 4))

    def clip_coords(self, coords):
        # TODO: should this all be in pixel space?
        coords = np.clip(coords, 0.0, 1.0)
        return coords

    def lookup_coords(self, x, coords):
        x_shape = np.expand_dims(np.expand_dims(x.shape, -1), -1)
        i_coords = np.round(coords * (x_shape[:2] - 1)).astype('int32')
        return x[i_coords[0], i_coords[1]]

    def get_reconstruction(self, patches=None, combined=None):
        if combined is not None:
            patches = make_patch_grid(combined, self.patch_size)
        if patches is None:
            patches = self.target_patches
        patches = self.lookup_coords(patches, self.coords)
        recon = combine_patches_grid(patches, self.input_shape)
        return recon

    def scale(self, new_shape, new_target_img):
        '''Create a new matcher of the given shape and replace its
        state with a scaled up version of the current matcher's state.
        '''
        new_matcher = PatchMatcher(new_shape, new_target_img, patch_size=self.patch_size,
                patch_stride=self.patch_stride, jump_size=self.jump_size,
                num_propagation_steps=self.num_input_rows,
                num_random_steps=self.num_random_steps,
                random_max_radius=self.random_max_radius,
                random_scale=self.random_scale)
        new_matcher.coords = congrid(self.coords, new_matcher.coords.shape, method='neighbour')
        new_matcher.similarity = congrid(self.similarity, new_matcher.coords.shape, method='neighbour')
        return new_matcher


if __name__ == '__main__':
    import sys
    import time
    from scipy.misc import imsave

    from image_analogy.img_utils import load_image, preprocess_image, deprocess_image

    content_image_path, style_image_path, output_prefix = sys.argv[1:]
    jump_size = 1.0
    num_steps = 7
    patch_size = 1
    patch_stride = 1

    feat_chans = 512
    feat_style_shape = (feat_chans, 12, 18)
    feat_style = np.random.uniform(0.0, 1.0, feat_style_shape)
    feat_in_shape = (feat_chans, 17, 10)
    feat_in = np.random.uniform(0.0, 1.0, feat_in_shape)
    matcher = PatchMatcher(feat_in_shape[::-1], feat_style, patch_size=patch_size)
    feat_in_normed = matcher.normalize_patches(matcher.get_patches_for(feat_in))
    for i in range(num_steps):
        matcher.update_with_patches(feat_in_normed)
    r = matcher.get_reconstruction()

    content_img_img = load_image(content_image_path)
    content_n_channels, content_n_rows, content_n_cols = content_img_img.shape[::-1]
    content_img = preprocess_image(content_img_img, content_n_cols, content_n_rows)[0]#.transpose((2,1,0))
    style_img = load_image(style_image_path)
    style_n_channels, style_n_rows, style_n_cols = content_img_img.shape[::-1]
    style_img =  preprocess_image(
        load_image(style_image_path), style_n_cols, style_n_rows)[0]#.transpose((2,1,0))
    pg = make_patch_grid(content_img, patch_size)
    result = combine_patches_grid(pg, content_img.shape[::-1])
    outimg = deprocess_image(result, contrast_percent=0)
    imsave(output_prefix + '_bestre.png', outimg)

    # # #
    matcher = PatchMatcher((content_n_cols, content_n_rows, content_n_channels), style_img, patch_size=patch_size)
    for i in range(num_steps):
        start = time.time()
        matcher.update(content_img, reverse_propagation=bool(i % 2))
        print(matcher.similarity.min(), matcher.similarity.max(), matcher.similarity.mean())
        end = time.time()
        #print end-start
    start = time.time()
    result = matcher.get_reconstruction(patches=matcher.target_patches)
    print(result.shape)
    end = time.time()
    print(end-start)
    outimg = deprocess_image(result, contrast_percent=0)
    # # imsave takes (rows, cols, channels)
    imsave(output_prefix + '_best.png', outimg)
