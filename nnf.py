import numpy as np
from keras import backend as K

from losses import reconstruct_from_patches_2d


## Patch-related ######################

def make_patches_grid(x, patch_size, patch_stride):
    '''Break image `x` up into a grid of patches.

    input shape: (channels, rows, cols)
    output shape: (rows, cols, channels, patch_rows, patch_cols)
    '''
    from theano.tensor.nnet.neighbours import images2neibs
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


def combine_patches(patches, out_shape):
    '''Reconstruct an image from these `patches`

    input shape: (rows, cols, channels, patch_row, patch_col)
    '''
    #patches = np.transpose(result, (1, 0, 2, 3, 4)) # (cols, rows, channels, pr, pw)
    num_rows, num_cols = patches.shape[:2]
    num_channels = patches.shape[-3]
    num_patches = num_rows * num_cols
    patches = np.reshape(patches, (num_patches, num_channels, patch_size, patch_size))  # (patches, channels, pr, pc)
    patches = np.transpose(patches, (0, 2, 3, 1)) # (channels, patches, p, p)
    recon = reconstruct_from_patches_2d(patches, out_shape)
    return recon.transpose(2, 0, 1)

###### NNF stuff ###################

def make_coords(num_rows, num_cols, random=True):
    '''Generate a grid of coordinates to optimize.'''
    if random:
        grid = [
            np.random.uniform(0.0, 1.0, (num_rows, num_cols)),
            np.random.uniform(0.0, 1.0, (num_rows, num_cols)),
        ]
    else:
        grid = np.meshgrid(
            np.arange(num_rows) / float(num_rows),
            np.arange(num_cols) / float(num_cols), indexing='ij')
    return np.array(grid)


def round_grid_coords(coords, dtype='int32'):
    return K.cast(K.round(coords), dtype)


def clip_coords(coords):
    coords = K.clip(coords, 0.0, 1.0)
    # coords[0,:] = K.clip(coords[0], 0, K.shape(coords)[1] - 1)
    # coords[1,:] = K.clip(coords[1], 1, K.shape(coords)[2] - 1)
    return coords


def lookup_coords(x, coords):
    x_shape = K.expand_dims(K.expand_dims(K.shape(x), -1), -1)
    i_coords = K.cast(K.round(coords * (x_shape[:2] - 1)), 'int32')
    return x[i_coords[0], i_coords[1]]


def get_patch_dims(patches):
    ps = K.shape(patches)
    num_rows, num_cols = ps[:2]
    num_channels = ps[2]
    return num_rows, num_cols, num_channels


def patch_similarity(source, coords, target):
    '''Check the similarity of the patches specified in coords.'''
    target_vals = lookup_coords(target, coords)
    err = source * target_vals[:,:,:,:,::-1]
    return K.squeeze(K.sum(err, axis=(2, 3, 4)), -1)


## RNN funcs ######################

def optimize_step(step_i, states):
    coords, similarity, content_img, style_img, rng, directions = states
    # propagation
    num_propagation_steps = 1
    for sim_i in range(num_propagation_steps):
        coords, similarity = propagate(
            sim_i, coords, similarity, content_img, style_img, directions)
    # random search
    num_random_steps = 8
    scaling_factor = 0.5
    max_radius = 1.0
    for alpha in range(num_random_steps):
        new_coords = clip_coords(coords + rng * max_radius * scaling_factor ** alpha)
        coords, similarity = eval_state(coords, new_coords, similarity, content_img, style_img)
    return coords, [coords, similarity]


def propagate(step_i, coords, similarity, content_img, style_img, directions):
    if step_i % 2:
        roll_direction = -1
    else:
        roll_direction = 1
    sign = float(roll_direction)
    new_coords = clip_coords(T.roll(coords, roll_direction, 1) + K.expand_dims(K.expand_dims(directions[0] * sign, -1), -1))
    coords_row, similarity_row = eval_state(coords, new_coords, similarity, content_img, style_img)
    new_coords = clip_coords(T.roll(coords, roll_direction, 2) + K.expand_dims(K.expand_dims(directions[1] * sign, -1), -1))
    coords_col, similarity_col = eval_state(coords, new_coords, similarity, content_img, style_img)
    coords, similarity = take_best(coords_row, similarity_row, coords_col, similarity_col)
    return coords, similarity


def eval_state(coords, new_coords, similarity, content_img, style_img):
    new_similarity = patch_similarity(content_img, new_coords, style_img)
    delta_similarity = new_similarity - similarity
    coords = K.switch(delta_similarity > 0, new_coords, coords)
    best_similarity = K.switch(delta_similarity > 0, new_similarity, similarity)
    return coords, best_similarity


def take_best(coords_a, similarity_a, coords_b, similarity_b):
    delta_similarity = similarity_a - similarity_b
    best_coords = K.switch(delta_similarity > 0, coords_a, coords_b)
    best_similarity = K.switch(delta_similarity > 0, similarity_a, similarity_b)
    return best_coords, best_similarity

# Interface #####################

def build_nnf(content, style, num_steps=5, jump_size=0.5):
    num_rows, num_cols = content.shape[:2]
    coords = make_coords(num_rows, num_cols)
    initial_error = np.empty((num_rows, num_cols), dtype='float32')
    initial_error.fill(1000000000.0)

    coords_ph = K.placeholder((num_rows, num_cols, 2))
    bs_steps = K.expand_dims(K.expand_dims(K.variable(np.arange(num_steps)), dim=0))
    rng = K.random_normal(coords.shape, 0, jump_size)
    #rng = K.random_uniform(coords.shape, -jump_size, jump_size)
    best_coords, _, _ = K.rnn(
        optimize_step, bs_steps, [coords_ph, initial_error],
        constants=[content, style, rng])

    output_img = lookup_coords(style_img, best_coords)
    f_optimize = K.function([coords_ph], [output_img])

    print('optimizing')
    result = f_optimize([coords])


def mrf_nnf_loss(content, style, num_steps=10, jump_size=0.5, patch_size=3, patch_stride=1):
    '''Inspired by PatchMatch http://gfx.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/patchmatch.pdf

    image shapes: (channels, rows, cols)
    '''
    content_n_channels, content_n_rows, content_n_cols = K.shape(content)
    style_n_channels, style_n_rows, style_n_cols = K.shape(style)
    neighbor_step_row = 1.0 / style_n_rows
    neighbor_step_col = 1.0 / style_n_cols
    # the number of coordinates is determined by the make_patch_grid function
    num_coord_rows = 1 + (content_n_rows - patch_size) // patch_stride
    num_coord_cols = 1 + (content_n_cols - patch_size) // patch_stride
    # extract patches from feature maps
    content_patches, content_patches_norm = make_patch_grid(content, patch_size, patch_stride)
    style_patches, style_patches_norm = make_patch_grid(style, patch_size, patch_stride)

    # set up our initial state
    coords = make_coords(num_coord_rows, num_coord_cols)
    coords_ph = K.placeholder((2, num_coord_rows, num_coord_cols))
    initial_similarity = np.zeros((num_coord_rows, num_coord_cols), dtype='float32')
    initial_similarity = K.variable(initial_similarity)
    directions = K.variable(np.array([[neighbor_step_row, 0.0], [0.0, neighbor_step_col]], dtype='float32'))
    # dumb sequence to turn the crank of the RNN
    bs_steps = K.expand_dims(K.expand_dims(K.variable(np.arange(num_steps)), dim=0))
    rng = K.random_uniform(coords.shape, -jump_size, jump_size)
    print('building patch match rnn')
    best_coords, all_best_coords, _ = K.rnn(
        optimize_step, bs_steps, [coords_ph, initial_similarity],
        constants=[
            content_img_patches / content_patches_norm,
            style_img_patches / style_patches_norm, rng, directions
        ])
    output_patches = lookup_coords(style_img_patches, best_coords)
    f_p = K.function([coords_ph], [output_patches])
    print('optimizing patches')
    patches = f_p([coords])[0]
    loss = K.sum(K.square(best_style_patches - content_patches)) / patch_size ** 2
    return loss
