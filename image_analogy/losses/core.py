from keras import backend as K


def total_variation_loss(x, num_rows, num_cols):
    '''designed to keep the generated image locally coherent'''
    assert K.ndim(x) == 4
    a = K.square(x[:, :, 1:, :num_cols-1] - x[:, :, :num_rows-1, :num_cols-1])
    b = K.square(x[:, :, :num_rows-1, 1:] - x[:, :, :num_rows-1, :num_cols-1])
    return K.sum(K.pow(a + b, 1.25))


def content_loss(a, b):
    return K.sum(K.square(a - b))
