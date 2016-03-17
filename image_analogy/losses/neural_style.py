'''This is from the keras neural style example.'''
from keras import backend as K


# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram


# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def neural_style_loss(style, combination, num_channels, img_width, img_height):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    size = img_width * img_height
    return K.sum(K.square(S - C)) / (4. * (num_channels ** 2) * (size ** 2))
