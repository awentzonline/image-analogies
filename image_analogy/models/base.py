import time

import numpy as np
from keras import backend as K

from image_analogy.losses.core import total_variation_loss


class BaseModel(object):
    '''Model to be extended.'''
    def __init__(self, net, args):
        self.set_net(net)
        self.args = args

    def set_net(self, net):
        self.net = net
        self.net_input = net.layers[0].input
        self.layer_map = dict([(layer.name, layer) for layer in self.net.layers])
        self._f_layer_outputs = {}

    def build(self, a_image, ap_image, b_image, output_shape):
        self.output_shape = output_shape
        loss = self.build_loss(a_image, ap_image, b_image)
        # get the gradients of the generated image wrt the loss
        grads = K.gradients(loss, self.net_input)
        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)
        self.f_outputs = K.function([self.net_input], outputs)

    def build_loss(self, a_image, ap_image, b_image):
        '''Create an expression for the loss as a function of the image inputs.'''
        loss = K.variable(0.0)
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        loss += self.args.tv_weight * total_variation_loss(self.net_input, *b_image.shape[2:])
        return loss

    def precompute_static_features(self, a_image, ap_image, b_image):
        # figure out which layers we need to extract
        a_layers, ap_layers, b_layers = set(), set(), set()
        if self.args.analogy_weight:
            for layerset in (a_layers, ap_layers, b_layers):
                layerset.update(self.args.analogy_layers)
        if self.args.mrf_weight:
            ap_layers.update(self.args.mrf_layers)
        if self.args.b_bp_content_weight:
            b_layers.update(self.args.b_content_layers)
        if self.args.neural_style_weight:
            ap_layers.update(self.args.neural_style_layers)
        # let's get those features
        all_a_features = self.get_features(a_image, a_layers)
        all_ap_image_features = self.get_features(ap_image, ap_layers)
        all_b_features = self.get_features(b_image, b_layers)
        return all_a_features, all_ap_image_features, all_b_features

    def get_features(self, x, layers):
        if not layers:
            return None
        f = K.function([self.net_input], [self.get_layer_output(layer_name) for layer_name in layers])
        feature_outputs = f([x])
        features = dict(zip(layers, feature_outputs))
        return features

    def get_f_layer(self, layer_name):
        return K.function([self.net_input], [self.get_layer_output(layer_name)])

    def get_layer_output(self, name):
        if not name in self._f_layer_outputs:
            layer = self.layer_map[name]
            self._f_layer_outputs[name] = layer.output
        return self._f_layer_outputs[name]

    def get_layer_output_shape(self, name):
        layer = self.layer_map[name]
        return layer.output_shape

    def eval_loss_and_grads(self, x):
        x = x.reshape(self.output_shape)
        outs = self.f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values
