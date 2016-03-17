import time

import numpy as np
from keras import backend as K

from image_analogy.losses.analogy import analogy_loss
from image_analogy.losses.core import content_loss
from image_analogy.losses.mrf import mrf_loss
from image_analogy.losses.neural_style import neural_style_loss

from .base import BaseModel


class AnalogyModel(BaseModel):
    '''Model for image analogies.'''

    def build_loss(self, a_image, ap_image, b_image):
        '''Create an expression for the loss as a function of the image inputs.'''
        print('Building loss...')
        loss = super(AnalogyModel, self).build_loss(a_image, ap_image, b_image)
        # Precompute static features for performance
        print('Precomputing static features...')
        all_a_features, all_ap_image_features, all_b_features = self.precompute_static_features(a_image, ap_image, b_image)
        print('Building and combining losses...')
        if self.args.analogy_weight != 0.0:
            for layer_name in self.args.analogy_layers:
                a_features = all_a_features[layer_name][0]
                ap_image_features = all_ap_image_features[layer_name][0]
                b_features = all_b_features[layer_name][0]
                # current combined output
                layer_features = self.get_layer_output(layer_name)
                combination_features = layer_features[0, :, :, :]
                al = analogy_loss(a_features, ap_image_features,
                    b_features, combination_features,
                    use_full_analogy=self.args.use_full_analogy,
                    patch_size=self.args.patch_size,
                    patch_stride=self.args.patch_stride)
                loss += (self.args.analogy_weight / len(self.args.analogy_layers)) * al

        if self.args.mrf_weight != 0.0:
            for layer_name in self.args.mrf_layers:
                ap_image_features = K.variable(all_ap_image_features[layer_name][0])
                layer_features = self.get_layer_output(layer_name)
                # current combined output
                combination_features = layer_features[0, :, :, :]
                sl = mrf_loss(ap_image_features, combination_features,
                    patch_size=self.args.patch_size,
                    patch_stride=self.args.patch_stride)
                loss += (self.args.mrf_weight / len(self.args.mrf_layers)) * sl

        if self.args.b_bp_content_weight != 0.0:
            for layer_name in self.args.b_content_layers:
                b_features = K.variable(all_b_features[layer_name][0])
                # current combined output
                bp_features = self.get_layer_output(layer_name)
                cl = content_loss(bp_features, b_features)
                loss += self.args.b_bp_content_weight / len(self.args.b_content_layers) * cl

        if self.args.neural_style_weight != 0.0:
            for layer_name in self.args.neural_style_layers:
                ap_image_features = K.variable(all_ap_image_features[layer_name][0])
                layer_features = self.get_layer_output(layer_name)
                layer_shape = self.get_layer_output_shape(layer_name)
                # current combined output
                combination_features = layer_features[0, :, :, :]
                nsl = neural_style_loss(ap_image_features, combination_features, 3, self.output_shape[-2], self.output_shape[-1])
                loss += (self.args.neural_style_weight / len(self.args.neural_style_layers)) * nsl
        return loss
