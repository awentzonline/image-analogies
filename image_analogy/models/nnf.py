import time

import numpy as np
from keras import backend as K

from image_analogy.losses.core import content_loss
from image_analogy.losses.nnf import nnf_analogy_loss, NNFState, PatchMatcher
from image_analogy.losses.neural_style import neural_style_loss

from .base import BaseModel


class NNFModel(BaseModel):
    '''Faster model for image analogies.'''
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
        f_inputs = [self.net_input]
        for nnf in self.feature_nnfs:
            f_inputs.append(nnf.placeholder)
        self.f_outputs = K.function(f_inputs, outputs)

    def eval_loss_and_grads(self, x):
        x = x.reshape(self.output_shape)
        f_inputs = [x]
        # update the patch indexes
        # start_t = time.time()
        for nnf in self.feature_nnfs:
            nnf.update(x, num_steps=self.args.mrf_nnf_steps)
            new_target = nnf.matcher.get_reconstruction()
            f_inputs.append(new_target)
        # print('PatchMatch update in {:.2f} seconds'.format(time.time() - start_t))
        # run it through
        outs = self.f_outputs(f_inputs)
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    def build_loss(self, a_image, ap_image, b_image):
        '''Create an expression for the loss as a function of the image inputs.'''
        print('Building loss...')
        loss = super(NNFModel, self).build_loss(a_image, ap_image, b_image)
        # Precompute static features for performance
        print('Precomputing static features...')
        all_a_features, all_ap_image_features, all_b_features = self.precompute_static_features(a_image, ap_image, b_image)
        print('Building and combining losses...')
        if self.args.analogy_weight:
            for layer_name in self.args.analogy_layers:
                a_features = all_a_features[layer_name][0]
                ap_image_features = all_ap_image_features[layer_name][0]
                b_features = all_b_features[layer_name][0]
                # current combined output
                layer_features = self.get_layer_output(layer_name)
                combination_features = layer_features[0, :, :, :]
                al = nnf_analogy_loss(
                    a_features, ap_image_features, b_features, combination_features,
                    num_steps=self.args.analogy_nnf_steps, patch_size=self.args.patch_size,
                    patch_stride=self.args.patch_stride, jump_size=1.0)
                loss += (self.args.analogy_weight / len(self.args.analogy_layers)) * al

        existing_feature_nnfs = getattr(self, 'feature_nnfs', [None] * len(self.args.mrf_layers))
        self.feature_nnfs = []
        if self.args.mrf_weight:
            for layer_name, existing_nnf in zip(self.args.mrf_layers, existing_feature_nnfs):
                ap_image_features = all_ap_image_features[layer_name][0]
                # current combined output
                layer_features = self.get_layer_output(layer_name)
                combination_features = layer_features[0, :, :, :]
                input_shape = self.get_layer_output_shape(layer_name)
                if existing_nnf and not self.args.randomize_mnf_nnf:
                    matcher = existing_nnf.matcher.scale((input_shape[3], input_shape[2], input_shape[1]), ap_image_features)
                else:
                    matcher = PatchMatcher(
                        (input_shape[3], input_shape[2], input_shape[1]), ap_image_features,
                        patch_size=self.args.patch_size, jump_size=1.0, patch_stride=self.args.patch_stride)
                nnf = NNFState(matcher, self.get_f_layer(layer_name))
                self.feature_nnfs.append(nnf)
                sl = content_loss(combination_features, nnf.placeholder)
                loss += (self.args.mrf_weight / len(self.args.mrf_layers)) * sl

        if self.args.b_bp_content_weight:
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
