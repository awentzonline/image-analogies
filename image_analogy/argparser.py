import argparse
import os

from keras import backend as K


VGG_ENV_VAR = 'VGG_WEIGHT_PATH'


class CommaSplitAction(argparse.Action):
    '''Split n strip incoming string argument.'''
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [v.strip() for v in values.split(',')])


def parse_args():
    '''Parses command line arguments for the image analogy command.'''
    parser = argparse.ArgumentParser(description='Neural image analogies with Keras.')
    parser.add_argument('a_image_path', metavar='ref', type=str,
                        help='Path to the reference image mask (A)')
    parser.add_argument('ap_image_path', metavar='base', type=str,
                        help='Path to the source image (A\')')
    parser.add_argument('b_image_path', metavar='ref', type=str,
                        help='Path to the new mask for generation (B)')
    parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                        help='Prefix for the saved results (B\')')
    # size-related
    parser.add_argument('--width', dest='out_width', type=int,
                        default=0, help='Set output width')
    parser.add_argument('--height', dest='out_height', type=int,
                        default=0, help='Set output height')
    parser.add_argument('--scales', dest='num_scales', type=int,
                        default=3, help='Run at N different scales')
    parser.add_argument('--min-scale', dest='min_scale', type=float,
                        default=0.25, help='Smallest scale to iterate')
    parser.add_argument('--a-scale-mode', dest='a_scale_mode', type=str,
                        default='none', help='Method of scaling A and A\' relative to B')
    parser.add_argument('--a-scale', dest='a_scale', type=float,
                        default=1.0, help='Additional scale factor for A and A\'')
    parser.add_argument('--output-full', dest='output_full_size', action='store_true',
                        help='Output all intermediate images at full size regardless of current scale.')
    # optimizer
    parser.add_argument('--iters', dest='num_iterations_per_scale', type=int,
                        default=5, help='Number of iterations per scale')
    parser.add_argument('--model', dest='match_model', type=str,
                        default='patchmatch', help='Matching algorithm (patchmatch or brute)')
    parser.add_argument('--mrf-nnf-steps', dest='mrf_nnf_steps', type=int,
                        default=5, help='Number of patchmatch updates per iteration for local coherence loss.')
    parser.add_argument('--randomize-mrf-nnf', dest='randomize_mnf_nnf', action='store_true',
                        help='Randomize the local coherence similarity matrix at the start of a new scale instead of scaling it up.')
    parser.add_argument('--analogy-nnf-steps', dest='analogy_nnf_steps', type=int,
                        default=15, help='Number of patchmatch updates for the analogy loss (done once per scale).')
    # loss
    parser.add_argument('--tv-w', dest='tv_weight', type=float,
                        default=1.0, help='Weight for TV loss.')
    parser.add_argument('--analogy-w', dest='analogy_weight', type=float,
                        default=1.0, help='Weight for analogy loss.')
    parser.add_argument('--analogy-layers', dest='analogy_layers', action=CommaSplitAction,
                        default=['conv3_1', 'conv4_1'],
                        help='Comma-separated list of layer names to be used for the analogy loss')
    parser.add_argument('--use-full-analogy', dest='use_full_analogy', action="store_true",
                        help='Use the full set of analogy patches (slower/more memory but maybe more accurate)')
    parser.add_argument('--mrf-w', dest='mrf_weight', type=float,
                        default=0.5, help='Weight for MRF loss between A\' and B\'')
    parser.add_argument('--mrf-layers', dest='mrf_layers', action=CommaSplitAction,
                        default=['conv3_1', 'conv4_1'],
                        help='Comma-separated list of layer names to be used for the MRF loss')
    parser.add_argument('--b-content-w', dest='b_bp_content_weight', type=float,
                        default=0.0, help='Weight for content loss between B and B\'')
    parser.add_argument('--content-layers', dest='b_content_layers', action=CommaSplitAction,
                        default=['conv3_1', 'conv4_1'],
                        help='Comma-separated list of layer names to be used for the content loss')
    parser.add_argument('--nstyle-w', dest='neural_style_weight', type=float,
                        default=0.0, help='Weight for neural style loss between A\' and B\'')
    parser.add_argument('--nstyle-layers', dest='neural_style_layers', action=CommaSplitAction,
                        default=['conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                        help='Comma-separated list of layer names to be used for the neural style')
    parser.add_argument('--patch-size', dest='patch_size', type=int,
                        default=1, help='Patch size used for matching.')
    parser.add_argument('--patch-stride', dest='patch_stride', type=int,
                        default=1, help='Patch stride used for matching. Currently required to be 1.')
    # VGG
    parser.add_argument('--vgg-weights', dest='vgg_weights', type=str,
                        default=os.environ.get(VGG_ENV_VAR, 'vgg16_weights.h5'), help='Path to VGG16 weights.')
    parser.add_argument('--pool-mode', dest='pool_mode', type=str,
                        default='max', help='Pooling mode for VGG ("avg" or "max")')
    # jitter
    parser.add_argument('--jitter', dest='jitter', type=float,
                        default=0, help='Magnitude of random shift at scale x1')
    parser.add_argument('--color-jitter', dest='color_jitter', type=float,
                        default=0, help='Magnitude of random jitter to each pixel')
    parser.add_argument('--contrast', dest='contrast_percent', type=float,
                        default=0.02, help='Drop the bottom x percentile and scale by the top (100 - x)th percentile')
    args = parser.parse_args()

    assert args.a_scale_mode in ('ratio', 'none', 'match'), 'a-scale-mode must be set to one of "ratio", "none", or "match"'
    if K.backend() == 'tensorflow':
        print('Tensorflow detected. Forcing --a-scale-mode=match (A images are scaled to same size as B images)')
        args.a_scale_mode = 'match'
    # make sure weights are in place
    if not os.path.exists(args.vgg_weights):
        print('Model weights not found (see "--vgg-weights" parameter).')
        return None
    return args
