import copy
from math import sqrt
from functools import partial

import tensorflow as tf

import app.hparams as hparams
import app.ops as ops


class ModelModule(object):
    '''
    abstract class for a sub-module of model

    Args:
        model: Model instance
    '''
    def __init__(self, model, name):
        pass

    def __call__(self, s_dropout_keep=1.):
        raise NotImplementedError()


class Encoder(ModelModule):
    '''
    maps log-magnitude-spectra to embedding
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_mixture, s_dropout_keep=1.):
        '''
        Args:
            s_mixture: tensor variable
                3d tensor of shape [batch_size, length, fft_size]

            s_dropout_keep: scalar const or variable
                keep probability for dropout layer

        Returns:
            [batch_size, length, fft_size, embedding_size]

        Notes:
            `length` is a not constant
        '''
        raise NotImplementedError()


class Estimator(ModelModule):
    '''
    Given embedding in T-F domain, estimates attractor location
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_embed):
        '''
        Args:
            s_embed: tensor of shape [batch_size, length, fft_size, embedding_size]

        Returns:
            s_attractors: tensor of shape [batch_size, num_signals, embedding_size]
        '''
        raise NotImplementedError()


@hparams.register_encoder('toy')
class ToyEncoder(Encoder):
    '''
    This encoder is a 3 layer MLP for debugging purposes
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_signals, s_dropout_keep=1.):
        with tf.variable_scope(self.name):
            s_mid = ops.lyr_linear(
                'linear0', s_signals, hparams.FFT_SIZE*2, axis=-1)
            s_mid = ops.relu(s_mid, hparams.RELU_LEAKAGE)
            s_out = ops.lyr_linear(
                'linear1', s_mid,
                hparams.FEATURE_SIZE * hparams.EMBED_SIZE, axis=-1)
            s_out = tf.reshape(
                s_out,
                [hparams.BATCH_SIZE,
                    -1, hparams.FEATURE_SIZE, hparams.EMBED_SIZE])
        return s_out


@hparams.register_encoder('bilstm-orig')
class BiLstmEncoder(Encoder):
    def __init__(self, model, name):
        self.name = name
        self.model = model

    def __call__(self, s_signals, s_dropout_keep=1.):
        rev_signal = (slice(None), slice(None, None, -1))
        fn_dropout = partial(tf.nn.dropout, keep_prob=s_dropout_keep)

        with tf.variable_scope(self.name):
            s_signals = s_signals - tf.reduce_mean(
                s_signals, axis=(1,2), keep_dims=True)

            s_mid0_fwd = self.model.lyr_lstm(
                'lstm0_fwd', s_signals, 300, t_axis=-2)
            s_mid0_bwd = self.model.lyr_lstm(
                'lstm0_bwd', s_signals[rev_signal], 300, t_axis=-2)
            s_mid0 = tf.concat(
                [s_mid0_fwd, s_mid0_bwd[rev_signal]], axis=-1)
            s_mid0 = fn_dropout(s_mid0)

            s_mid1_fwd = self.model.lyr_lstm(
                'lstm1_fwd', s_mid0, 300, t_axis=-2)
            s_mid1_bwd = self.model.lyr_lstm(
                'lstm1_bwd', s_mid0[rev_signal], 300, t_axis=-2)
            s_mid1 = tf.concat(
                [s_mid1_fwd, s_mid1_bwd[rev_signal]], axis=-1)
            s_mid1 = fn_dropout(s_mid1)

            s_mid1 = s_mid1 - tf.reduce_mean(
                s_mid1, axis=(1,2), keep_dims=True)

            s_mid2_fwd = self.model.lyr_lstm(
                'lstm2_fwd', s_mid1, 300, t_axis=-2)
            s_mid2_bwd = self.model.lyr_lstm(
                'lstm2_bwd', s_mid1[rev_signal], 300, t_axis=-2)
            s_mid2 = tf.concat(
                [s_mid2_fwd, s_mid2_bwd[rev_signal]], axis=-1)
            s_mid2 = fn_dropout(s_mid2)

            s_out_fwd = self.model.lyr_lstm(
                'lstm3_fwd', s_mid2, 300, t_axis=-2)
            s_out_bwd = self.model.lyr_lstm(
                'lstm3_bwd', s_mid2[rev_signal], 300, t_axis=-2)
            s_out = tf.concat(
                [s_out_fwd, s_out_bwd[rev_signal]], axis=-1)
            s_out = fn_dropout(s_out)

            s_out = s_out - tf.reduce_mean(
                s_out, axis=(1,2), keep_dims=True)

            # init_range = 2. / sqrt(300)
            init_range = 1.85
            s_out = ops.lyr_linear(
                'output',
                s_out,
                hparams.FEATURE_SIZE * hparams.EMBED_SIZE,
                w_init=tf.random_uniform_initializer(
                    -init_range, init_range, dtype=hparams.FLOATX),
                bias=None
                )
            s_out = tf.reshape(
                s_out,
                [hparams.BATCH_SIZE, -1, hparams.FEATURE_SIZE, hparams.EMBED_SIZE])
        return s_out


@hparams.register_estimator('anchor')
class AnchoredEstimator(Estimator):
    '''
    Estimate attractor from best combination from
    anchors, then perform 1-step EM
    '''
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_embed):
        with tf.variable_scope(self.name):
            v_anchors = tf.get_variable(
                'anchors', [hparams.NUM_ANCHOR, hparams.EMBED_SIZE],
                initializer=tf.random_normal_initializer(
                    stddev=1.))

            # all combinations of anchors
            s_anchor_sets = ops.combinations(
                v_anchors, hparams.MAX_N_SIGNAL)

            # equation (6)
            s_anchor_assignment = tf.einsum(
                'btfe,pce->bptfc',
                s_embed, s_anchor_sets)
            s_anchor_assignment = tf.nn.softmax(s_anchor_assignment)

            # equation (7)
            s_attractor_sets = tf.einsum(
                'bptfc,btfe->bpce',
                s_anchor_assignment, s_embed)
            s_attractor_sets /= tf.expand_dims(
                tf.reduce_sum(s_anchor_assignment, axis=(2,3)), -1)

            # equation (8)
            s_in_set_similarities = tf.reduce_max(
                tf.matmul(
                    s_attractor_sets,
                    tf.transpose(s_attractor_sets, [0, 1, 3, 2])),
                axis=(-1, -2))

            # equation (9)
            s_subset_choice = tf.argmin(s_in_set_similarities, axis=1)
            s_subset_choice = tf.transpose(tf.stack([
                tf.range(hparams.BATCH_SIZE, dtype=tf.int64),
                s_subset_choice]))
            s_attractors = tf.gather_nd(s_attractor_sets, s_subset_choice)

        if hparams.DEBUG:
            self.s_attractor_sets = s_attractor_sets

        return s_attractors

