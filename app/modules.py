import copy
from math import sqrt
from functools import partial

import numpy as np
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
        def bilstm(
                name_, model_,
                s_input_, hdim_,
                t_axis_, axis_,
                w_init_, b_init_,
                s_dropout_keep_):
            ndim = len(s_input_.get_shape().as_list())
            t_axis_ %= ndim
            rev_signal = (slice(None),)*t_axis_ + (slice(None, None, -1),)
            s_output_fwd = model_.lyr_lstm(
                name_+'_fwd', s_input_, hdim_,
                t_axis=t_axis_, w_init=w_init_, b_init=b_init_)
            s_output_bwd = model_.lyr_lstm(
                name_+'_bwd', s_input_[rev_signal], hdim_,
                t_axis=t_axis_, w_init=w_init_, b_init=b_init_)
            s_output = tf.concat(
                [s_output_fwd, s_output_bwd[rev_signal]], axis=axis_)
            return tf.nn.dropout(s_output, keep_prob=s_dropout_keep_)

        with tf.variable_scope(self.name):
            s_signals = s_signals - tf.reduce_mean(
                s_signals, axis=(1,2), keep_dims=True)

            hdim = 300
            init_range = .75 / sqrt(hdim)
            w_initer = tf.random_uniform_initializer(
                -init_range, init_range, dtype=hparams.FLOATX)

            b_init_value = np.zeros([hdim*4], dtype=hparams.FLOATX)
            b_init_value[hdim*1:hdim*2] = 1.5  # input gate
            b_init_value[hdim*2:hdim*3] = -1.  # forget gate
            b_init_value[hdim*3:hdim*4] = 1.  # output gate
            b_initer = tf.constant_initializer(b_init_value, dtype=hparams.FLOATX)

            s_mid0 = bilstm(
                'lstm0', self.model,
                s_signals, hdim,
                -2, -1,
                w_initer, b_initer, s_dropout_keep)
            s_mid1 = bilstm(
                'lstm1', self.model,
                s_mid0, hdim,
                -2, -1,
                w_initer, b_initer, s_dropout_keep)
            s_mid2 = bilstm(
                'lstm2', self.model,
                s_mid1, hdim,
                -2, -1,
                w_initer, b_initer, s_dropout_keep)
            s_out = bilstm(
                'lstm3', self.model,
                s_mid2, hdim,
                -2, -1,
                w_initer, b_initer, s_dropout_keep)

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
                bias=None)
            s_out = tf.reshape(
                s_out, [
                    hparams.BATCH_SIZE, -1,
                    hparams.FEATURE_SIZE, hparams.EMBED_SIZE])
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

