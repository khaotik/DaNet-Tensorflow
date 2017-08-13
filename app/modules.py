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
    Estimates attractor location, either from TF-embedding,
    or true source
    '''
    USE_TRUTH=True  # set this to true if it uses ground truth
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_embed, **kwargs):
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


# TODO move this into ops.py
def _lyr_bilstm(
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


@hparams.register_encoder('lstm-orig')
class LstmEncoder(Encoder):
    '''
    LSTM network as in original paper
    '''
    def __init__(self, model, name):
        self.name = name
        self.model = model
        self.debug_fetches = {}

    def __call__(self, s_signals, s_dropout_keep=1.):
        with tf.variable_scope(self.name):
            s_signals = s_signals - tf.reduce_mean(
                s_signals, axis=(1,2), keep_dims=True)

            hdim = 600
            init_range = 1.15 / sqrt(hdim)
            w_initer = tf.random_uniform_initializer(
                -init_range, init_range, dtype=hparams.FLOATX)

            b_init_value = np.zeros([hdim*4], dtype=hparams.FLOATX)
            b_init_value[hdim*1:hdim*2] = 1.5  # input gate
            b_init_value[hdim*2:hdim*3] = -1.  # forget gate
            b_init_value[hdim*3:hdim*4] = 1.  # output gate
            b_initer = tf.constant_initializer(b_init_value, dtype=hparams.FLOATX)

            s_mid0 = self.model.lyr_lstm(
                'lstm0', s_signals, hdim,
                t_axis=-2, axis=-1,
                w_init=w_initer, b_init=b_initer)
            s_mid1 = self.model.lyr_lstm(
                'lstm1', s_mid0, hdim,
                t_axis=-2, axis=-1,
                w_init=w_initer, b_init=b_initer)
            s_mid2 = self.model.lyr_lstm(
                'lstm2', s_mid1, hdim,
                t_axis=-2, axis=-1,
                w_init=w_initer, b_init=b_initer)
            s_out = self.model.lyr_lstm(
                'lstm3', s_mid2, hdim,
                t_axis=-2, axis=-1,
                w_init=w_initer, b_init=b_initer)

            s_out = s_out - tf.reduce_mean(
                s_out, axis=(1,2), keep_dims=True)

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



@hparams.register_encoder('bilstm-orig')
class BiLstmEncoder(Encoder):
    '''
    Bi-LSTM network as in original paper
    '''
    def __init__(self, model, name):
        self.name = name
        self.model = model

    def __call__(self, s_signals, s_dropout_keep=1.):
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

            s_mid0 = _lyr_bilstm(
                'lstm0', self.model,
                s_signals, hdim,
                -2, -1,
                w_initer, b_initer, s_dropout_keep)
            s_mid1 = _lyr_bilstm(
                'lstm1', self.model,
                s_mid0, hdim,
                -2, -1,
                w_initer, b_initer, s_dropout_keep)
            s_mid2 = _lyr_bilstm(
                'lstm2', self.model,
                s_mid1, hdim,
                -2, -1,
                w_initer, b_initer, s_dropout_keep)
            s_out = _lyr_bilstm(
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


@hparams.register_encoder('conv-bilstm-v1')
class ConvBiLstmEncoder(Encoder):
    '''
    Experimental CNN-LSTM hybrid network
    '''
    def __init__(self, model, name):
        self.name = name
        self.model = model

    def __call__(self, s_signals, s_dropout_keep=1.):
        with tf.variable_scope(self.name):
            s_signals = tf.expand_dims(s_signals, 1)

            nb = hparams.BATCH_SIZE
            nfft = hparams.FFT_SIZE

            init_range = 2. / sqrt(nfft)
            w_initer = tf.random_uniform_initializer(
                -init_range, init_range, dtype=hparams.FLOATX)

            b_init_value = np.zeros([nfft*4], dtype=hparams.FLOATX)
            b_init_value[nfft*1:nfft*2] = 1.  # input gate
            b_init_value[nfft*2:nfft*3] = -1.  # forget gate
            b_init_value[nfft*3:nfft*4] = 1.  # output gate
            b_initer = tf.constant_initializer(
                b_init_value, dtype=hparams.FLOATX)

            s_mid0 = tf.layers.conv2d(
                s_signals, 8, 5,
                activation=partial(ops.relu, alpha=hparams.RELU_LEAKAGE),
                data_format='channels_first', padding='same')
            s_mid0 = tf.layers.conv2d(
                s_mid0, 16, 5,
                activation=partial(ops.relu, alpha=hparams.RELU_LEAKAGE),
                data_format='channels_first', padding='same')
            # [B, 16, T/2, FFT_SIZE/4]
            s_mid0 = tf.layers.max_pooling2d(
                s_mid0, (2,2), (2,2),data_format='channels_first')

            s_mid1 = tf.layers.conv2d(
                s_mid0, 32, 3,
                activation=partial(ops.relu, alpha=hparams.RELU_LEAKAGE),
                data_format='channels_first', padding='same')
            s_mid1 = tf.layers.conv2d(
                s_mid1, 16, 3,
                activation=partial(ops.relu, alpha=hparams.RELU_LEAKAGE),
                data_format='channels_first', padding='same')
            # [B, 16, T/4, FFT_SIZE/8]
            s_mid1 = tf.layers.max_pooling2d(
                s_mid1, (2,2), (2,2),data_format='channels_first')

            s_mid1 -= tf.reduce_mean(s_mid1, axis=(1,2,3), keep_dims=True)

            # [B, T/4, FFT_SIZE*2]
            s_mid2 = tf.reshape(
                tf.transpose(s_mid1, [0, 2, 1, 3]),
                [nb, -1, nfft*2])
            s_mid2 = _lyr_bilstm(
                'lstm0', self.model,
                s_mid2, nfft,
                -2, -1,
                w_initer, b_initer, s_dropout_keep)
            s_mid3 = _lyr_bilstm(
                'lstm1', self.model,
                s_mid2, nfft,
                -2, -1,
                w_initer, b_initer, s_dropout_keep)
            s_mid3 = tf.transpose(tf.reshape(
                s_mid3, [nb, -1, 16, nfft//8]),
                (0, 2, 1, 3))

            s_mid3 += s_mid1
            s_mid3 -= tf.reduce_mean(s_mid3, axis=(1,2,3), keep_dims=True)

            conv_init_range = 3e-1
            conv_w_initer = tf.random_uniform_initializer(
                -conv_init_range, conv_init_range, dtype=hparams.FLOATX)
            # [B, 16, T/2, FFT_SIZE/4]
            s_mid4 = tf.layers.conv2d(
                s_mid3, 32, 3,
                activation=partial(ops.relu, alpha=hparams.RELU_LEAKAGE),
                data_format='channels_first', padding='same',
                kernel_initializer=conv_w_initer)
            s_mid4 = tf.layers.conv2d(
                s_mid4, 64, 3,
                activation=partial(ops.relu, alpha=hparams.RELU_LEAKAGE),
                data_format='channels_first', padding='same',
                kernel_initializer=conv_w_initer)
            s_mid4 = tf.reshape(s_mid4, [
                nb, 16, 2, 2, -1, nfft//8])
            s_mid4 = tf.transpose(s_mid4, [0, 1, 4, 2, 5, 3])
            s_mid4 = tf.reshape(s_mid4, [nb, 16, -1, nfft//4])

            s_mid5 = tf.layers.conv2d(
                s_mid4, 16, 5,
                activation=partial(ops.relu, alpha=hparams.RELU_LEAKAGE),
                data_format='channels_first', padding='same')
            # [B, 8, T/2, FFT_SIZE/4]
            s_mid5 = tf.layers.conv2d(
                s_mid5, 8, 5,
                activation=partial(ops.relu, alpha=hparams.RELU_LEAKAGE),
                data_format='channels_first', padding='same')
            # [B, T, FFT_SIZE]
            s_mid5 = tf.reshape(
                tf.transpose(s_mid5, [0, 2, 1, 3]),
                [nb, -1, nfft])

            s_out = tf.layers.dense(
                s_mid5, hparams.FEATURE_SIZE * hparams.EMBED_SIZE,
                use_bias=False)
            s_out = tf.reshape(s_out, [
                nb, -1, hparams.FEATURE_SIZE, hparams.EMBED_SIZE])

        if hparams.DEBUG:
            self.debug_fetches = dict(
                conv_act=s_mid1, lstm_act=s_mid3, mid4=s_mid4)

        return s_out


@hparams.register_estimator('truth')
class AverageEstimator(Estimator):
    '''
    Estimate attractor from simple average of true assignment
    '''
    USE_TRUTH = True
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_embed, s_src_pwr, s_mix_pwr, s_embed_flat=None):
        if s_embed_flat is None:
            s_embed_flat = tf.reshape(
                s_embed,
                [hparams.BATCH_SIZE, -1, hparams.EMBED_SIZE])
        with tf.variable_scope(self.name):
            s_src_assignment = tf.argmax(s_src_pwr, axis=1)
            s_indices = tf.reshape(
                s_src_assignment,
                [hparams.BATCH_SIZE, -1])
            fn_segmean = lambda _: tf.unsorted_segment_sum(
                _[0], _[1], hparams.MAX_N_SIGNAL)
            s_attractors = tf.map_fn(
                fn_segmean, (s_embed_flat, s_indices), hparams.FLOATX)
            s_attractors_wgt = tf.map_fn(
                fn_segmean, (tf.ones_like(s_embed_flat), s_indices),
                hparams.FLOATX)
            s_attractors /= (s_attractors_wgt + 1.)

        if hparams.DEBUG:
            self.debug_fetches = dict()
        # float[B, C, E]
        return s_attractors


@hparams.register_estimator('truth-weighted')
class WeightedAverageEstimator(Estimator):
    '''
    Estimate attractor from simple average of true assignment
    '''
    USE_TRUTH = True
    def __init__(self, model, name):
        self.name = name

    def __call__(self, s_embed, s_src_pwr, s_mix_pwr, s_embed_flat=None):
        if s_embed_flat is None:
            s_embed_flat = tf.reshape(
                s_embed,
                [hparams.BATCH_SIZE, -1, hparams.EMBED_SIZE])
        with tf.variable_scope(self.name):
            s_mix_pwr_flat = tf.reshape(
                s_mix_pwr, [hparams.BATCH_SIZE, -1, 1])
            s_src_assignment = tf.argmax(s_src_pwr, axis=1)
            s_indices = tf.reshape(
                s_src_assignment,
                [hparams.BATCH_SIZE, -1])
            fn_segmean = lambda _: tf.unsorted_segment_sum(
                _[0], _[1], hparams.MAX_N_SIGNAL)
            s_attractors = tf.map_fn(fn_segmean, (
                s_embed_flat * s_mix_pwr_flat, s_indices),
                hparams.FLOATX)
            s_attractors_wgt = tf.map_fn(fn_segmean, (
                s_mix_pwr_flat, s_indices),
                hparams.FLOATX)
            s_attractors /= (s_attractors_wgt + hparams.EPS)

        if hparams.DEBUG:
            self.debug_fetches = dict()
        # float[B, C, E]
        return s_attractors


@hparams.register_estimator('anchor')
class AnchoredEstimator(Estimator):
    '''
    Estimate attractor from best combination from
    anchors, then perform 1-step EM
    '''
    USE_TRUTH = False
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
            self.debug_fetches = dict(
                asets=s_attractor_sets,
                anchors=v_anchors,
                subset_choice=s_subset_choice)

        return s_attractors

