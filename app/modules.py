import copy
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

            s_out = ops.lyr_linear(
                'output',
                s_out,
                hparams.FEATURE_SIZE * hparams.EMBED_SIZE,
                bias=False)
            s_out = tf.reshape(
                s_out,
                [hparams.BATCH_SIZE, -1, hparams.FEATURE_SIZE, hparams.EMBED_SIZE])
        return s_out

