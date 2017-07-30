'''
TensorFlow Implementation of "GAN for Single Source Audio Separation"

TODO docs
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from math import sqrt, isnan
import argparse
from sys import stdout
from collections import OrderedDict
from functools import reduce
import os
import copy

import numpy as np
import scipy.signal
import scipy.io.wavfile
import tensorflow as tf
# remove annoying "I tensorflow ..." logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import app.datasets as datasets
import app.hparams as hparams
import app.modules as modules
import app.ops as ops
import app.ozers as ozers
import app.utils as utils


# Global vars
g_sess = tf.Session()
g_args = None
g_model = None
g_dataset = None

def _dict_add(dst, src):
    for k,v in src.items():
        if k not in dst:
            dst[k] = v
        else:
            dst[k] += v


def _dict_mul(di, coeff):
    for k,v in di.items():
        di[k] = v * coeff


def _dict_format(di):
    return ' '.join('='.join((k, str(v))) for k,v in di.items())


def load_wavfile(filename):
    '''
    This loads a WAV file, resamples to 16k/sec rate,
    then preprocess it

    Args:
        filename: string

    Returns:
        numpy array of shape [time, FEATURE_SIZE]
    '''
    if filename is None:
        # TODO in this case, draw a sample from dataset instead of raise ?
        raise FileNotFoundError(
                'WAV file not specified, '
                'please specify via --input-file argument.')
    smprate, data = scipy.io.wavfile.read(filename)
    raise NotImplementedError()


def save_wavfile(filename, feature):
    '''
    Saves time series of features into a WAV file

    Args:
        filename: string
        feature: 2D float array of shape [time, FEATURE_SIZE]
    '''
    raise NotImplementedError()


class Model(object):
    '''
    Base class for a fully trainable model

    Should be singleton
    '''
    def __init__(self, name='BaseModel'):
        self.name = name
        self.s_states_di = {}

    def lyr_lstm(
            self, name, s_x, hdim,
            axis=-1, t_axis=0, op_linear=ops.lyr_linear):
        '''
        Args:
            name: string
            s_x: input tensor
            hdim: size of hidden layer
            axis: which axis will RNN op get performed on
            t_axis: which axis would be the timeframe
            op_rnn: RNN layer function, defaults to ops.lyr_lstm
        '''
        x_shp = s_x.get_shape().as_list()
        ndim = len(x_shp)
        assert -ndim <= axis < ndim
        assert -ndim <= t_axis < ndim
        axis = axis % ndim
        t_axis = t_axis % ndim
        assert axis != t_axis
        # make sure t_axis is 0, to make scan work
        if t_axis != 0:
            if axis == 0:
                axis = t_axis % ndim
            perm = list(range(ndim))
            perm[0], perm[t_axis] = perm[t_axis], perm[0]
            s_x = tf.transpose(s_x, perm)
        x_shp[t_axis], x_shp[0] = x_shp[0], x_shp[t_axis]
        idim = x_shp[axis]
        assert isinstance(idim, int)
        h_shp = copy.copy(x_shp[1:])
        h_shp[axis-1] = hdim
        with tf.variable_scope(name):
            zero_init = tf.constant_initializer(0.)
            v_cell = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name='cell',
                trainable=False,
                initializer=zero_init)
            v_hid = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name='hid',
                trainable=False,
                initializer=zero_init)
            self.s_states_di[v_cell.name] = v_cell
            self.s_states_di[v_hid.name] = v_hid

            init_range = 0.1 / sqrt(hdim)
            op_lstm = lambda _h, _x: ops.lyr_lstm_flat(
                name='LSTM',
                s_x=_x, v_cell=_h[0], v_hid=_h[1],
                axis=axis-1, op_linear=op_linear,
                w_init=tf.random_uniform_initializer(
                    -init_range, init_range, dtype=hparams.FLOATX))
            s_cell_seq, s_hid_seq = tf.scan(
                op_lstm, s_x, initializer=(v_cell, v_hid))
        return s_hid_seq if t_axis == 0 else tf.transpose(s_hid_seq, perm)

    def lyr_gru(
            self, name, s_x, hdim,
            axis=-1, t_axis=0, op_linear=ops.lyr_linear):
        '''
        Args:
            name: string
            s_x: input tensor
            hdim: size of hidden layer
            axis: which axis will RNN op get performed on
            t_axis: which axis would be the timeframe
            op_rnn: RNN layer function, defaults to ops.lyr_gru
        '''
        x_shp = s_x.get_shape().as_list()
        ndim = len(x_shp)
        assert -ndim <= axis < ndim
        assert -ndim <= t_axis < ndim
        axis = axis % ndim
        t_axis = t_axis % ndim
        assert axis != t_axis
        # make sure t_axis is 0, to make scan work
        if t_axis != 0:
            if axis == 0:
                axis = t_axis % ndim
            perm = list(range(ndim))
            perm[0], perm[t_axis] = perm[t_axis], perm[0]
            s_x = tf.transpose(s_x, perm)
        x_shp[t_axis], x_shp[0] = x_shp[0], x_shp[t_axis]
        idim = x_shp[axis]
        assert isinstance(idim, int)
        h_shp = copy.copy(x_shp[1:])
        h_shp[axis-1] = hdim
        with tf.variable_scope(name):
            zero_init = tf.constant_initializer(0.)
            v_cell = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name='cell',
                trainable=False,
                initializer=zero_init)
            self.s_states_di[v_cell.name] = v_cell

            init_range = 0.1 / sqrt(hdim)
            op_gru = lambda _h, _x: ops.lyr_gru_flat(
                'GRU', _x, _h[0],
                axis=axis-1, op_linear=op_linear,
                w_init=tf.random_uniform_initializer(
                    -init_range, init_range, dtype=hparams.FLOATX))
            s_cell_seq, = tf.scan(
                op_gru, s_x, initializer=(v_cell,))
        return s_cell_seq if t_axis == 0 else tf.transpose(s_cell_seq, perm)

    def save_params(self, filename, step=None):
        global g_sess
        save_dir = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.saver.save(g_sess,
                        filename,
                        global_step=step)

    def load_params(self, filename):
        # if not os.path.exists(filename):
            # stdout.write('Parameter file "%s" does not exist\n' % filename)
            # return False
        self.saver.restore(g_sess, filename)
        return True

    def build(self):
        # create sub-modules
        encoder = hparams.get_encoder()(
            self, 'encoder')


        # ===================
        # build the model

        input_shape = [
            hparams.BATCH_SIZE,
            hparams.MAX_N_SIGNAL,
            None,
            hparams.FEATURE_SIZE]

        s_src_signals = tf.placeholder(
            hparams.COMPLEXX,
            input_shape,
            name='source_signal')
        s_dropout_keep = tf.placeholder(
            hparams.FLOATX,
            [], name='dropout_keep')
        reger = hparams.get_regularizer()
        with tf.variable_scope('global', regularizer=reger):
            # TODO add mixing coeff ?

            # get mixed signal
            s_mixed_signals = tf.reduce_sum(
                s_src_signals, axis=1)

            s_src_signals_log = tf.log1p(tf.abs(s_src_signals))
            s_mixed_signals_log = tf.log1p(tf.abs(s_mixed_signals))
            # int[B, T, F]
            s_src_assignment = tf.argmax(s_src_signals_log, axis=1)
            # float[B, T, F, E]
            s_embed = encoder(s_mixed_signals_log)
            with tf.name_scope('attractor'):
                s_indices = tf.reshape(
                    s_src_assignment,
                    [hparams.BATCH_SIZE, -1])
                s_embed_flat = tf.reshape(
                    s_embed,
                    [hparams.BATCH_SIZE, -1, hparams.EMBED_SIZE])
                fn_segmean = lambda _: tf.unsorted_segment_sum(
                    _[0], _[1], hparams.MAX_N_SIGNAL)
                # float[B, C, E]
                s_attractors = tf.map_fn(
                    fn_segmean, (s_embed_flat, s_indices), hparams.FLOATX)

            s_logits = tf.matmul(
                s_embed_flat, tf.transpose(s_attractors, [0, 2, 1]))
            s_logits = tf.reshape(
                s_logits, [
                    hparams.BATCH_SIZE,
                    -1, hparams.FEATURE_SIZE,
                    hparams.MAX_N_SIGNAL])

            s_masks = tf.nn.sigmoid(s_logits)
            s_separated_signals = tf.expand_dims(s_mixed_signals_log, -1) * s_masks
            s_separated_signals = tf.transpose(s_separated_signals, [0, 3, 1, 2])
            s_loss = tf.reduce_mean(
                tf.square(s_separated_signals - s_src_signals_log), axis=None)


        # ===============
        # prepare summary
        # TODO add impl & summary for word error rate

        # FIXME gan_loss summary is broken
        with tf.name_scope('summary'):
            s_loss_summary = tf.summary.scalar('loss', s_loss)

        # apply optimizer
        ozer = hparams.get_optimizer()(
            learn_rate=hparams.LR, lr_decay=hparams.LR_DECAY)

        v_params_li = tf.trainable_variables()

        op_sgd_step = ozer.minimize(
            s_loss, var_list=v_params_li)
        self.op_init_params = tf.variables_initializer(v_params_li)
        self.op_init_states = tf.variables_initializer(
            list(self.s_states_di.values()))

        self.train_feed_keys = [
            s_src_signals, s_dropout_keep]
        train_summary = tf.summary.merge(
            [s_loss_summary])
        self.train_fetches = [
            train_summary, dict(loss=s_loss),
            op_sgd_step]

        # TODO more test stuff
        self.test_feed_keys = self.train_feed_keys
        test_summary = tf.summary.merge([s_loss_summary])
        self.test_fetches = [test_summary, dict(loss=s_loss)]

        # TODO inference code
        # self.infer_feed_keys = [s_mixed_signals, s_dropout_keep]
        # self.infer_fetches = dict(signals=s_separated_signals)

        self.saver = tf.train.Saver(var_list=v_params_li)


    def train(self, n_epoch, dataset):
        global g_args
        train_writer = tf.summary.FileWriter(hparams.SUMMARY_DIR, g_sess.graph)
        for i_epoch in range(n_epoch):
            cli_report = OrderedDict()
            for i_batch, data_pt in enumerate(dataset.epoch(
                    'train', hparams.BATCH_SIZE, shuffle=True)):
                to_feed = dict(zip(self.train_feed_keys, data_pt + (hparams.DROPOUT_KEEP_PROB,)))
                step_summary, step_fetch = g_sess.run(
                    self.train_fetches, to_feed)[:2]
                self.reset_state()
                train_writer.add_summary(step_summary)
                stdout.write(':')
                stdout.flush()
                _dict_add(cli_report, step_fetch)
            _dict_mul(cli_report, 1. / (i_batch+1))
            if not g_args.no_save_on_epoch:
                if any(map(isnan, cli_report.values())):
                    if i_epoch:
                        stdout.write('\nEpoch %d/%d got NAN values, restoring last checkpoint ... ')
                        stdout.flush()
                        i_epoch -= 1
                        # FIXME: this path don't work windows
                        self.load_params('saves/' + self.name + ('_e%d' % (i_epoch+1)))
                        stdout.write('done')
                        stdout.flush()
                        continue
                    else:
                        stdout.write('\nRun into NAN during 1st epoch, exiting ...')
                        sys.exit(-1)
                self.save_params('saves/' + self.name + ('_e%d' % (i_epoch+1)))
                stdout.write('S')
            stdout.write('\nEpoch %d/%d %s\n' % (
                i_epoch+1, n_epoch, _dict_format(cli_report)))
            stdout.flush()
            if g_args.no_test_on_epoch:
                continue
            cli_report = OrderedDict()
            for i_batch, data_pt in enumerate(dataset.epoch(
                    'test',
                    hparams.BATCH_SIZE,
                    shuffle=False)):
                # note: disable dropout during test
                to_feed = dict(zip(self.train_feed_keys, data_pt + (1.,)))
                step_summary, step_fetch = g_sess.run(
                    self.test_fetches, to_feed)[:2]
                self.reset_state()
                train_writer.add_summary(step_summary)
                stdout.write('.')
                stdout.flush()
                _dict_add(cli_report, step_fetch)
            _dict_mul(cli_report, 1. / (i_batch+1))
            stdout.write('\nTest  %d/%d %s\n' % (
                i_epoch+1, n_epoch, _dict_format(cli_report)))
            stdout.flush()

    def test(self, dataset):
        global g_args
        train_writer = tf.summary.FileWriter(
            hparams.ASR_SUMMARY_DIR, g_sess.graph)
        cli_report = {}
        for data_pt in dataset.epoch(
                'test', hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL):
            to_feed = dict(zip(self.train_feed_keys, data_pt + (1.,)))
            step_summary, step_fetch = g_sess.run(
                self.test_fetches, to_feed)[:2]
            train_writer.add_summary(step_summary)
            stdout.write('.')
            stdout.flush()
            _dict_add(cli_report, step_fetch)
        stdout.write('Test: %s\n' % (
            _dict_format(cli_report)))

    def reset(self):
        '''re-initialize parameters, resets timestep'''
        g_sess.run(tf.global_variables_initializer())

    def reset_state(self):
        '''reset RNN states'''
        g_sess.run([self.op_init_states])

    def parameter_count(self):
        '''
        Returns: integer
        '''
        v_vars_li = tf.trainable_variables()
        return sum(
            reduce(int.__mul__, v.get_shape().as_list()) for v in v_vars_li)


def main():
    global g_args, g_model, g_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name',
        default='UnamedExperiment',
        help='name of experiment, affects checkpoint saves')
    parser.add_argument('-m', '--mode',
        default='train', help='Mode, "train", "test", "demo" or "interactive"')
    parser.add_argument('-i', '--input-pfile',
        help='path to input model parameter file')
    parser.add_argument('-o', '--output-pfile',
        help='path to output model parameters file')
    parser.add_argument('-ne', '--num-epoch',
        type=int, default=10, help='number of training epoch')
    parser.add_argument('--no-save-on-epoch',
        action='store_true', help="don't save parameter after each epoch")
    parser.add_argument('--no-test-on-epoch',
        action='store_true', help="don't sweep test set after training epoch")
    parser.add_argument('-if', '--input-file',
        help='input WAV file for "demo" mode')
    g_args = parser.parse_args()

    # TODO manage device
    stdout.write('Preparing dataset "%s" ... ' % hparams.DATASET_TYPE)
    stdout.flush()
    g_dataset = hparams.get_dataset()()
    g_dataset.install_and_load()
    stdout.write('done\n')
    stdout.flush()

    print('Encoder type: "%s"' % hparams.ENCODER_TYPE)

    stdout.write('Building model ... ')
    stdout.flush()
    g_model = Model(name=g_args.name)
    g_model.build()
    stdout.write('done\n')

    if g_args.input_pfile is not None:
        stdout.write('Loading paramters from %s ... ' % g_args.input_pfile)
        g_model.load_params(g_args.input_pfile)
        stdout.write('done\n')
    stdout.flush()
    g_model.reset()

    if g_args.mode == 'interactive':
        print('Now in interactive mode, you should run this with python -i')
        return
    elif g_args.mode == 'train':
        g_model.train(n_epoch=g_args.num_epoch, dataset=g_dataset)
        if g_args.output_pfile is not None:
            stdout.write('Saving parameters into %s ... ' % g_args.output_pfile)
            stdout.flush()
            g_model.save_params(g_args.output_pfile)
            stdout.write('done\n')
            stdout.flush()
    elif g_args.mode == 'test':
        g_model.test(dataset=g_dataset)
    elif g_args.mode == 'demo':
        # prepare data point
        if g_args.input_file is None:
            filename = 'demo.wav'
            for features in g_dataset.epoch('test', hparams.MAX_N_SIGNAL):
                break
            save_wavfile(filename, features[0][0] + features[0][1])
            features = np.sum(features[0], axis=0, keepdims=True)
        else:
            filename = g_args.input_file
            features = load_wavfile(g_args.input_file)

        # run with inference mode and save results
        # TODO this has to use whole BATCH_SIZE, inefficient !
        data_pt = (np.tile(features, [hparams.BATCH_SIZE] + [1]*2),)
        result = g_sess.run(
            g_model.infer_fetches,
            dict(zip(
                g_model.infer_feed_keys,
                data_pt + (hparams.DROPOUT_KEEP_PROB,))))
        signals = result['signals'][:(hparams.MAX_N_SIGNAL+1)]
        filename, fileext = os.path.splitext(filename)
        for i, s in enumerate(signals):
            save_wavfile(
                filename + ('_separated_%d' % (i+1)) + fileext, s)

    else:
        raise ValueError(
            'Unknown mode "%s"' % g_args.mode)


def debug_test():
    stdout.write('Building model ... ')
    g_model = Model()
    g_model.build()
    stdout.write('done')
    stdout.flush()
    g_model.reset()


if __name__ == '__main__':
    main()
    # debug_test()
