'''
TensorFlow Implementation of "Speaker-Independent Speech Separation with Deep Attractor Network"

TODO docs
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from math import sqrt, isnan, ceil
from random import randint
import argparse
from sys import stdout
from collections import OrderedDict
from functools import reduce
from itertools import permutations
from colorsys import hsv_to_rgb
import sys
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
    This loads a WAV file, resamples to hparams.SMPRATE,
    then preprocess it

    Args:
        filename: string

    Returns:
        numpy array of shape [time, FEATURE_SIZE]
    '''
    if filename is None:
        # TODO in this case, draw a sample from dataset instead of raise ?
        raise IOError(
                'WAV file not specified, '
                'please specify via --input-file argument.')
    smprate, data = scipy.io.wavfile.read(filename)
    fft_size = hparams.FFT_SIZE
    fft_stride = hparams.FFT_STRIDE
    if smprate != hparams.SMPRATE:
        data = scipy.signal.resample(
            data, int(ceil(len(data) * hparams.SMPRATE / smprate)))
    Zxx = scipy.signal.stft(
        data,
        window=hparams.FFT_WND,
        nperseg=fft_size,
        noverlap=fft_size - fft_stride)[2]
    return Zxx.astype(hparams.COMPLEXX).T


def save_wavfile(filename, feature):
    '''
    Saves time series of features into a WAV file

    Args:
        filename: string
        feature: 2D float array of shape [time, FEATURE_SIZE]
    '''
    data = utils.istft(
        feature, stride=hparams.FFT_STRIDE, window=hparams.FFT_WND)
    scipy.io.wavfile.write(filename, hparams.SMPRATE, data)


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
            axis=-1, t_axis=0,
            op_linear=ops.lyr_linear,
            w_init=None, b_init=None):
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

            op_lstm = lambda _h, _x: ops.lyr_lstm_flat(
                name='LSTM',
                s_x=_x, v_cell=_h[0], v_hid=_h[1],
                axis=axis-1, op_linear=op_linear,
                w_init=w_init, b_init=b_init)
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

            s_src_signals_pwr = tf.abs(s_src_signals)
            s_mixed_signals_phase = tf.atan2(
                tf.imag(s_mixed_signals), tf.real(s_mixed_signals))
            s_mixed_signals_power = tf.abs(s_mixed_signals)
            s_mixed_signals_log = tf.log1p(s_mixed_signals_power)
            # int[B, T, F]
            # float[B, T, F, E]
            s_embed = encoder(s_mixed_signals_log)
            s_embed_flat = tf.reshape(
                s_embed,
                [hparams.BATCH_SIZE, -1, hparams.EMBED_SIZE])

            # TODO make attractor estimator a submodule ?
            estimator = hparams.get_estimator(
                hparams.TRAIN_ESTIMATOR_METHOD)(self, 'train_estimator')
            s_attractors = estimator(
                s_embed,
                s_src_pwr=s_src_signals_pwr,
                s_mix_pwr=s_mixed_signals_power)

            using_same_method = (
                hparams.INFER_ESTIMATOR_METHOD ==
                hparams.TRAIN_ESTIMATOR_METHOD)

            if using_same_method:
                s_valid_attractors = s_attractors
            else:
                valid_estimator = hparams.get_estimator(
                    hparams.INFER_ESTIMATOR_METHOD
                )(self, 'infer_estimator')
                assert not valid_estimator.USE_TRUTH
                s_valid_attractors = valid_estimator(s_embed)

            separator = hparams.get_separator(
                hparams.SEPARATOR_TYPE)(self, 'separator')
            s_separated_signals_pwr = separator(
                s_mixed_signals_power, s_attractors, s_embed_flat)

            if using_same_method:
                s_separated_signals_pwr_valid = s_separated_signals_pwr
            else:
                s_separated_signals_pwr_valid = separator(
                    s_mixed_signals_power, s_valid_attractors, s_embed_flat)

            # loss and SNR for training
            s_train_loss, v_perms, s_perm_sets = ops.pit_mse_loss(
                s_src_signals_pwr, s_separated_signals_pwr)
            s_perm_idxs = tf.stack([
                tf.tile(
                    tf.expand_dims(tf.range(hparams.BATCH_SIZE), 1),
                    [1, hparams.MAX_N_SIGNAL]),
                tf.gather(v_perms, s_perm_sets)], axis=2)
            s_perm_idxs = tf.reshape(
                s_perm_idxs, [hparams.BATCH_SIZE*hparams.MAX_N_SIGNAL, 2])
            s_separated_signals_pwr = tf.gather_nd(
                s_separated_signals_pwr, s_perm_idxs)
            s_separated_signals_pwr = tf.reshape(
                s_separated_signals_pwr, [
                    hparams.BATCH_SIZE,
                    hparams.MAX_N_SIGNAL,
                    -1, hparams.FEATURE_SIZE])

            s_mixed_signals_phase = tf.expand_dims(s_mixed_signals_phase, 1)
            s_separated_signals = tf.complex(
                tf.cos(s_mixed_signals_phase) * s_separated_signals_pwr,
                tf.sin(s_mixed_signals_phase) * s_separated_signals_pwr)
            s_train_snr = tf.reduce_mean(ops.batch_snr(
                s_src_signals, s_separated_signals, is_complex=True))

            # ^ for validation / inference
            s_valid_loss, v_perms, s_perm_sets = ops.pit_mse_loss(
                s_src_signals_pwr, s_separated_signals_pwr_valid)
            s_perm_idxs = tf.stack([
                tf.tile(
                    tf.expand_dims(tf.range(hparams.BATCH_SIZE), 1),
                    [1, hparams.MAX_N_SIGNAL]),
                tf.gather(v_perms, s_perm_sets)],
                axis=2)
            s_perm_idxs = tf.reshape(
                s_perm_idxs, [hparams.BATCH_SIZE*hparams.MAX_N_SIGNAL, 2])
            s_separated_signals_pwr_valid_pit = tf.gather_nd(
                s_separated_signals_pwr_valid, s_perm_idxs)
            s_separated_signals_pwr_valid_pit = tf.reshape(
                s_separated_signals_pwr_valid_pit, [
                    hparams.BATCH_SIZE,
                    hparams.MAX_N_SIGNAL,
                    -1, hparams.FEATURE_SIZE])

            s_separated_signals_valid = tf.complex(
                tf.cos(s_mixed_signals_phase) * s_separated_signals_pwr_valid_pit,
                tf.sin(s_mixed_signals_phase) * s_separated_signals_pwr_valid_pit)
            s_separated_signals_infer = tf.complex(
                tf.cos(s_mixed_signals_phase) * s_separated_signals_pwr_valid,
                tf.sin(s_mixed_signals_phase) * s_separated_signals_pwr_valid)
            s_valid_snr = tf.reduce_mean(ops.batch_snr(
                s_src_signals, s_separated_signals_valid, is_complex=True))


        # ===============
        # prepare summary
        # TODO add impl & summary for word error rate

        # FIXME gan_loss summary is broken
        with tf.name_scope('train_summary'):
            s_loss_summary_t = tf.summary.scalar('loss', s_train_loss)
            s_snr_summary_t = tf.summary.scalar('SNR', s_train_snr)

        with tf.name_scope('valid_summary'):
            s_loss_summary_v = tf.summary.scalar('loss', s_valid_loss)
            s_snr_summary_v = tf.summary.scalar('SNR', s_valid_snr)

        # apply optimizer
        ozer = hparams.get_optimizer()(
            learn_rate=hparams.LR, lr_decay=hparams.LR_DECAY)

        v_params_li = tf.trainable_variables()
        r_apply_grads = ozer.compute_gradients(s_train_loss, v_params_li)
        if hparams.GRAD_CLIP_THRES is not None:
            r_apply_grads = [(tf.clip_by_value(
                g, -hparams.GRAD_CLIP_THRES, hparams.GRAD_CLIP_THRES), v)
                for g, v in r_apply_grads if g is not None]
        self.op_sgd_step = ozer.apply_gradients(r_apply_grads)

        self.op_init_params = tf.variables_initializer(v_params_li)
        self.op_init_states = tf.variables_initializer(
            list(self.s_states_di.values()))

        self.train_feed_keys = [
            s_src_signals, s_dropout_keep]
        train_summary = tf.summary.merge(
            [s_loss_summary_t, s_snr_summary_t])
        self.train_fetches = [
            train_summary,
            dict(loss=s_train_loss, SNR=s_train_snr),
            self.op_sgd_step]

        self.valid_feed_keys = self.train_feed_keys
        valid_summary = tf.summary.merge([s_loss_summary_v, s_snr_summary_v])
        self.valid_fetches = [
            valid_summary,
            dict(loss=s_valid_loss, SNR=s_valid_snr)]

        self.infer_feed_keys = [s_mixed_signals, s_dropout_keep]
        self.infer_fetches = dict(signals=s_separated_signals_infer)

        if hparams.DEBUG:
            self.debug_feed_keys = [s_src_signals, s_dropout_keep]
            self.debug_fetches = dict(
                embed=s_embed,
                attrs=s_attractors,
                input=s_src_signals,
                output=s_separated_signals)
            self.debug_fetches.update(encoder.debug_fetches)
            self.debug_fetches.update(separator.debug_fetches)
            if estimator is not None:
                self.debug_fetches.update(estimator.debug_fetches)

        self.saver = tf.train.Saver(var_list=v_params_li)


    def train(self, n_epoch, dataset):
        global g_args
        train_writer = tf.summary.FileWriter(hparams.SUMMARY_DIR, g_sess.graph)
        for i_epoch in range(n_epoch):
            cli_report = OrderedDict()
            for i_batch, data_pt in enumerate(dataset.epoch(
                    'train',
                    hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL, shuffle=True)):
                spectra = np.reshape(
                    data_pt[0], [
                        hparams.BATCH_SIZE,
                        hparams.MAX_N_SIGNAL,
                        -1, hparams.FEATURE_SIZE])
                if hparams.MAX_TRAIN_LEN is not None:
                    if spectra.shape[2] > hparams.MAX_TRAIN_LEN:
                        beg = randint(
                            0, spectra.shape[2] - hparams.MAX_TRAIN_LEN-1)
                        spectra = spectra[:, :, beg:beg+hparams.MAX_TRAIN_LEN]
                to_feed = dict(
                    zip(self.train_feed_keys, (
                        spectra, hparams.DROPOUT_KEEP_PROB)))
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
                        stdout.write(
                            '\nEpoch %d/%d got NAN values, restoring last checkpoint ... ')
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
            if g_args.no_valid_on_epoch:
                continue
            cli_report = OrderedDict()
            for i_batch, data_pt in enumerate(dataset.epoch(
                    'valid',
                    hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL,
                    shuffle=False)):
                # note: this disables dropout during validation
                to_feed = dict(
                    zip(self.train_feed_keys, (
                        np.reshape(
                            data_pt[0], [
                                hparams.BATCH_SIZE,
                                hparams.MAX_N_SIGNAL,
                                -1, hparams.FEATURE_SIZE]),
                        1.)))
                step_summary, step_fetch = g_sess.run(
                    self.valid_fetches, to_feed)[:2]
                self.reset_state()
                train_writer.add_summary(step_summary)
                stdout.write('.')
                stdout.flush()
                _dict_add(cli_report, step_fetch)
            _dict_mul(cli_report, 1. / (i_batch+1))
            stdout.write('\nValid  %d/%d %s\n' % (
                i_epoch+1, n_epoch, _dict_format(cli_report)))
            stdout.flush()

    def test(self, dataset):
        global g_args
        train_writer = tf.summary.FileWriter(
            hparams.SUMMARY_DIR, g_sess.graph)
        cli_report = {}
        for data_pt in dataset.epoch(
                'test', hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL):
            # note: this disables dropout during test
            to_feed = dict(
                zip(self.train_feed_keys, (
                    np.reshape(data_pt[0], [hparams.BATCH_SIZE, hparams.MAX_N_SIGNAL, -1, hparams.FEATURE_SIZE]),
                    1.)))
            step_summary, step_fetch = g_sess.run(
                self.valid_fetches, to_feed)[:2]
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
        default='UnnamedExperiment',
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
    parser.add_argument('--no-valid-on-epoch',
        action='store_true',
        help="don't sweep validation set after training epoch")
    parser.add_argument('-if', '--input-file',
        help='input WAV file for "demo" mode')
    parser.add_argument('-ds', '--dataset',
        help='choose dataset to use, overrides hparams.DATASET_TYPE')
    parser.add_argument('-lr', '--learn-rate',
        help='Learn rate, overrides hparams.LR')
    parser.add_argument('-tl', '--train-length',
        help='segment length during training, overrides hparams.MAX_TRAIN_LEN')
    g_args = parser.parse_args()

    # TODO manage device

    # Do override from arguments
    if g_args.learn_rate is not None:
        hparams.LR = float(g_args.learn_rate)
        assert hparams.LR >= 0.
    if g_args.train_length is not None:
        hparams.MAX_TRAIN_LEN = int(g_args.train_length)
        assert hparams.MAX_TRAIN_LEN >= 2
    if g_args.dataset is not None:
        hparams.DATASET_TYPE = g_args.dataset

    stdout.write('Preparing dataset "%s" ... ' % hparams.DATASET_TYPE)
    stdout.flush()
    g_dataset = hparams.get_dataset()()
    g_dataset.install_and_load()
    stdout.write('done\n')
    stdout.flush()

    print('Encoder type: "%s"' % hparams.ENCODER_TYPE)
    print('Separator type: "%s"' % hparams.SEPARATOR_TYPE)
    print('Training estimator type: "%s"' % hparams.TRAIN_ESTIMATOR_METHOD)
    print('Inference estimator type: "%s"' % hparams.INFER_ESTIMATOR_METHOD)

    stdout.write('Building model ... ')
    stdout.flush()
    g_model = Model(name=g_args.name)
    if g_args.mode in ['demo', 'debug']:
        hparams.BATCH_SIZE = 1
        print(
            '\n  Warning: setting hparams.BATCH_SIZE to 1 for "demo" mode'
            '\n... ', end='')
        if g_args.mode == 'debug':
            hparams.DEBUG = True
    g_model.build()
    stdout.write('done\n')

    g_model.reset()
    if g_args.input_pfile is not None:
        stdout.write('Loading paramters from %s ... ' % g_args.input_pfile)
        g_model.load_params(g_args.input_pfile)
        stdout.write('done\n')
    stdout.flush()

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
        colors = np.asarray([
            hsv_to_rgb(h, .95, .98)
            for h in np.arange(
                hparams.MAX_N_SIGNAL, dtype=np.float32
            ) / hparams.MAX_N_SIGNAL])
        if g_args.input_file is None:
            filename = 'demo.wav'
            for src_signals in g_dataset.epoch('test', hparams.MAX_N_SIGNAL):
                break
            max_len = max(map(len, src_signals[0]))
            max_len += (-max_len) % hparams.LENGTH_ALIGN
            src_signals_li = [
                utils.random_zeropad(x, max_len-len(x), axis=-2)
                for x in src_signals[0]]
            src_signals = np.stack(src_signals_li)
            raw_mixture = np.sum(src_signals, axis=0)
            save_wavfile(filename, raw_mixture)
            true_mixture = np.log1p(np.abs(src_signals))
            true_mixture = - np.einsum(
                'nwh,nc->whc', true_mixture, colors)
            true_mixture /= np.min(true_mixture)
        else:
            filename = g_args.input_file
            raw_mixture = load_wavfile(g_args.input_file)
            true_mixture = np.log1p(np.abs(raw_mixture))

        # run with inference mode and save results
        data_pt = (np.expand_dims(raw_mixture, 0),)
        result = g_sess.run(
            g_model.infer_fetches,
            dict(zip(
                g_model.infer_feed_keys,
                data_pt + (hparams.DROPOUT_KEEP_PROB,))))
        signals = result['signals'][0]
        filename, fileext = os.path.splitext(filename)
        for i, s in enumerate(signals):
            save_wavfile(
                filename + ('_separated_%d' % (i+1)) + fileext, s)

        # visualize result
        if 'DISPLAY' not in os.environ:
            print('Warning: no display found, not generating plot')
            return

        import matplotlib.pyplot as plt
        signals = np.log1p(np.abs(signals))
        signals = - np.einsum(
            'nwh,nc->nwhc', signals, colors)
        signals /= np.min(signals)
        for i, s in enumerate(signals):
            plt.subplot(1, len(signals)+2, i+1)
            plt.imshow(np.log1p(np.abs(s)))
        fake_mixture = 0.9 * np.sum(signals, axis=0)
        # fake_mixture /= np.max(fake_mixture)
        plt.subplot(1, len(signals)+2, len(signals)+1)
        plt.imshow(fake_mixture)
        plt.subplot(1, len(signals)+2, len(signals)+2)
        plt.imshow(true_mixture)
        plt.show()
    elif g_args.mode == 'debug':
        import matplotlib.pyplot as plt
        for input_ in g_dataset.epoch(
            'test', hparams.MAX_N_SIGNAL, shuffle=True):
            break
        max_len = max(map(len, input_[0]))
        max_len += (-max_len) % hparams.LENGTH_ALIGN
        input_li = [
            utils.random_zeropad(x, max_len-len(x), axis=-2)
            for x in input_[0]]
        input_ = np.expand_dims(np.stack(input_li), 0)
        data_pt = (input_,)
        debug_data = g_sess.run(
            g_model.debug_fetches,
            dict(zip(
                g_model.debug_feed_keys,
                data_pt + (1.,))))
        debug_data['input'] = input_
        scipy.io.savemat('debug/debug_data.mat', debug_data)
        print('Debug data written to debug/debug_data.mat')
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
