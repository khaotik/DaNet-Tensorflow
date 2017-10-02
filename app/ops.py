'''
collection of commonly used Ops and layers
'''
from math import factorial
from functools import reduce
import itertools

import numpy as np
import tensorflow as tf

from app.hparams import hparams


def dimshuffle(s_x, *axes, name='dimshuffle'):
    '''
    Emulates theano.dimshuffle

    Args:
        s_x: tensor
        axes: sequence of int or 'x'
        name: string
    '''
    with tf.name_scope(name):
        assert all([i == 'x' or isinstance(i, int) for i in axes])

        to_insert = sorted(i for i,j in enumerate(axes) if j=='x')
        perm = [i for i in axes if isinstance(i, int)]
        assert len(perm) == s_x.get_shape().ndims

        s_y = tf.transpose(s_x, perm)
        for i in to_insert:
            s_y = tf.expand_dims(s_x, i)

    return s_y


def lyr_linear(
        name, s_x, odim,
        axis=-1, bias=True, w_init=None, b_init=None):
    '''
    Like tf.xw_plus_b, but works on arbitrary shape

    Args:
        name: string
        s_x: tensor variable
        odim: integer
        axis: integer
        bias: boolean, whether to use bias
        w_init: initializer for W
        b_init: initializer for B
    '''
    assert isinstance(odim, int)
    x_shape = s_x.get_shape().as_list()
    idim = x_shape[axis]
    ndim = len(x_shape)
    assert -ndim <= axis < ndim
    assert isinstance(idim, int)
    with tf.variable_scope(name):
        v_w = tf.get_variable(
            'W', [idim, odim],
            initializer=w_init,
            dtype=hparams.FLOATX)
        if ndim == 1:
            s_y = tf.matmul(tf.expand_dims(s_x, 0), v_w)
            s_y = tf.squeeze(s_y, 0)
        elif ndim == 2:
            if axis % 2 == 1:
                s_y = tf.matmul(s_x, v_w)
            else:
                s_y = tf.matmul(tf.transpose(s_x), v_w)
                s_y = tf.transpose(s_x)
        elif (axis+1) % ndim == 0:
            s_batch_shp = tf.shape(s_x)[:-1]
            s_x = tf.reshape(
                s_x,
                [tf.reduce_prod(s_batch_shp, axis=None), x_shape[-1]])
            s_y = tf.matmul(s_x, v_w)
            s_y = tf.reshape(s_y, tf.concat([s_batch_shp, [odim]], axis=0))
        else:
            s_y = tf.tensordot(s_x, v_w, [[axis], [0]])
        if bias:
            if b_init is None:
                b_init = tf.constant_initializer(0., dtype=hparams.FLOATX)
            v_b = tf.get_variable(
                    'B', [odim],
                    initializer=b_init,
                    dtype=hparams.FLOATX)
            s_b = tf.reshape(v_b, [odim] + [1] * (ndim - (axis % ndim) - 1))
            s_y = s_y + s_b
    return s_y


def relu(s_x, alpha=0.):
    '''
    Leaky relu. Same as relu when alpha is 0.

    Same as theano.tensor.nnet.relu

    Args:
        s_x: input
        alpha: float constant, default 0.
    '''
    if alpha == 0.:
        s_y = tf.nn.relu(s_x)
    else:
        s_y = tf.maximum(s_x*alpha, s_x)
    return s_y


def lyr_lstm_flat(
        name, s_x, v_cell, v_hid, axis=-1, op_linear=lyr_linear,
        w_init=None, b_init=None):
    '''
    Generic LSTM layer that works with arbitrary shape & linear operator

    Args:
        name: string
        s_x: symbolic tensor
        v_cell: tensor variable
        v_hid: tensor variable
        axis: integer, which axis to perform linear operation
        op_linear: linear operation

    Returns:
        (s_cell_tp1, s_hid_tp1)

    Notes:
        - It's a *flat* layer, which means it doesn't create state variable
        - The size of s_x along axis must be known
    '''
    idim = s_x.get_shape().as_list()[axis]
    assert idim is not None
    cell_shp = v_cell.get_shape().as_list()
    hid_shp = v_hid.get_shape().as_list()
    hdim = cell_shp[axis]
    assert hdim == hid_shp[axis]

    with tf.variable_scope(name):
        s_inp = tf.concat([s_x, v_hid], axis=axis)
        s_act = op_linear(
            'linear', s_inp, hdim*4,
            axis=axis, w_init=w_init, b_init=b_init)
        s_cell_new, s_gates = tf.split(s_act, [hdim, hdim*3], axis=axis)
        s_igate, s_fgate, s_ogate = tf.split(
            tf.nn.sigmoid(s_gates), 3, axis=axis)
        s_cell_tp1 = s_igate*s_cell_new + s_fgate*v_cell
        s_hid_tp1 = s_ogate * tf.tanh(s_cell_tp1)
    return (s_cell_tp1, s_hid_tp1)


def lyr_gru_flat(
        name, s_x, v_cell, axis=-1,
        op_linear=lyr_linear, w_init=None, b_init=None):
    '''
    Generic GRU layer that works with arbitrary shape & linear operator

    Args:
        name: string
        s_x: symbolic tensor
        v_cell: tensor variable, state of GRU RNN
        axis: integer, which axis to perform linear operation
        op_linear: linear operation

    Returns:
        (s_cell_tp1,)

    Notes:
        The size of s_x along axis must be known
    '''
    idim = s_x.get_shape().as_list()[axis]
    assert idim is not None
    cell_shp = v_cell.get_shape().as_list()
    hdim = cell_shp[axis]

    if b_init is None:
        b_init = tf.constant_initializer(1., dtype=hparams.FLOATX)

    with tf.variable_scope(name):
        s_inp = tf.concat([s_x, v_cell], axis=axis)
        s_act = op_linear('gates', s_inp, hdim*2, axis=axis)
        s_rgate, s_igate = tf.split(tf.nn.sigmoid(s_act), 2, axis=axis)
        s_inp2 = tf.concat([s_x, v_cell * s_rgate], axis=axis)
        s_cell_new = op_linear(
            'linear',
            s_inp2, hdim, axis=axis, w_init=w_init, b_init=b_init)
        s_cell_new = tf.tanh(s_cell_new)
        s_cell_tp1 = v_cell * s_igate + s_cell_new * (1.-s_igate)
    return (s_cell_tp1,)


def batch_snr(clear_signal, noisy_signal):
    '''
    batched signal to noise ratio, assuming zero mean

    Args:
        clear_signal: batched array
        noisy_signal: batched_array

    Returns: vector of shape [batch_size]
    '''
    clear_signal_shp = clear_signal.get_shape().as_list()
    noisy_signal_shp = noisy_signal.get_shape().as_list()
    ndim = len(clear_signal_shp)
    reduce_axes = list(range(1, ndim))
    assert len(noisy_signal_shp) == ndim
    noise = clear_signal - noisy_signal

    if clear_signal.dtype.is_complex and noisy_signal.dtype.is_complex:
        clear_signal = tf.abs(clear_signal)
        noise = tf.abs(noise)

    if reduce_axes:
        signal_pwr = tf.reduce_mean(
            tf.square(clear_signal), axis=reduce_axes)
        noise_pwr = tf.reduce_mean(
            tf.square(noise), axis=reduce_axes)
    else:
        signal_pwr = tf.square(clear_signal)
        noise_pwr = tf.square(noise)

    coeff = 4.342944819
    return coeff * (tf.log(signal_pwr + hparams.EPS) - tf.log(noise_pwr + hparams.EPS))


def batch_cross_snr(clear_signal, noisy_signal):
    '''
    signal to noise raio, assuming zero mean

    Args:
        clear_signal: array of shape [batch_size, m, ...]
        noisy_signal: array of shape [batch_size, n, ...]

    Returns:
        array of shape [batch_size, m, n]
    '''
    clear_signal_shp = clear_signal.get_shape().as_list()
    noisy_signal_shp = noisy_signal.get_shape().as_list()
    ndim = len(clear_signal_shp)
    assert len(noisy_signal_shp) == ndim
    assert ndim >= 2

    clear_signal = tf.expand_dims(clear_signal, 2)  # [b, m, 1, ...]
    noisy_signal = tf.expand_dims(noisy_signal, 1)  # [b, 1, n, ...]
    noise = clear_signal - noisy_signal
    reduce_axes = list(range(3, ndim+1))

    if reduce_axes:
        signal_pwr = tf.reduce_mean(
            tf.square(clear_signal), axis=reduce_axes)
        noise_pwr = tf.reduce_mean(
            tf.square(noise), axis=reduce_axes)
    else:
        signal_pwr = tf.square(clear_signal)
        noise_pwr = tf.square(noise)

    coeff = 4.342944819
    return coeff * (
        tf.log(signal_pwr + hparams.EPS) - tf.log(noise_pwr + hparams.EPS))


def batch_segment_mean(s_data, s_indices, n):
    s_data_shp = tf.shape(s_data)
    s_data_flat = tf.reshape(
        s_data, [tf.prod(s_data_shp[:-1]), s_data_shp[-1]])
    s_indices_flat = tf.reshape(s_indices, [-1])
    s_results = tf.unsorted_segment_sum(s_data_flat, s_indices_flat, n)
    s_weights = tf.unsorted_segment_sum(
        tf.ones_like(s_indices_flat),
        s_indices_flat, n)
    return s_results / tf.cast(tf.expand_dims(s_weights, -1), hparams.FLOATX)


def combinations(s_data, subset_size, total_size=None, name=None):
    assert isinstance(subset_size, int)
    assert subset_size > 0
    if total_size is None:
        total_size = s_data.get_shape().as_list()[0]

    if total_size is None:
        raise ValueError(
            "tensor size on axis 0 is unknown,"
            " please supply 'total_size'")
    else:
        assert isinstance(total_size, int)
        assert subset_size <= total_size

    c_combs = tf.constant(
        list(itertools.combinations(range(total_size), subset_size)),
        dtype=hparams.INTX,
        name=('combs' if name is None else name))

    return tf.gather(s_data, c_combs)


def perm_argmin(
        s_x, axes=(-2, -1), perm_size=None,
        keep_dims=False, name='pi_argmin', _cache={}):
    '''
    This Op finds permutation that gives smallest sum, in a batched manner.

    Args:
        s_x: tensor
        axes: 2-tuple of int, tensor shape on the two axes must be the same
        perm_size: size of permutation, infer from tensor shape by default
        name: string
        _cache: DON'T USE, internal variable

    Returns:
        returns (outputs permutations)
        outputs is the indices of permutation, its shape removes the two axes.
        permutations is a constant tensor, a stack one-hot permutation matrices.
    '''
    assert isinstance(axes, tuple) and (len(axes) == 2)
    assert isinstance(perm_size, (type(None), int))
    x_shp = s_x.get_shape().as_list()
    x_ndim = s_x.get_shape().ndims
    if x_ndim < 2:
        raise ValueError(
            'Must be a tensor with at least rank-2, got %d' % x_ndim)
    assert -x_ndim <= axes[0] < x_ndim
    assert -x_ndim <= axes[1] < x_ndim
    axes = (axes[0] % x_ndim, axes[1] % x_ndim)
    axes = tuple(sorted(axes))
    assert isinstance(axes[0], int)
    assert isinstance(axes[1], int)

    perm_sizes = [x_shp[a] for a in axes]
    perm_sizes.append(perm_size)
    perm_sizes = [s for s in perm_sizes if s is not None]
    if not len(perm_sizes):
        raise ValueError('Unknown permutation size.')
    if not reduce(int.__eq__, perm_sizes):
        raise ValueError('Conflicting permutation size.')
    perm_size = perm_sizes[0]
    if perm_size <= 0:
        raise ValueError('Permutation size must be positive, got %d' % perm_size)
    if perm_size not in _cache:
        num_perm = factorial(perm_size)
        v_perms = np.asarray(
            list(itertools.permutations(range(perm_size))),
            dtype=hparams.INTX)
        v_perms_mask = np.zeros(
            [num_perm, perm_size, perm_size],
            dtype=hparams.INTX)
        v_perms_mask[
            np.arange(num_perm),
            np.arange(perm_size),
            v_perms] = 1
        import pdb; pdb.set_trace();
        v_perms_mask = tf.constant(
            v_perms_mask, dtype=hparams.INTX,
            )
        _cache[perm_size] = v_perms_mask
    else:
        v_perms_mask = _cache[perm_size]

    out_shp = x_shp.copy()
    out_shp[axes[0]] = perm_size
    del(out_shp[axes[1]])

    shuffle_li = ['x'] * (x_ndim + 1)
    shuffle_li[-1] = 0
    shuffle_li[axes[0]] = 1
    shuffle_li[axes[1]] = 2
    s_perms_mask = dimshuffle(v_perms_mask, *shuffle_li)
    s_output = tf.argmin(
        tf.reduce_sum(
            s_x * s_perms_mask,
            axis=axes, keep_dims=keep_dims),
        axis=-1)
    return s_output, v_perms_mask


def pit_mse_loss(s_x, s_y, pit_axis=1, perm_size=None, name='pit_loss'):
    '''
    Permutation invariant MSE loss, batched version

    Args:
        s_x: tensor
        s_y: tensor
        pit_axis: which axis permutations occur
        perm_size: size of permutation, infer from tensor shape by default
        name: string

    Returns:
        s_loss, v_perms, s_loss_sets_idx

        s_loss: scalar loss
        v_perms: constant int matrix of permutations
        s_perm_sets_idx: int matrix, indicating selected permutations

    '''
    x_shp = s_x.get_shape().as_list()
    ndim = len(x_shp)

    batch_size = x_shp[0]
    if batch_size is None:
        batch_size = hparams.BATCH_SIZE

    assert -ndim <= pit_axis < ndim
    pit_axis %= ndim
    assert pit_axis != 0
    reduce_axes = [
        i for i in range(1, ndim+1) if i not in [pit_axis, pit_axis+1]]
    with tf.variable_scope(name):
        v_perms = tf.constant(
            list(itertools.permutations(range(hparams.MAX_N_SIGNAL))),
            dtype=hparams.INTX)
        s_perms_onehot = tf.one_hot(
            v_perms, hparams.MAX_N_SIGNAL, dtype=hparams.FLOATX)

        s_x = tf.expand_dims(s_x, pit_axis+1)
        s_y = tf.expand_dims(s_y, pit_axis)
        if s_x.dtype.is_complex and s_y.dtype.is_complex:
            s_diff = s_x - s_y
            s_cross_loss = tf.reduce_mean(
                tf.square(tf.real(s_diff)) + tf.square(tf.imag(s_diff)),
                reduce_axes)
        else:
            s_cross_loss = tf.reduce_mean(
                tf.squared_difference(s_x, s_y), reduce_axes)
        s_loss_sets = tf.einsum(
            'bij,pij->bp', s_cross_loss, s_perms_onehot)
        s_loss_sets_idx = tf.argmin(s_loss_sets, axis=1)
        s_loss = tf.gather_nd(
            s_loss_sets,
            tf.stack([
                tf.range(hparams.BATCH_SIZE, dtype=tf.int64),
                s_loss_sets_idx], axis=1))
        s_loss = tf.reduce_mean(s_loss)
    return s_loss, v_perms, s_loss_sets_idx

