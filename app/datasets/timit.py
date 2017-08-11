import os
from six.moves import cPickle as pickle
import string
from functools import reduce
import gc

import numpy as np

import app.hparams as hparams
from app.datasets.dataset import Dataset

# TODO should we use pathlib to handle path?
#      (also means dropping python2 support)

@hparams.register_dataset('timit')
class TimitDataset(Dataset):
    '''TIMIT dataset'''
    CHARSET = string.ascii_lowercase + ' '
    PHONEME_LI = (
        'aa_ae_ah_ao_aw_ax_ax-h_axr_ay_b_bcl_ch_d_dcl_dh_'
        'dx_eh_el_em_en_eng_epi_er_ey_f_g_gcl_h#_hh_hv_ih_'
        'ix_iy_jh_k_kcl_l_m_n_ng_nx_ow_oy_p_pau_pcl_q_r_'
        's_sh_t_tcl_th_uh_uw_ux_v_w_y_z_zh').split('_')
    PHONEME_DI = {v: k for k, v in enumerate(PHONEME_LI)}
    WORD_DI = {v: k for k, v in enumerate(CHARSET)}
    def __init__(self):
        self.is_loaded = False

    def epoch(self, subset, batch_size, shuffle=False):
        # TODO add a validation set ?
        if subset not in self.subset:
            raise KeyError(
                'Unknown subset "%s", valid options are %s' %
                (subset, list(self.subset.keys())))
        signals_li, phonemes_li, texts_li = self.subset[subset]
        tot_size = len(signals_li)
        batch_size_ = batch_size
        assert tot_size == len(phonemes_li)
        assert tot_size == len(texts_li)
        if shuffle:
            idx_li = np.random.permutation(tot_size)
        else:
            idx_li = np.arange(tot_size)
        for i in range(0, tot_size-batch_size_, batch_size_):
            signals_batch_li = [signals_li[j] for j in idx_li[i:i+batch_size_]]
            texts_batch_li = [texts_li[j] for j in idx_li[i:i+batch_size_]]
            sig_len = max(map(len, signals_batch_li))
            txt_len = max(map(len, texts_batch_li))
            signals_batch = np.stack(
                [np.pad(s, ((0, sig_len-len(s)), (0, 0)), mode='constant')
                    for s in signals_batch_li])
            text_indices = np.empty(
                (reduce(int.__add__, map(len, texts_batch_li)), 2),
                dtype=hparams.INTX)
            text_values = np.concatenate(texts_batch_li)

            idx = 0
            for j, t in enumerate(texts_batch_li):
                l = len(t)
                text_indices[idx:idx+l, 0] = j
                text_indices[idx:idx+l, 1] = np.arange(l)
                idx += l

            text_shape = (batch_size_, txt_len)
            yield signals_batch, (text_indices, text_values, text_shape)

        if tot_size % batch_size_:
            signals_batch_li = [signals_li[j] for j in idx_li[-batch_size_:]]
            texts_batch_li = [texts_li[j] for j in idx_li[-batch_size_:]]
            sig_len = len(signals_li[-1])
            txt_len = max(map(len, texts_li[-batch_size_:]))
            signals_batch = np.stack(
                [np.pad(s, ((0, sig_len-len(s)), (0, 0)), mode='constant')
                    for s in signals_batch_li])
            text_indices = np.empty(
                (reduce(int.__add__, map(len, texts_batch_li)), 2), dtype=hparams.INTX)
            text_values = np.concatenate(texts_batch_li)

            idx = 0
            for i, t in enumerate(texts_batch_li):
                l = len(t)
                text_indices[idx:idx+l, 0] = i
                text_indices[idx:idx+l, 1] = np.arange(l)
                idx += l

            text_shape = (batch_size_, txt_len)
            yield signals_batch, (text_indices, text_values, text_shape)

    def install_and_load(self):
        # TODO automatically install if fails to find anything
        FILE_NOT_FOUND_MSG = (
            'Did not found TIMIT file "%s"'
            ', make sure you download and install the dataset')
        self.subset = {}
        path = os.path.join(os.path.dirname(__file__), 'TIMIT', '%s_set.pkl')
        for subset in ['train', 'test']:
            filepath = path % subset
            if not os.path.exists(filepath):
                raise IOError(
                    FILE_NOT_FOUND_MSG % filepath)

            with open(filepath, 'rb') as f:
                gc.disable()
                all_data = [pickle.load(f)]
                all_data.append(pickle.load(f))
                all_data.append(pickle.load(f))
                gc.enable()
            self.subset[subset] = all_data

        # use same subset for validation / test
        # as TIMIT is small
        self.subset['valid'] = self.subset['test']

    @classmethod
    def encode_from_str(cls, s):
        return np.asarray([cls.WORD_DI[c] for c in s], dtype='int32')

    @classmethod
    def decode_to_str(cls, arr):
        charset = cls.CHARSET + '$'
        s = ''.join(charset[i] for i in arr)
        return s.strip(' $')
