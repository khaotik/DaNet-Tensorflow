from __future__ import division, print_function
from math import ceil, sqrt
import os
import random
from sys import stdout, stderr

import numpy as np
import scipy.signal
import scipy.io.wavfile
import h5py

FFT_SIZE = 256
FFT_WND = np.sqrt(scipy.signal.hann(FFT_SIZE))
SMPRATE = 8000
FLOATX = 'float32'
MAX_MIX_SNR = 5.  # max mixture SNR in Db
SEED = 1337
DEBUG = False
FILENAME = 'wsj0-danet.hdf5'

g_gender_di = {
    '001': 'M', '002': 'F', '00a': 'F', '00b': 'M',
    '00c': 'M', '00d': 'M', '00f': 'F', '203': 'F',
    '400': 'M', '430': 'F', '431': 'M', '432': 'F',
    '440': 'M', '441': 'F', '442': 'M', '443': 'M',
    '444': 'F', '445': 'F', '446': 'M', '447': 'M'}

COMPLEXX = dict(float32='complex64', float64='complex128')[FLOATX]
assert FFT_SIZE % 4 == 0

def is_female(filename):
    global g_gender_di
    return g_gender_di[
        os.path.basename(filename)[:3]] == 'F'

def load_file(fname, smprate=16000):
    '''
    load a NIST SPHERE file from WSJ0 dataset, then return
    a numpy float32 vector. Resample if needed.

    The returned array will always have lenght of multiples of FFT_SIZE
    to ease preprocessing, this is done via zero padding at the end.

    '''
    res = os.system(
        './sph2pipe -f rif %s speech.wav' % fname)
    if res:
        raise RuntimeError('File is corrupt')
    smprate_real, data = scipy.io.wavfile.read('speech.wav')
    if smprate_real == smprate:
        data = data.astype(FLOATX)
    elif (smprate_real % smprate) == 0:
        # integer factor downsample
        smpfactor = smprate_real // smprate
        data = np.pad(
            data, [(0, (-len(data)) % smpfactor)], mode='constant')
        data = np.reshape(data, [len(data)//smpfactor, smpfactor])
        data = np.mean(data.astype(FLOATX), axis=1)
    else:
        newlen = int(ceil(len(data) * (smprate / smprate_real)))
        # FIXME this resample is very slow on prime length
        data = scipy.signal.resample(data, newlen).astype(FLOATX)
    return data


def gen_2spkr_mixture(names_li_):
    '''
    generates a two-speaker mixture

    Args:
        names_li_: list of filenames

    Returns:
        (spectra, time)

        spectra is complex valued matrix of shape (length, 1+fft_size//2)
        time is float type, in seconds
    '''
    u, v = random.randint(0, len(names_li_)-1), random.randint(0, len(names_li_)-1)
    while u == v:
        u, v = random.randint(0, len(names_li_)-1), random.randint(0, len(names_li_)-1)
    u_wav = load_file(names_li_[u], smprate=SMPRATE)
    v_wav = load_file(names_li_[v], smprate=SMPRATE)

    u_spectra = scipy.signal.stft(
        u_wav,
        nperseg=FFT_SIZE,
        noverlap=3*(FFT_SIZE//4))[2].T
    v_spectra = scipy.signal.stft(
        v_wav,
        nperseg=FFT_SIZE,
        noverlap=3*(FFT_SIZE//4))[2].T
    ulen = len(u_spectra)
    vlen = len(v_spectra)
    if ulen > vlen:
        n_pad = random.randint(0, ulen-vlen)
        v_spectra = np.pad(
            v_spectra,
            [(n_pad, ulen-vlen-n_pad), (0,0)],
            mode='constant')
    elif vlen > ulen:
        n_pad = random.randint(0, ulen-vlen)
        u_spectra = np.pad(
            u_spectra,
            [(n_pad, ulen-vlen-n_pad), (0,0)],
            mode='constant')
    u_pwr, v_pwr = np.mean(u_wav * u_wav), np.mean(v_wav * v_wav)
    avg_pwr = sqrt(u_pwr * v_pwr)

    u_coeff = 0.05 * random.uniform(-MAX_MIX_SNR, MAX_MIX_SNR)
    v_coeff = - u_coeff
    u_coeff, v_coeff = 10 ** u_coeff, 10 ** v_coeff
    u_coeff *= (u_pwr / avg_pwr); v_coeff *= (v_pwr / avg_pwr)

    mix_spectra = u_coeff * u_spectra, v_coeff * v_spectra
    t = max(ulen, vlen) * FFT_SIZE / SMPRATE
    return mix_spectra, t

stdout.write('Getting file names ...'); stdout.flush()
with open('train_set_files', 'r') as f:
    train_names_li = f.readlines()
with open('valid_set_files', 'r') as f:
    valid_names_li = f.readlines()
with open('test_set_files', 'r') as f:
    test_names_li = f.readlines()

train_names_li = list(sorted(map(lambda _: _[:-1], train_names_li)))
valid_names_li = list(sorted(map(lambda _: _[:-1], valid_names_li)))
test_names_li = list(sorted(map(lambda _: _[:-1], test_names_li)))
# FIXME this setting is not the same as original paper
stdout.write(' done\n'); stdout.flush()

random.seed(SEED)
np.random.seed(SEED)
dataset_file = h5py.File(FILENAME, mode='w')
data_t = h5py.special_dtype(vlen=np.dtype(COMPLEXX))

def add_subset(name, names_li):
    stdout.write('Generating subset "%s" ...' % name); stdout.flush()
    dataset = dataset_file.create_dataset(
        '%s_spectra' % name, (len(names_li),), dtype=data_t)
    dataset_shapes = dataset_file.create_dataset(
        '%s_spectra_shapes' % name, (len(names_li), 2), dtype=np.int32)
    dataset_slabels = dataset_file.create_dataset(
        '%s_spectra_shape_labels' % name, (2,), dtype='S8')
    dataset_slabels[...] = [
        'length'.encode('utf8'),
        'fft_size'.encode('utf8')]
    err_cnt = 0
    for i, fname in enumerate(names_li):
        if i>5 and DEBUG:
            break

        try:
            wav = load_file(fname, SMPRATE)
        except:
            err_cnt += 1
            if err_cnt > 100:
                stderr.write(
                    'Too many file reading failure, abort.'
                    ' Raising lastest exception:\n')
                raise
            continue

        spectra = scipy.signal.stft(
            wav.astype('float32'),
            window=FFT_WND,
            nperseg=FFT_SIZE,
            noverlap=(FFT_SIZE*3)//4)[2].T.astype(COMPLEXX)
        dataset[i - err_cnt] = spectra.flat
        dataset_shapes[i - err_cnt] = np.array(
            [len(spectra), 1+FFT_SIZE//2], dtype=np.int32)
        stdout.write('.'); stdout.flush()
    dataset.dims.create_scale(dataset_shapes, 'shapes')
    dataset.dims.create_scale(dataset_slabels, 'shape_labels')
    dataset.dims[0].attach_scale(dataset_shapes)
    dataset.dims[0].attach_scale(dataset_slabels)
    dataset_size = len(names_li) - err_cnt
    stdout.write(' done\n'); stdout.flush()
    return dataset_size


train_size = add_subset('train', train_names_li)
valid_size = add_subset('valid', valid_names_li)
test_size = add_subset('test', test_names_li)

split_array = np.empty(
    3, dtype=np.dtype([
        ('split', 'a', 5),
        ('source', 'a', 15),
        ('start', np.int64),
        ('stop', np.int64),
        ('indices', h5py.special_dtype(ref=h5py.Reference)),
        ('available', np.bool, 1),
        ('comment', 'a', 1)]))

split_array[0]['split'] = 'train'.encode('utf8')
split_array[1]['split'] = 'valid'.encode('utf8')
split_array[2]['split'] = 'test'.encode('utf8')

split_array[0]['source'] = 'train_spectra'.encode('utf8')
split_array[1]['source'] = 'valid_spectra'.encode('utf8')
split_array[2]['source'] = 'test_spectra'.encode('utf8')
split_array[:]['start'] = 0
split_array[0]['stop'] = train_size
split_array[1]['stop'] = valid_size
split_array[2]['stop'] = test_size
split_array[:]['indices'] = h5py.Reference()
split_array[:]['available'] = True
split_array[:]['comment'] = '.'.encode('utf8')

dataset_file.attrs['split'] = split_array
dataset_file.close()
