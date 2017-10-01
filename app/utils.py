from random import randint
from math import ceil

import numpy as np
import scipy.io.wavfile
import scipy.signal

import app.hparams as hparams


def prompt_yesno(q_):
    while True:
        action = input(q_ + ' [Y]es [n]o : ')
        if action == 'Y':
            return True
        elif action == 'n':
            return False


def prompt_overwrite(filename_):
    '''
    If save file obj_.__getattribute__(attr_) exists, prompt user to
    give up, overwrite, or make copy.

    obj_.__getattribute__(attr_) should be a string, the string may be
    changed after prompt
    '''
    try:
        savfile = open(filename_, 'x')
    except FileExistsError:
        while True:
            action = input(
                'file %s exists, overwrite? [Y]es [n]o [c]opy : '%filename_)
            if action == 'Y':
                return filename_
            elif action == 'n':
                return ''
            elif action == 'c':
                i=0
                while True:
                    new_filename = filename_+'.'+str(i)
                    try:
                        savfile = open(new_filename, 'x')
                    except FileExistsError:
                        i+=1
                        continue
                    break
                return new_filename
    else:
        savfile.close()


def istft(X, stride, window):
    """
    Inverse short-time fourier transform.

    Args:
        X: complex matrix of shape (length, 1 + fft_size//2)

        stride: integer

        window: 1D array, should be (X.shape[1] - 1) * 2

    Returns:
        floating-point waveform samples (1D array)
    """
    fftsize = (X.shape[1] - 1) * 2
    x = np.zeros(X.shape[0]*stride)
    wsum = np.zeros(X.shape[0]*stride)
    for n, i in enumerate(range(0, len(x)-fftsize, stride)):
        x[i:i+fftsize] += np.real(np.fft.irfft(X[n])) * window   # overlap-add
        wsum[i:i+fftsize] += window ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x


def random_zeropad(X, padlen, axis=-1):
    '''
    This randomly do zero padding in both directions, on specified axis
    The sum of padding length equals to `padlen`
    '''
    if padlen == 0:
        return X
    l = randint(0, padlen)
    r = padlen - l

    ndim = X.ndim
    assert -ndim <= axis < ndim
    axis %= X.ndim
    pad = [(0,0)] * axis + [(l, r)] + [(0,0)] * (ndim-axis-1)
    return np.pad(X, pad, mode='constant')


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
    data = istft(
        feature, stride=hparams.FFT_STRIDE, window=hparams.FFT_WND)
    scipy.io.wavfile.write(filename, hparams.SMPRATE, data)


