import numpy as np
from itertools import product

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

