import os
import string
from sys import stdout
from six.moves import cPickle as pickle

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

# TODO add license from tensorpack/examples/CTC-TIMIT

# TODO merge these with py file
FLOATX = 'float32'
COMPLEXX = dict(float32='complex64', float64='complex128')[FLOATX]
FFT_SIZE = 256
FFT_STRIDE = 64
FFT_WND = np.sqrt(signal.hanning(FFT_SIZE))
SMPRATE = 8000
EPS = 1e-7

# "$" means end of text/phoneme
CHARSET = string.ascii_lowercase + ' '
PHONEME_LIST = (
    '$_aa_ae_ah_ao_aw_ax_ax-h_axr_ay_b_bcl_ch_d_dcl_dh_'
    'dx_eh_el_em_en_eng_epi_er_ey_f_g_gcl_h#_hh_hv_ih_'
    'ix_iy_jh_k_kcl_l_m_n_ng_nx_ow_oy_p_pau_pcl_q_r_'
    's_sh_t_tcl_th_uh_uw_ux_v_w_y_z_zh').split('_')

PHONEME_DIC = {v: k for k, v in enumerate(PHONEME_LIST)}
WORD_DIC = {v: k for k, v in enumerate(CHARSET)}

INTX = 'int32'
FLOATX = 'float32'

def load_wav_file(fname, smprate=16000):
    '''
    load a WAV file, then return a numpy float32 vector.
    Resample if needed.

    The returned array will always have lenght of multiples of FFT_SIZE
    to ease preprocessing, this is done via zero padding at the end.

    '''
    smprate_real, data = wavfile.read(fname)
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


def read_timit_txt(f):
    line = f.readlines()[0].strip().split(' ')[2:]
    line = ' '.join(line).replace('.', '').lower()
    line += '$'
    return np.asarray([WORD_DIC[c] for c in line if c in CHARSET], dtype=INTX)


def read_timit_phoneme(f):
    pho = []
    for line in f:
        line = line.strip().split(' ')[-1]
        pho.append(PHONEME_DIC[line])
    pho.append(PHONEME_LIST[0])
    return np.asarray(pho)


TRAIN_DIR = './train'
TEST_DIR = './test'
train_files = os.listdir('./train')
test_files = os.listdir('./test')

train_signals = []
train_phonemes = []
train_texts = []
os.chdir(TRAIN_DIR)
for fname in train_files:
    if not fname.endswith('.wav'):
        continue
    if fname.startswith('sa'):
        continue

    waveform = load_wav_file(fname, smprate=SMPRATE)
    Zxx = signal.stft(
        waveform,
        window=FFT_WND,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE-FFT_STRIDE)[2].astype(COMPLEXX).T
    with open(fname.upper().replace('.WAV', '.TXT'), 'r') as f:
        text = read_timit_txt(f)
    with open(fname.upper().replace('.WAV', '.PHN'), 'r') as f:
        phoneme = read_timit_phoneme(f)

    train_signals.append(Zxx)
    train_texts.append(text)
    train_phonemes.append(phoneme)

    stdout.write('.')
    stdout.flush()
os.chdir('../')

test_signals = []
test_phonemes = []
test_texts = []
os.chdir(TEST_DIR)
for fname in test_files:
    if not fname.endswith('.wav'):
        continue
    if fname.startswith('sa'):
        continue

    waveform = load_wav_file(fname, smprate=SMPRATE)
    Zxx = signal.stft(
        waveform,
        window=FFT_WND,
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE-FFT_STRIDE)[2].astype(COMPLEXX).T
    with open(fname.upper().replace('.WAV', '.TXT')) as f:
        text = read_timit_txt(f)

    with open(fname.upper().replace('.WAV', '.PHN')) as f:
        phoneme = read_timit_phoneme(f)

    test_signals.append(Zxx)
    test_texts.append(text)
    test_phonemes.append(phoneme)

    stdout.write('.')
    stdout.flush()
os.chdir('../')

# sort the whole batch by length, so minibatches
# need less zero padding -> higher performance
train_order = np.argsort(list(map(len, train_signals)))
test_order = np.argsort(list(map(len, test_signals)))

train_signals = [train_signals[i] for i in train_order]
test_signals = [test_signals[i] for i in test_order]
train_phonemes = [train_phonemes[i] for i in train_order]
test_phonemes = [test_phonemes[i] for i in test_order]
train_texts = [train_texts[i] for i in train_order]
test_texts = [test_texts[i] for i in test_order]

with open('train_set.pkl', 'wb') as f:
    pickle.dump(train_signals, f)
    pickle.dump(train_phonemes, f)
    pickle.dump(train_texts, f)

with open('test_set.pkl', 'wb') as f:
    pickle.dump(test_signals, f)
    pickle.dump(test_phonemes, f)
    pickle.dump(test_texts, f)

stdout.write('\nFinished preprocessing\n')
