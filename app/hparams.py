'''
hyperparameters
'''
import numpy as np
import scipy.signal
import tensorflow as tf

# Hyperparameters are in CAPS
# TODO use tf.app.flags to parse hyperparam from input
#      or consider use json file to store hyperparams

# [--DATA TYPE--]
FLOATX = 'float32'       # default type for float
INTX = 'int32'           # defualt type for int

# [--PREPROCESSING--]
# WARNING, if you change anything under this category,
# please re-run data preprocessing script

# STFT segment size, stride and window function
FFT_SIZE = 256
FFT_STRIDE = 64
FFT_WND = np.sqrt(scipy.signal.hann(FFT_SIZE)).astype(FLOATX)
SMPRATE = 8000          # sampling rate

# [--DIMENSIONS--]
BATCH_SIZE = 32         # minibatch size
MAX_N_SIGNAL = 2        # speech sources to separate


LENGTH_ALIGN = 4        # zero pad spectra length multiples of this, useful for CNN
MAX_TRAIN_LEN = 128     # limit signal length during training, can be None
EMBED_SIZE = 20         # embedding size

# [--TRAINING--]
RELU_LEAKAGE = 0.3      # how leaky relu is, 0 -> relu, 1 -> linear
EPS = 1e-7              # to prevent sqrt() log() etc cause NaN
DROPOUT_KEEP_PROB = 1.  # probability to keep in dropout layer
REG_SCALE = 1e-2        # regularization loss scale
REG_TYPE = 'L2'         # regularization type, "L2", "L1" or "none"
LR = 3e-4               # learn rate
LR_DECAY = .8           # learn rate decaying, can be None

# "fixed" -> decay learn rate on each epoch
# "adaptive" -> only decay if validation or training error don't get better
LR_DECAY_TYPE = 'adaptive'
NUM_EPOCH_PER_LR_DECAY = 10

# clamp absolute gradient value within this value, None for no clip
GRAD_CLIP_THRES = 100.

# [--ARCHITECTURE--]
# TRAIN_ESTIMATOR_METHOD options:
# "truth"
# "truth-weighted"
# "truth-threshold"
# "anchor"
TRAIN_ESTIMATOR_METHOD = 'truth-weighted'
# TRAIN_ESTIMATOR_METHOD options:
# "anchor"
INFER_ESTIMATOR_METHOD = 'anchor'
NUM_ANCHOR = 6

# check "modules.py" to see available sub-modules
# ENCODER_TYPE options:
#   lstm-orig
#   bilstm-orig
#   conv-bilstm-v1
#   toy
ENCODER_TYPE = 'bilstm-orig'
# SEPARATOR_TYPE options:
#   dot-orig
SEPARATOR_TYPE = 'dot-sigmoid-orig'
# OPTIMIZER_TYPE options:
#   adam
#   sgd
OPTIMIZER_TYPE = 'adam'  # "sgd" or "adam"

# [--MISC--]
DATASET_TYPE = 'timit'  # "toy", "timit", or "wsj0"

SUMMARY_DIR = './logs'

# ==========================================================================
# normally you don't need touch anything below if you just want to tweak
# some hyperparameters

DEBUG = False
COMPLEXX = dict(float32='complex64', float64='complex128')[FLOATX]
FEATURE_SIZE = 1 + FFT_SIZE // 2

assert isinstance(DROPOUT_KEEP_PROB, float)
assert 0. < DROPOUT_KEEP_PROB <= 1.


# registry
encoder_registry = {}
estimator_registry = {}
separator_registry = {}
ozer_registry = {}
dataset_registry = {}


# decorators & getters
def register_encoder(name):
    def wrapper(cls):
        encoder_registry[name] = cls
        return cls
    return wrapper


def get_encoder():
    return encoder_registry[ENCODER_TYPE]


def register_estimator(name):
    def wrapper(cls):
        estimator_registry[name] = cls
        return cls
    return wrapper


def get_estimator(name):
    return estimator_registry[name]


def register_separator(name):
    def wrapper(cls):
        separator_registry[name] = cls
        return cls
    return wrapper


def get_separator(name):
    return separator_registry[name]


def register_optimizer(name):
    def wrapper(fn):
        ozer_registry[name] = fn
        return fn
    return wrapper


def get_optimizer():
    return ozer_registry[OPTIMIZER_TYPE]


def register_dataset(name):
    def wrapper(fn):
        dataset_registry[name] = fn
        return fn
    return wrapper


def get_dataset():
    return dataset_registry[DATASET_TYPE]


def get_regularizer():
    reger = dict(
        none=None,
        L1=tf.contrib.layers.l1_regularizer,
        L2=tf.contrib.layers.l2_regularizer)[REG_TYPE](REG_SCALE)
    return reger
