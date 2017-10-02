'''
Hyperparameters
'''
import re
import json

import numpy as np
import scipy.signal
import tensorflow as tf

# Hyperparameters are in CAPS
# TODO use tf.app.flags to parse hyperparams from input
#      or consider use json file to store hyperparams

class Hyperparameter:
    '''
    Contains hyperparameter settings
    '''
    pattern = r'[A-Z_]+'
    encoder_registry = {}
    estimator_registry = {}
    separator_registry = {}
    ozer_registry = {}
    dataset_registry = {}

    def __init__(self):
        pass

    def digest(self):
        '''
        When hyperparameters are updated, this function should be called.

        This performs asserts, and derive some inferred hyperparams.
        '''
        self.COMPLEXX = dict(
            float32='complex64', float64='complex128')[self.FLOATX]
        self.FEATURE_SIZE = 1 + self.FFT_SIZE // 2
        assert isinstance(self.DROPOUT_KEEP_PROB, float)
        assert 0. < self.DROPOUT_KEEP_PROB <= 1.

        # FIXME: security concern by using eval?
        self.FFT_WND = eval(self.FFT_WND)

    def load(self, di):
        '''
        load from a dict

        Args:
            di: dict, string -> string
        '''
        assert isinstance(di, dict)
        pat = re.compile(self.pattern)
        for k,v in di.items():
            if None is pat.fullmatch(k):
                raise NameError
            assert isinstance(v, (str, int, float, bool, type(None)))
        self.__dict__.update(di)

    def load_json(self, file_):
        '''
        load from JSON file

        Args:
            file_: string or file-like
        '''
        if isinstance(file_, (str, bytes)):
            file_ = open(file_, 'r')
        di = json.load(file_)
        self.load(di)

    # decorators & getters
    @classmethod
    def register_encoder(cls_, name):
        def wrapper(cls):
            cls_.encoder_registry[name] = cls
            return cls
        return wrapper

    def get_encoder(self):
        return type(self).encoder_registry[self.ENCODER_TYPE]

    @classmethod
    def register_estimator(cls_, name):
        def wrapper(cls):
            cls_.estimator_registry[name] = cls
            return cls
        return wrapper

    def get_estimator(self, name):
        return type(self).estimator_registry[name]

    @classmethod
    def register_separator(cls_, name):
        def wrapper(cls):
            cls_.separator_registry[name] = cls
            return cls
        return wrapper

    def get_separator(self, name):
        return type(self).separator_registry[name]

    @classmethod
    def register_optimizer(cls_, name):
        def wrapper(fn):
            cls_.ozer_registry[name] = fn
            return fn
        return wrapper

    def get_optimizer(self):
        return type(self).ozer_registry[self.OPTIMIZER_TYPE]

    @classmethod
    def register_dataset(cls_, name):
        def wrapper(fn):
            cls_.dataset_registry[name] = fn
            return fn
        return wrapper

    def get_dataset(self):
        return type(self).dataset_registry[self.DATASET_TYPE]

    def get_regularizer(self):
        reger = {
            None: (lambda _:None),
            'L1':tf.contrib.layers.l1_regularizer,
            'L2':tf.contrib.layers.l2_regularizer}[self.REG_TYPE](self.REG_SCALE)
        return reger


hparams = Hyperparameter()

# old, obsolete code
# REMOVE when merging PR
"""
# [--DATA TYPE--]
FLOATX = 'float32'       # default type for float
INTX = 'int32'           # default type for int

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
REG_TYPE = 'L2'         # regularization type, "L2", "L1" or None
LR = 3e-4               # learn rate
LR_DECAY = .8           # learn rate decaying, can be None

# "fixed" -> decay learn rate on each epoch
# "adaptive" -> only decay if validation or training error don't get better
# None -> don't decay learning rate
LR_DECAY_TYPE = None
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
ENCODER_TYPE = 'toy'
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

# registry
encoder_registry = {}
estimator_registry = {}
separator_registry = {}
ozer_registry = {}
dataset_registry = {}
"""

