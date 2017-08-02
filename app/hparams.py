'''
hyperparameters
'''

# Hyperparameters are in CAPS
# TODO use tf.app.flags to parse hyperparam from input
#      or consider use json file to store hyperparams
BATCH_SIZE = 8  # minibatch size
MAX_N_SIGNAL = 3
FFT_SIZE = 256  # segmenet size in STFT
FFT_STRIDE = 64  # segmenet stride in STFT
EMBED_SIZE = 20  # embedding size

FLOATX = 'float32'  # default type for float
INTX = 'int32'  # defualt type for int

RELU_LEAKAGE = 0.3  # how leaky relu is, 0 -> relu, 1 -> linear
EPS = 1e-7  # to prevent sqrt() log() etc cause NaN
DROPOUT_KEEP_PROB = 0.8  # probability to keep in dropout layer
assert isinstance(DROPOUT_KEEP_PROB, float)
assert 0. < DROPOUT_KEEP_PROB <= 1.
REG_SCALE = 1e-2  # regularization loss scale
REG_TYPE = 'L2'  # regularization type, "L2", "L1" or "none"

# "truth", "k-means", "fixed" or "anchor"
TRAIN_ATTRACTOR_METHOD = 'anchor'
# "k-means", "fixed", "anchor"
INFER_ATTRACTOR_METHOD = 'anchor'
NUM_ANCHOR = 4

# check "modules.py" to see available sub-modules
ENCODER_TYPE = 'bilstm-orig'
OPTIMIZER_TYPE = 'adam'  # "sgd" or "adam"
LR = 3e-4  # learn rate
LR_DECAY = None

DATASET_TYPE = 'toy'  # "toy", "timit", or "wsj0"

SUMMARY_DIR = './logs'

# ==========================================================================
# normally you don't need touch anything below if you just want to tweak
# some hyperparameters

COMPLEXX = dict(float32='complex64', float64='complex128')[FLOATX]
FEATURE_SIZE = 1 + FFT_SIZE // 2

import tensorflow as tf

# registry
encoder_registry = {}
estimator_registry = {}
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
