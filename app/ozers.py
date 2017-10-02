'''
Optimizers
'''
import tensorflow as tf

from app.hparams import hparams


@hparams.register_optimizer('sgd')
def sgd_ozer(learn_rate, lr_decay=None, lr_decay_epoch=2, **kwargs):
    kwargs.update(dict(learning_rate=learn_rate))
    return tf.train.GradientDescentOptimizer(**kwargs)


@hparams.register_optimizer('adam')
def adam_ozer(learn_rate, lr_decay=None, lr_decay_epoch=2, **kwargs):
    kwargs.update(dict(learning_rate=learn_rate))
    return tf.train.AdamOptimizer(**kwargs)
