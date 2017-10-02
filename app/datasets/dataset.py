from itertools import product

import numpy as np

from app.hparams import hparams


class Dataset(object):
    def __init__(self):
        self.is_loaded = False

    def epoch(self, subset, batch_size, shuffle=False):
        '''
        Iterator, yields batches of numpy array
        Args:
            subset: string
            batch_size: int
            shuffle: bool

        Yields:
            (signals,)
            signals is a rank-3 float32 array: [batch_size, time, features]
        '''
        raise NotImplementedError()

    def install_and_load(self):
        '''
        Download and preprocess dataset and store it on local disk.
        This method should check whether data is available in ./datasets,
        or already downloaded in ./downloads

        Raises:
            raise RuntimeError is something fails
        '''
        raise NotImplementedError()

    def encode_from_str(arr):
        raise NotImplementedError()

    def decode_to_str(arr):
        raise NotImplementedError()

@hparams.register_dataset('toy')
class WhiteNoiseData(object):
    '''
    this always generates uniform noise
    '''
    # make it more general, support more shape
    def __init__(self):
        self.is_loaded = False

    def epoch(self, subset, batch_size, shuffle=False):
        if not self.is_loaded:
            raise RuntimeError('Dataset is not loaded.')
        for _ in range(10):
            signal = np.random.rand(
                batch_size,
                128, hparams.FEATURE_SIZE).astype(hparams.FLOATX)
            yield (signal,)

    def install_and_load(self):
        self.is_loaded = True
        return
