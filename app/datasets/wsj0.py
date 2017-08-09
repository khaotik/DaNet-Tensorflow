import os

import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme

import app.hparams as hparams
from app.datasets.dataset import Dataset


@hparams.register_dataset('wsj0')
class Wsj0Dataset(Dataset):
    '''WSJ0 dataset'''
    def __init__(self):
        self.is_loaded = False

    def __del__(self):
        if self.is_loaded:
            self.h5file.close()

    def install_and_load(self):
        path = os.path.join(
            os.path.dirname(__file__), 'WSJ0', 'wsj0-danet.hdf5')
        self.h5file = h5py.File(path, 'r')
        train_set = H5PYDataset(
            self.h5file, which_sets=('train',))
        valid_set = H5PYDataset(
            self.h5file, which_sets=('valid',))
        test_set = H5PYDataset(
            self.h5file, which_sets=('test',))
        self.subset = dict(
            train=train_set, valid=valid_set, test=test_set)
        self.is_loaded = True

    def epoch(self, subset, batch_size, shuffle=False):
        dataset = self.subset[subset]
        handle = dataset.open()
        dset_size = self.h5file.attrs['split'][
            dict(train=0, valid=1, test=2)[subset]][3]
        indices = np.arange(
            ((dset_size + batch_size - 1) // batch_size)*batch_size)
        indices %= dset_size
        if shuffle:
            np.random.shuffle(indices)
        req_itor = SequentialScheme(
            examples=indices, batch_size=batch_size).get_request_iterator()
        for req in req_itor:
            data_pt = dataset.get_data(handle, req)
            max_len = max(map(len, data_pt[0]))
            spectra_li = [np.pad(
                x, [(0, max_len - len(x)), (0, 0)],
                mode='constant') for x in data_pt[0]]
            spectra = np.stack(spectra_li)
            spectra = np.reshape(
                spectra, [
                    hparams.BATCH_SIZE,
                    hparams.MAX_N_SIGNAL,
                    max_len, hparams.FEATURE_SIZE])
            yield (spectra,)
        dataset.close(handle)
