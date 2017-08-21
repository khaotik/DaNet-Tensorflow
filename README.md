# DaNet-Tensorflow
Tensorflow implementation of "Speaker-Independent Speech Separation with Deep Attractor Network"

[Link](https://arxiv.org/abs/1707.03634) to original paper

### Deep clustering
This codebase also contains an implementation for Deep Clustering model.
Details are inside `README_DPCL.md`.

**STILL WORK IN PROGRESS, EXPECT BUGS**

## Requirements

### General

numpy / scipy

tensorflow >= 1.2

matplotlib (optional, for visualization)

### TIMIT dataset

You need a utility program `sndfile-convert`

On Ubuntu, this can be installed as:

`apt-get install sndfile-programs`

The source code is also available at [here](https://github.com/erikd/libsndfile)


You should follow `app/datasets/TIMIT/readme` for dataset preparation.

### WSJ0 dataset

h5py / [fuel](https://github.com/mila-udem/fuel)

If you can't connect to Internet, you need to prepare `sph2pipe` utility under `app/datasets/WSJ0`.
It's available for download [here](http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz)

With internet connection, the script automatically will download it for you.


You should follow `app/datasets/WSJ0/readme` for dataset preparation.

## Usage

### Setup dataset

Currently, TIMIT and WSJ0 datasets are implemented.
You can use the "toy" dataset for debugging. It just some white noise.


Check **Requirements** section for detail.

### Setup hyperparameter

Before performing any experiment, you should set hyperparameters in `app/hparams.py`

For example, you can setup batch size, learn rate, dataset type ...

Most of settings are self explanatory, or commented in code.


**Note** If you get out of memory (OOM) error from tensorflow, you can try using a lower `BATCH_SIZE`.

**Note** If you change `FFT_SIZE`, `FFT_STRIDE`, `FFT_WND`, `SMP_RATE`,
you should do dataset preprocessing again.

**Note** If you change model architecture, the previously saved model parameter may not be compatible.

### Perform experiments

Under the root dirctory of this repo:

- train a model for 10 epoch and see accuracy

```bash
    python main.py
```


- train a model for 100 epoch and save it

```bash
    python main.py -ne=100 -o='params.ckpt'
```


- continue from last saved model, train 100 more epoch, save back

```bash
    python main.py -ne=100 -i='params.ckpt' -o='params.ckpt'
```


- test the trained model on test set

```bash
    python main.py -i='params.ckpt' -m=test
```


- draw a sample from test set, then separate it:

```
    $ python main.py -i='params.ckpt' -m=demo
    $ ls *.wav
    demo.wav demo_separated_1.wav demo_separated_2.wav
```


- separate a given WAV file:

```
    $ python main.py -i='params.cpkt' -m=demo -if=file.wav
    $ ls *.wav
    file.wav file_separated_1.wav file_separated_2.wav
```


- launch tensorboard and see graphs

```bash
    tensorboard --logdir=./logs/`
```


- for more CLI arguments, do

```bash
    python main.py --help
```


### Use custom dataset

 - Make a file `app/datasets/my_dataset.py`.

 - Make a subclass of `app.datasets.dataset.Dataset`

```python
    @hparams.register_dataset('my_dataset')
    class MyDataset(Dataset):
        ...
```

You can use `app/datasets/timit.py` as an reference.

 - In `app/datasets/__init__.py`, add:

```python
    import app.datasets.my_dataset
```

 - To use your dataset, set `DATASET_TYPE='my_dataset'` in `app/hparams.py`


### Customize model

You can make subclass of `Estimator`, `Encoder`, or `Separator` to tweak model.

- `Encoder` is for getting embedding from log-magnitude spectra.

- `Estimator` is for estimating attractor points from embedding.

- `Separator` uses mixture spectra, mixture embedding and attractor to get separated spectra.


You can set encoder type by setting `ENCODER_TYPE` in `hparams.py`

You can set estimator type by setting
`TRAIN_ESTIMATOR_METHOD` and `INFER_ESTIMATOR_METHOD` in `hparams.py`

You can set separator type by setting `SEPARATOR_TYPE` in `hparams.py`


Make sure to use `@register_*` decorator for your class.
See code in `app/modules.py` for details. There are existing sub-modules.

To change overall model architecture, modify `Model.build()` in `main.py`


## Limitations

- Only the favorable `"anchor"` method for estimating attractor location during inference is implemented.
  During training, it's also possible to use ground truth to give attractor location.

- TIMIT dataset is small, so we use same set for test and validation.

- We use WSJ0 `si_tr_s` / `si_dt_05` / `si_et_05` subsets as training / validation / test set respectively.
  The speakers are randomly chosen and mixed at runtime.

  This setup is slightly different to orignal paper.

- Only single GPU training is implemented.

- Doesn't work on Windows.
