# DaNet-Tensorflow
Tensorflow implementation of "Speaker-Independent Speech Separation with Deep Attractor Network"

[Link](https://arxiv.org/abs/1707.03634) to original paper

### 2021 Note: I am NOT the original author of paper. This code runs but won't learn well. I've got no time to work on this. If you managed to get the models working, let me know.

**STILL WORK IN PROGRESS, EXPECT BUGS**

## Requirements

numpy / scipy

tensorflow >= 1.2

matplotlib (optional, for visualization)

h5py / [fuel](https://github.com/mila-udem/fuel) (optional, for certain datasets)

## Usage

### Prepare datasets

Currently, TIMIT and WSJ0 datasets are implemented.
You can use the "toy" dataset for debugging. It just some white noise.

- TIMIT dataset

Follow `app/datasets/TIMIT/readme` for dataset preparation.

- WSJ0 dataset

Follow `app/datasets/WSJ0/readme` for dataset preparation.

**After setting up a dataset, you may want to change** `DATASET_TYPE` in hyperparameters.

### Setup hyperparameters

This is to change batch size, learning rate, dataset type etc ...

- The recommended way: using JSON file

There's a `default.json` file at the root directory. You make your own and change
*some* of the values. For example you can create a JSON file with:

```json
{
    DATASET_TYPE="timit",
    LR=1e-2,
    BATCH_SIZE=8
}
```

Save it as `my_setup.json`, now you can run the script with:

```bash
python main.py -c my_setup.json
```

- The direct way: using command line arguments

Some commonly used hyperparameters can be overridden by CLI args.

For example, to set learning rate:

```bash
python main.py -lr=1e-2
```

Here's a incomplete list of them:

```
# set learning rate, overrides LR
-lr
--learn-rate

# set dataset to use, overrides DATASET_TYPE
-ds
--dataset

# set batch size, overrides 
-bs
--batch-size

# set
```

**Note** If you get out of memory (OOM) error from tensorflow, you can try using a lower `BATCH_SIZE`.

**Note** If you change `FFT_SIZE`, `FFT_STRIDE`, `FFT_WND`, `SMP_RATE`,
you should do dataset preprocessing again.

**Note** If you change model architecture, the previously saved model parameter may not be compatible.

### Perform experiments

Under the root directory of this repo:

- train a model for 10 epoch and see accuracy, using TIMIT dataset

```bash
    python main.py -ds='timit'
```


- train a model using your own hyperparameters

```bash
    python main.py -c my_setup.json
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

 - To use your dataset, set `DATASET_TYPE` to `"my_dataset"` in JSON config file


### Customize model

You can make subclass of `Estimator`, `Encoder`, or `Separator` to tweak model.

- `Encoder` is for getting embedding from log-magnitude spectra.

- `Estimator` is for estimating attractor points from embedding.

- `Separator` uses mixture spectra, mixture embedding and attractor to get separated spectra.


You can set encoder type by setting `ENCODER_TYPE` in hyperparameters.

You can set estimator type by setting
`TRAIN_ESTIMATOR_METHOD` and `INFER_ESTIMATOR_METHOD` in hyperparameters.

You can set separator type by setting `SEPARATOR_TYPE` in hyperparameters.


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
