# DPCL-Tensorflow
Tensorflow implementation of “Deep clustering: Discriminative embeddings for segmentation and separation”

[Link](https://arxiv.org/abs/1508.04306) to original paper

## Requirements

Same as DaNet model, see main `README.md` for details.

## Usage

### Setup dataset

Same as DaNet model, see main `README.md` for details.

### Setup hyperparameter

There is a `[--DPCL--]` section in `app/hparams.py`. It contains hyperparameters
specific to Deep Clustering model.

Otherwise, basic hyperparameters such as `BATCH_SIZE`, `LR` are shared between models.
See main `README.md` for details.

### Perform experiments

Run `dpcl.py` for Deep Clustering related experiments.
Arguments are identical to `main.py`, see main `README.md` for details.

### Use custom dataset

Same as DaNet model, see main `README.md` for details.

### Customize model

Deep Clustering shares “encoder” module with DaNet. It doesn’t use other modules.
See main `README.md` for more details.
