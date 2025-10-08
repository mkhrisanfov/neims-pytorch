## NEIMS-PyTorch

A PyTorch implementation of the [NEIMS](https://github.com/brain-research/deep-molecular-massspec) model for efficient prediction of electron ionization mass spectra.

### Overview

The original NEIMS architecture was suggested in the following study:

*Wei, J. N., Belanger, D., Adams, R. P., & Sculley, D. (2019). Rapid Prediction of Electron–Ionization Mass Spectrometry Using Neural Networks. ACS Central Science, 5(4), 700–708. https://doi.org/10.1021/acscentsci.9b00085*

The original source code is avaliable on [GitHub](https://github.com/brain-research/deep-molecular-massspec). However, it uses outdated Python 3.6 and Tensorflow 1.13. Therefore, its installation is not always straightforward, it also does not support GPU acceleration, which can be much faster for processig larger datasets.

This repository contains an unofficial implementation of NEIMS with several tweaks using Python 3.12 and PyTorch 2.8 (latest available stable combination, can be upgraded in the future). The ReLU activation function was swapped for SiLU, also the number of hidden layers and neurons in these layers was optimized with optuna.

### Installation

Install [uv](github.com/astral-sh/uv).
Clone the repository:

```
git clone https://github.com/mkhrisanfov/neims-pytorch
```

Install [dependencies](./pyproject.toml) and project using uv:

```
uv sync
```

Alternatively, dependencies can be installed using [requirements.txt](./requirements.txt) file (will install CPU-only version of pytorch):

```
pip install -r requirements.txt
```

CUDA-compatible pytorch versions can be installed using official [PyTorch](https://pytorch.org/) instructions.

### Inference

Install the project with its dependencies.
Load the weights from the huggingface ([mkhrisanfov/neims-pytorch](https://huggingface.co/mkhrisanfov/neims-pytorch)) and place them into the `models/` folder:

```
wget -P ./models https://huggingface.co/mkhrisanfov/neims-pytorch/resolve/main/NEIMSPyTorch.pth
```

Run the inference script with input file containing a list of smiles strings:

```
python ./src/neims_pytorch/predict.py input_file.txt smiles output_file.msp ./models/NEIMSPyTorch.pth
```

or a list of inchi strings:

```
python ./src/neims_pytorch/predict.py input_file.txt inchi output_file.msp ./models/NEIMSPyTorch.pth
```

The output will be an .MSP file with original identifiers (inchi or smiles), as well as InChI, SMILES, InChIKeys generated from RDKit molecules, and predicted EI mass spectra.

There are two optional arguments that can be used: batch size and `--use_cuda` flag. Full example:

```
python ./src/neims_pytorch/predict.py input_file.txt inchi output_file.msp ./models/NEIMSPyTorch.pth 512 --use_cuda
```

Explanation

```
python ./src/neims_pytorch/predict.py <input file name> <identifier type: inchi or smiles> <output file name> <model weights> <batch size, default is 64> <--use_cuda to use CUDA>
```

### Training and Optimization

The model is implemented to be trained using .MSP files (large mass spectral databases are advised) with the fixed set of hyperparameters using a 80-20% training-validation split:

```
python  ./src/neims_pytorch/train.py database.msp
```

The hyperparameters can be optimized the following way:

```
python  ./src/neims_pytorch/optimize.py database.msp
```

The weights for each run will be placed into `models/` folder. Logs, including Tensorboard scalars for train and test at each epoch available at `logs/`. Optuna logs will be available at `logs/NEIMS.db`
