## NEIMS-PyTorch

A PyTorch implementation of the [NEIMS](https://github.com/brain-research/deep-molecular-massspec) model for efficient prediction of electron ionization mass spectra.

### Overview

The original NEIMS architecture was suggested in the following study:

_Wei, J. N., Belanger, D., Adams, R. P., & Sculley, D. (2019). Rapid Prediction of Electron–Ionization Mass Spectrometry Using Neural Networks. ACS Central Science, 5(4), 700–708. https://doi.org/10.1021/acscentsci.9b00085_

The original source code is avaliable on [GitHub](https://github.com/brain-research/deep-molecular-massspec). However, it uses outdated Python 3.6 and Tensorflow 1.13. Therefore, its installation is not always straightforward, it also does not support GPU acceleration, which can be much faster for processig larger datasets.

This repository contains an unofficial implementation of NEIMS with several tweaks using Python 3.12 and PyTorch 2.8 (latest stable combination available, can be upgraded in the future). The ReLU activation function was swapped for SiLU, also the number of hidden layers and neurons in these layers was optimized with optuna.

### Installation

The project requires **Python 3.12+**
Clone the repository:

```
git clone https://github.com/mkhrisanfov/neims-pytorch
```

Change folder to `neims-pytorch`:

```
cd neims-pytorch
```

Create a virtual environment (or install globally, for advanced users, skip to installing dependencies):

```
python -m venv .venv
```

**Activate virtual environment** for Linux:

```bash
source .venv/bin/activate
```

or **activate virtual environment** for Windows (cmd):

```cmd
.venv\Scripts\activate
```

or **activate virtual environment** for Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and `neims_pytorch` package from [pyproject.toml](./pyproject.toml):

```
pip install -e .
```

Load the weights from the huggingface ([mkhrisanfov/neims-pytorch](https://huggingface.co/mkhrisanfov/neims-pytorch)) and place them into the `models/` folder:

```
wget -P ./models https://huggingface.co/mkhrisanfov/neims-pytorch/resolve/main/NEIMSPyTorch.pth
```

### Inference

**Activate virtual environment (see above).**

Run the inference script with input file containing a list of smiles strings:

```
neims-predict input_file.txt smiles output_file.msp ./models/NEIMSPyTorch.pth
```

or a list of inchi strings:

```
neims-predict input_file.txt inchi output_file.msp ./models/NEIMSPyTorch.pth
```

The output will be an .MSP file with original identifiers (inchi or smiles), as well as InChI, SMILES, InChIKeys generated from RDKit molecules, and predicted EI mass spectra.

There are two optional arguments that can be used: batch size and `--use_cuda` flag. Full syntax:

```
neims-predict <input file name> <identifier type: inchi or smiles> <output file name> <model weights> <batch size, default is 64> <--use_cuda to use CUDA>
```

### Training and Optimization

The model is implemented to be trained using .MSP files (large mass spectral databases like MassBank, NIST are advised) with the fixed set of hyperparameters using a 80-20% training-validation split:

```
neims-train database.msp
```

The hyperparameters can be optimized the following way:

```
neims-optimize database.msp
```

The weights for each run will be placed into `models/` folder. Logs, including Tensorboard scalars for train and test at each epoch available at `logs/`. Optuna logs will be available at `logs/NEIMS.db`

### Troubleshooting

**If the commands above are not working** (virtual environment is not active or package was installed improperly). Replace the shortcut commands with the full ones, starting from `neims-pytorch/` folder.

For Linux:

- `neims-predict` -> `.venv/bin/python ./src/neims_pytorch/predict.py`
- `neims-train` -> `.venv/bin/python ./src/neims_pytorch/train.py`
- `neims-optimize` -> `.venv/bin/python ./src/neims_pytorch/optimize.py`

For example:

```
.venv/bin/python ./src/neims_pytorch/predict.py input_file.txt smiles output_file.msp ./models/NEIMSPyTorch.pth
```

For Windows:

- `neims-predict` -> `.venv\Scripts\python.exe src\neims_pytorch\predict.py`
- `neims-train` -> `.venv\Scripts\python.exe src\neims_pytorch\train.py`
- `neims-optimize` -> `.venv\Scripts\python.exe src\neims_pytorch\optimize.py`

For example:

```
.venv\Scripts\python.exe src\neims_pytorch\predict.py input_file.txt smiles output_file.msp models\NEIMSPyTorch.pth
```
