# MACE_Multi_GPU

An implementation of the multi-GPU version for MACE.

- MACE - Fast and accurate machine learning interatomic potentials with higher order equivariant message passing.

## Table of contents

- [MACE\_Multi\_GPU](#mace_multi_gpu)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
    - [MACE installation from source](#mace-installation-from-source(multi_gpu-version))
    - [Metis installation](#metis-installation)
  - [Usage of Pretrained Foundation Models](#usage-of-pretrained-foundation-models)
    - [MACE-MP: Materials Project Force Fields](#mace-mp-materials-project-force-fields)
    - [MACE-OFF: Transferable Organic Force Fields](#mace-off-transferable-organic-force-fields)

## Installation

### MACE installation from source(multi_gpu version)

- conda

```shell
# Create a virtual environment and activate it
conda create --name mace_multi_gpu
conda activate mace_multi_gpu

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# (optional) Install MACE's dependencies from Conda as well
conda install numpy scipy matplotlib ase opt_einsum prettytable pandas e3nn

# Clone and install multi-GPU version of MACE (and all required packages)
git clone git@github.com:zEuler1027/mace_multi_gpu.git
pip install .
```

- pip

```shell
python -m venv mace mace_multi_gpu
source mace-multi_gpu/bin/activate

# Install PyTorch (for example, for CUDA 11.6 [cu116])
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Clone and install MACE (and all required packages)
git clone git@github.com:zEuler1027/mace_multi_gpu.git
pip install .
```

### Metis installation

Metis C++ dependency for Graphpartition and GraphDataParallel.

Run the following code in Python to easily install the metis library.

```python
import requests
import tarfile

# Download and extract the file
url = "http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz"
response = requests.get(url, stream=True)
file = tarfile.open(fileobj=response.raw, mode="r|gz")
file.extractall(path=".")

# Change working directory
%cd metis-5.1.0

# build metis dependency
!make config shared=1 prefix=~/.local/
!make install
!cp ~/.local/lib/libmetis.so /usr/lib/libmetis.so
!export METIS_DLL=/usr/lib/libmetis.so
!pip3 install metis-python

import metispy as metis # check metispy
```

## Usage of Pretrained Foundation Models

### MACE-MP: Materials Project Force Fields

- MACECalculator

```python
from mace.calculators import mace_mp
from ase import build

atoms = build.molecule('CH3CH2OH')
calc = mace_mp(model="medium", device='cuda')
atoms.calc = calc
print(atoms.get_potential_energy())
```
- MACEDataParallelCalculator

```python
from mace.calculators import mace_mp
from ase import build

atoms = build.molecule('CH3CH2OH')
calc = mace_mp(model="medium", mode='dp')
atoms.calc = calc
print(atoms.get_potential_energy())
```

### MACE-OFF: Transferable Organic Force Fields

- MACECalculator

```python
from mace.calculators import mace_off
from ase import build

atoms = build.molecule('CH3CH2OH')
calc = mace_off(model="medium", device='cuda')
atoms.calc = calc
print(atoms.get_potential_energy())
```
- MACEDataParallelCalculator

```python
from mace.calculators import mace_off
from ase import build

atoms = build.molecule('CH3CH2OH')
calc = mace_off(model="medium", mode='dp')
atoms.calc = calc
print(atoms.get_potential_energy())
```
