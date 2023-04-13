[![PyPI](https://img.shields.io/pypi/v/pyroundtrip)](https://pypi.org/project/pyroundtrip/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4560982.svg)](https://doi.org/10.5281/zenodo.3747161)


# Tutorials: Roundtrip density estimation with deep generative neural networks

<a href='https://github.com/kimmo1019/Roundtrip'><img align="left" src="https://github.com/kimmo1019/Roundtrip/blob/master/model.jpg" width="350">

Roundtrip is a deep generative neural density estimator which exploits the advantage of GANs for generating samples and estimates density by either importance sampling or Laplace approximation. 

Since TensorFlow v2 is not compatible with TensorFlow v1, the Roundtrip model was reimplemented with Python3 and TensorFlow v2. This tutorial provides instructions for using Roundtrip with Python3 and TensorFlow v2. Now Roundtrip is available on [PyPI](https://pypi.org/project/pyroundtrip/).
  
## Installation

### Prerequisites

Roundtrip can be installed via [PyPI](https://pypi.org/project/pyroundtrip/) with `pip`. 

### pip prerequisites

1. Install [Python]. we recommend Python>=3.6 and the [venv](https://docs.python.org/3/library/venv.html) or [pyenv](https://github.com/pyenv/pyenv/) for creating a virtual environment and version management system.

2. Take venv for instance. Create a virtual environment:

    ```shell
    python3 -m venv <venv_path>
    ```

3. Activate the virtual environment:

    ```shell
    source <venv_path>/bin/activate
    ```


### GPU prerequisites (optional)

Training Roundtrip model will be faster when accelerated with a GPU (not a must). Before installing Roundtrip, the CUDA and cuDNN environment should be setup.


### Install with pip

Install Roundtrip from PyPI using:

    ```
    pip install pyroundtrip
    ```

If you get a `Permission denied` error, use `pip install pyroundtrip --user` instead. Pip will automatically install all the dependent packages, such as TensorFlow.

Alteratively, pyroundtrip can also be installed through GitHub using::

    ``` 
    git clone https://github.com/kimmo1019/Roundtrip && cd Roundtrip/src
    pip install -e .
    ```

``-e`` is short for ``--editable`` and links the package to the original cloned
location such that pulled changes are also reflected in the environment.

## Results reproduction
  
Under the `src` folder, the `main.py` script is provided for reproducing the density estimation results in the paper.

We provide configuration files for different datasets in `configs` folder. One can run the following commond to reproduce the results.

```
python3 main.py -c configs/config_[DATASET].yaml
```

`DATASET` can be `indep_gmm`, `involute`, `mnist`, and `cifar10`.

## Command line

After installing Roundtrip through pip, one can get the usage of this command line by

```
roundtrip -h
```

This command line takes `npy`,`cvs`, or `txt` as input (nb_samples x nb_feats), one can run the following command to perform Roundtrip density estimation.

```
roundtrip -input data.npy --output_dir ./ -z_dim 2 -sd_x 0.5 -scale 0.5 
```
The results will be saved under the `output_dir` folder.

## Use Python API

We provide a tutorial notebook for implmentating the Roundtrip density estimation with Python API.




