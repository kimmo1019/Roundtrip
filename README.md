# Roundtrip

![model](https://github.com/kimmo1019/Rountrip/blob/master/model.jpg)

Roundtrip is a deep generative neural density estimator which exploits the advantage of GANs for generating samples and estimates density by either importance sampling or Laplace approximation. This repository provides source code for reproducing results in our paper, including simulation data and real data. 

## Table of Contents

- [Requirements](#Requirements)
- [Install](#install)
- [Reproduction](#reproduction)
	- [Simulation Data](#simulation-data)
    - [Real Data](#real-data)
        - [UCI Datasets](#uci-datasets)
        - [Image Datasets](#image-datasets)
    - [Outlier Detection](#outlier-detection)
- [Related Applications](#related-applications)
- [Contact](#contact)
- [Cite](#Cite)
- [License](#license)

## Requirements

- TensorFlow=1.13.1

## Install

Roundtrip can be downloaded by
```shell
git clone https://github.com/kimmo1019/Roundtrip
```
Installation has been tested in a Linux/MacOS platform.

## Reproduction

This section provides instructions on how to reproduce results in the original paper.

### Simulation data

We tested Roundtrip on three types of simulation datasets. 1) Indepedent Gaussian mixture. 2) 8-octagon Gaussian mixture. 3) Involute.

### Real Data

We tested Roundtrip on different types of real data including five datasets from UCI machine learning repository and two image datasets

#### UCI Datasets

The datasets can be downloaded at 

#### Image Datasets

MNIST and CIFAR-10 were used in our study.

### Outlier Detection

We introduced three outlier detection datasets from OOD library.

## Related Applications

Roundtrip has various applications including unsupervised learning, likelihood-free inference and sequencial sequantial Markov chain Monte Carlo.

## Contact

Feel free to open an issue in Github or contact `liu-q16@mails.tsinghua.edu.cn` if you have any problem in implementing Roundtrip model.

## Cite

If you use Roundtrip in your research, please consider cite our paper 

## License

This project is li