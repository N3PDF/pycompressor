![pytest](https://github.com/N3PDF/pycompressor/workflows/pytest/badge.svg)
[![documentation](https://github.com/N3PDF/pycompressor/workflows/docs/badge.svg)](https://n3pdf.github.io/pycompressor/)

### pycompressor

Fast and efficient python implementation of PDF set **compressor** (https://arxiv.org/abs/1504.06469). 

#### New features

Additional new features have been added to the following python package. The two main features are:
- **Covariance Matrix Adaptation-Evlotion strategy (CMA-ES):** in addition to the Genetic 
Algorithm (GA), there is now the possibility to choose as a minimizer the CMA. The choice
of minimizer can be defined in the `runcard.yml` file.
- **Generative Adversarial Strategy (GANs):** this is a standalone python [package](https://github.com/N3PDF/ganpdfs/tree/master)
that can enhance the statistics of the prior PDF replicas before compression by generating 
synthetic replicas. For more details, refer to the [documentation](https://n3pdf.github.io/ganpdfs/) 
(still has to be done). In a similar way, in order to trigger the enhancement, one just has to set 
the value of `enhance` in the runcard to be `True`. Setting this value to `False` will just run the 
standard compression. The GANs also requires extra-parameters (as shown in the example 
[runcard.yml](https://github.com/N3PDF/pycompressor/blob/master/runcard.yml)) that defines
the structure of the networks.

#### Installation

To install `pyCompressor`, just type:
```bash
python setup.py install
```
or if you are a developer:
```bash
python setup.py develop
```

#### How to use

The input parameters that define the compression is contained in a YAML file. To run
the `pycompressor` code, just type the following: 
```bash
pycomp runcards/runcard.yml [--threads NUMB_THREADS]
```
A detailed instruction on how to set the different parameters in the runcard can be found here.

#### Generating compressed PDF set & post-analysis

The code will create a folder named after the prior PDF sets. To generate the
compressed PDF grid, run the following command:
```bash
get-grid -i <PDF_NAME>/compressed_<PDF_NAME>_<NB_COMPRESSED>_output.dat
```
Note that if the compression is done from an enhanced set, the output folder will be append by `_enhanced`.

Finally, in order to generate ERF plots, enter in the `erfs_output` directory and run the following:
```bash
validate --random erf_randomized.dat --reduced erf_reduced.dat
``` 
This script can also plot the ERF validation from the old compressor code by adding the flag
`--format ccomp`.

#### Warning

This package cannot be installed with python 3.9 yet due to the numba dependency. This will be resolved
soon according to [#6579](https://github.com/numba/numba/pull/6579).
