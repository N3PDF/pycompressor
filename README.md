<p align="center">
  <b>pyCompressor</b>
</p>
<p align="center">
  Fast and efficient compression code for Monte Carlo PDF sets
</p>
<p align="center">
  <a href="https://github.com/N3PDF/pycompressor"><img src="https://github.com/N3PDF/pycompressor/workflows/pytest/badge.svg"></a>
  <a href="https://n3pdf.github.io/pycompressor/"><img src="https://github.com/N3PDF/pycompressor/workflows/docs/badge.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.4616385"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4616385.svg" alt="DOI"></a>
  <a href="https://doi.org/10.5281/zenodo.4616385"><img src="https://img.shields.io/badge/arXiv-2104.04535v2-b31b1b.svg" alt="DOI"></a>
</p>

### New features

Additional new features have been added to the following python package. The two main features are:
<p align="justify">
  <b> Covariance Matrix Adaptation-Evlotion strategy </b>
  In addition to the Genetic Algorithm (GA) implemented in the original compression, there is now the possibility to choose 
  as a minimizer the CMA. The choice of minimizer can be defined in the <a href="">runcard.yaml</a> file.
</p>
<p align="justify">
  <b> Generative Adversarial Strategy (GANs) </b>
  This is a standalone python <a href="https://github.com/N3PDF/ganpdfs/tree/master">package</a>
  that can enhance the statistics of the prior PDF replicas before compression by generating 
  synthetic replicas. For more details, refer to the <a href="https://n3pdf.github.io/ganpdfs/">documentation</a>. 
  In a similar way, in order to trigger the enhancement, one just has to set the value of <b>enhanced</b> 
  in the runcard to be <b>True</b>. Setting this value to <b>False</b> will just run the standard 
  compression. The GANs also requires extra-parameters (as shown in the example 
  <a href="https://github.com/N3PDF/pycompressor/blob/master/runcard.yml">runcard.yaml</a>) that 
  defines the structure of the networks.
</p>

### Installation

To install `pyCompressor`, just type:
```bash
python setup.py install
```
or if you are a developer:
```bash
python setup.py develop
```

### How to use

The input parameters that define the compression is contained in a YAML file. To run
the `pyCompressor` code, just type the following: 
```bash
pycomp runcards/runcard.yml [--threads NUMB_THREADS]
```
A detailed instruction on how to set the different parameters in the runcard can be found [here](https://n3pdf.github.io/pycompressor/howto/howto.html).
And to controo the parallelization, have a look at the following [section](https://n3pdf.github.io/pycompressor/howto/howto.html#controlling-the-parallelization).
Notiice that by default, the methodology is based on the [standard](https://arxiv.org/abs/1504.06469) approach.
In order to compress from an enhanced set, the entry `existing_enhanced` has to be set to `True`.

### Generating compressed PDF set & post-analysis

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

### Citation

If you use the package please at least cite one of the followings:

```bibtex
@article{Carrazza:2021hny,
  author = "Carrazza, Stefano and Cruz-Martinez, Juan M. and Rabemananjara, Tanjona R.",
  title = "{Compressing PDF sets using generative adversarial networks}",
  eprint = "2104.04535",
  archivePrefix = "arXiv",
  primaryClass = "hep-ph",
  doi = "10.1140/epjc/s10052-021-09338-8",
  journal = "Eur. Phys. J. C",
  volume = "81",
  number = "6",
  pages = "530",
  year = "2021"
}

@software{pycompressor,
    author       = {Rabemananjara, Tanjona R. and Cruz-Martinez, Juan M. and Carrazza, Stefano},
    title        = {N3PDF/pycompressor: pycompressor v1.1.0},
    month        = Mar.,
    year         = 2020,
    publisher    = {Zenodo},
    version      = {v1.1.0},
    doi          = {10.5281/zenodo.4616385},
    url          = {https://doi.org/10.5281/zenodo.4616385}
}
```
