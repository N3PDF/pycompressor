![pytest](https://github.com/N3PDF/pycompressor/workflows/pytest/badge.svg)
[![documentation](https://github.com/N3PDF/pycompressor/workflows/docs/badge.svg)](https://n3pdf.github.io/pycompressor/)

### pycompressor
----------------

Fast python implementation of PDF set **compressor** (https://arxiv.org/abs/1504.06469). A detailed
documentation will be (slowly) added to https://n3pdf.github.io/pycompressor/. For benchmarks,
have a look at this [folder](https://github.com/N3PDF/pycompressor/tree/GANsInterface/doc/source/img-src).


#### New features

Additional new features have been added to the following python package. The two main features
are:
- **Covariance Matrix Adaptation-Evlotion strategy (CMA-ES):** in addition to the Genetic 
Algorithm (GA), there is now the possibility to choose as a minimizer the CMA. The choice
of minimizer can be defined in the `runcard.yml` file.
```yaml
minimizer : genetic     # Or "cma"
```
- **Generative Adversarial Strategy (GANs):** this is a standalone python [package](https://github.com/N3PDF/ganpdfs/tree/master)
that can enhance the statistics of the prior PDF replicas before compression by generating 
synthetic replicas. For more details, refer to the [documentation](https://n3pdf.github.io/ganpdfs/) 
(still has to be done). In a similar way, in order to trigger the enhancement, one just has to set 
the value of `enhance` in the runcard to be `True`. Setting this value to `False` will just run the 
regular compression. The GANs also requires extra-parameters (as shown in the example 
[runcard.yml](https://github.com/N3PDF/pycompressor/blob/master/runcard.yml)) that defines
the structure of the networks.
```yaml
enhance : True
```
Below is shown a caricatural picture of the pycompressor workflow.
<p align="center">
  <img alt="diagram" src="https://github.com/N3PDF/pycompressor/blob/GANsInterface/doc/source/img-src/diagram.png" width=800>
</p>


#### Installation

To install pyCompressor, just type:
```bash
python setup.py install
```
or if you are a developer:
```bash
python setup.py develop
```
The package can aslo be installed via the Python Package Index (PyPI) by running
```bash
pip install pycompressor --upgrade
```

#### How to use

The input parameters that define the compression is contained in a YAML file. To run
the `pycompressor` code, just type the following: 
```bash
pycompressor runcard.yml
```
If `enhance` is turned `True`, the code will generate a folder that has the following
structure,
```bash
<PDF_NAME>_enhanced
├── checkpoint
│   ├── checkpoint
│   ├── ckpt-1.data-00000-of-00001
│   └── ckpt-1.index
│   └── ...
├── compress_<PDF_NAME>_enhanced_<NB_COMPRESSED_REPLICAS>_output.dat
├── filter.yml
├── input-runcard.json
├── iterations
│   └── pdf_generated_at_<ITERATION>.png
├── losses_info.json
└── nnfit
    ├── <PDF_NAME>_enhanced.info
    ├── replica_<REPLICA_INDEX>
    │   ├── <PDF_NAME>_enhanced.dat
    │   └── <PDF_NAME>.exportgrid
    └── ...
```
where:
- `checkpoint`: store the evolution of the GANs training. In case a long runnning training
is interupted, the last checkpoint can be restored and the training can re-start from there.
- `nnfit`: contains (i) the output grids (`exportgrid`) from the GANs-generated replicas. 
This is in the same format as the N3FIT-grid in order to take advantage of the [evolven3fit](https://github.com/NNPDF/nnpdf/blob/master/n3fit/evolven3fit/evolven3fit.cc) to evolve the grids; (ii) the `.dat` files that contain the evolved 
LHAPDF-like grid from `evolven3fit`. Then, the [postgans](https://github.com/N3PDF/ganpdfs/tree/GANsInterface/src/pycompressor/postgans.py)
module creates a symbolic link between the `.dat` files and the files that are put in
the LHAPDF data directory. The compressor then uses that PDF to be the prior.
- `compress_<PDF_NAME>_enhanced_<NB_COMPRESSED_REPLICAS>_output.dat`: contains the index of 
replicas that are retained from the compressor.

If on the other hand `enhance` is set to `False`, the above steps will be skipped and chooses
the input PDF as the prior. This will then generate a `compress_<PDF_NAME>_<NB_COMPRESSED_REPLICAS>_output.dat`
file that is located in the same parent directory as above.

#### Generating compressed PDF set & post-analysis

The code will create a folder named after the prior PDF sets. To generate the
compressed PDF grid, run the following command:
```bash
./tools/compressed_grid.py <PDF_NAME>(_enhanced)/compressed_<PDF_NAME>(_enhanced)_<NB_COMPRESSED>_output.dat
```
This will generate a folder `<PDF_NAME>(_enhanced)/compressed_<PDF_NAME>(_enhanced)_<NB_COMPRESSED>` 
containing the compressed PDF replicas.

Finally, in order to generate ERF plots, enter in the `erfs_output` directory and run the following:
```bash
../tools/pycompressor_validate.py --random erf_randomized.dat --reduced erf_reduced.dat
``` 
This script can also plot the ERF validation from the old compressor code by adding the flag
`--format ccomp`.

#### Bottlenecks

The following bottleneck(s) would be solved once everything is put in place.
- For the time being, the `filter.yml` that is required by `evolven3fit` has to be manually
put in place in the main directory. A systematic way of dealing with this will be put
in place once we know where everything goes.
