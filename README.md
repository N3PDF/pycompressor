![pytest](https://github.com/N3PDF/pycompressor/workflows/pytest/badge.svg)
[![documentation](https://github.com/N3PDF/pycompressor/workflows/docs/badge.svg)](https://n3pdf.github.io/pycompressor/)

### pycompressor
----------------

Python implementation of PDF set **compressor** (https://arxiv.org/abs/1504.06469).

#### Installation

To install pyCompressor, just type:
```bash
python setup.py install
```
or if you are a developer:
```bash
python setup.py develop
```

#### How to use

The input parameters that define the compression is contained in a YAML file. To run
the `pycompressor` code, one just needs to type the following: 

```bash
pycompressor runcard.yml
```

#### Post-Analysis

The code will create a folder named after the prior PDF sets. To generate the
compressed PDF grid, run the following command:
```bash
./tools/compressed_grid.py <PDF_NAME>/compressed_<PDF_NAME>_<NB_COMPRESSED>_output.dat
```
Finally, in order to generate ERF plots, copy the file `compressor_validate.C`
in `tools` into the `erfs_output` folder and run the following as root:
```bash
root -l compressor_validate.C
``` 
