![pytest](https://github.com/N3PDF/pycompressor/workflows/pytest/badge.svg)

### pycompressor

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

####  Running
```bash
pycompressor -p PDF_NAME -n NB_COMPRESSED_REPLICAS -m [MINIMIZER]
```

#### Post-Analysis
The code will generate a folder named after the prior PDF sets. To generate the
compressed PDF grid, run the following command:
```bash
./tools/compressed_grid.py PDF_NAME
```
Finally, in order to generate ERF plots, copy the file `compressor_validate.C`
in `tools` into the compressed folder and run (this requires root):
```bash
root -l compressor_validate.C
``` 
