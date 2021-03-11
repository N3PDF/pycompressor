How to use
==========

Once the **pyCompressor** package is installed, runnig a compression is very easy. It just takes
an input run card in which all the input parameters are defined. The run card is subdivided into
two distincts part: the compression and the GANs parameters.


.. code-block:: bash

   pycompressor runcards/runcard.yml [--threads NUMB_THREADS]


The compression per-say requires the following keys:

    - ``pdf`` - *str* : name of the inpit/prior PDF set
    - ``compressed`` - *int* : size of the compressed replicas
    - ``minimizer`` - *str* : name of the minimizer (`genetic` or `cma`)
    - ``est_dic`` - *dict* : dictionary of estimators
    - ``gans`` - *dict* : dictionary containing input parameters for GANs

 
One of the keys for the ``gan`` entry is a ``runcard`` which gets passed to the *ganpdfs* code.
For details on how to set the parameters for the GAN, have a look `here <https://n3pdf.github.io/ganpdfs/howto/howto.html>`_.


An example of an input card is shown below:


.. code-block:: yaml

   ###################################################
   # PDF Set                                         #
   ###################################################
   pdfsetting:
     pdf: NNPDF40_nnlo_as_0118_1000
     existing_enhanced: False
   
   ###################################################
   # Size of compressed PDF replicas                 #
   ###################################################
   compressed: 500
   
   ###################################################
   # Choice of Minimizer                             #
   # Options:                                        #
   #   - genetic                                     #
   #   - cma                                         #
   ###################################################
   minimizer: genetic
   
   ###################################################
   # Statistical Estimators                          #
   # Extra-options for Moment:                       #
   #   - moment5th                                   #
   #   - moment6th                                   #
   ###################################################
   est_dic:
     corr_estimators:
       - correlation
     stat_estimators:
       - kolmogorov_smirnov
     moment_estimators:
       - mean
       - stdev
       - skewness
       - kurtosis
   
   ###################################################
   # Enhance statistics of Prior                     #
   ###################################################
   gans  :
     enhance : False
     runcard : ganpdfs
     total_replicas: 3000



Running GANs within pyCompressor
--------------------------------


Although it is advised to run the `ganpdfs` code independently, it is possible to call it
within the `pyCompressor` code by setting ``enhance`` to `True` in the runcard. In this
scenario, the code will first enhance the statistic the prior using GANs.
Once the generation of the extra-replicas is finished, the output grids are evolved using
`evolven3fit <https://github.com/NNPDF/nnpdf/blob/master/n3fit/evolven3fit/evolven3fit.cc>`_. 
Then, the :mod:`pyCompressor.postgans` module (in a similar fashion as postfit) creates a 
symbolic link of both the original and the generated PDF sets into the LHAPDF data directory. 
The new enhanced Monte Carlo set of PDF replicas is then used as input to the compressor. 
Once the compression is finished, a folder is created in the main directory with the folowing 
structure:


.. code-block:: bash

   <PRIOR_PDF_NAME>_enhanced
   ├── filter.yml
   ├── input-runcard.json
   ├── losses_info.json
   ├── nnfit
   │    ├── <PDF_NAME>_enhanced.info
   │    ├── replica_<REPLICA_INDEX>
   │    │   ├── <PDF_NAME>_enhanced.dat
   │    │   └── <PDF_NAME>.exportgrid
   │    └── ...
   └── compress_<PRIOR_PDF_NAME>_enhanced_<NB_COMPRESSED_REPLICAS>_output.dat


where:

    - **losses_info.json** stores the losses of the generator and the critic/discriminator for the
      GANs model.
    - **filter.yml** contains the information on the theory ID use to reproduce the prior replicas.
    - **input-runcard.json** is a copy of the input parameters that were fed to the GANs.
    - **nnfit** has more or less the same folder structure as the output from n3fit. It contains the
      a `replica_$REPLICA_INDEX` that contains a `.exportgrid` file used by evolven3fit for the
      evolution. That is where the evolved grid in the format `.dat` is also stored.
    - **compress_<PDF_NAME>_enhanced_<NB_COMPRESSED_REPLICAS>_output.dat** contains the index of
      the reduced replicas along with the final ERF value.


If ``enhance`` is instead set to `False`, the folder will just simply be:


.. code-block:: bash

   <PRIOR_PDF_NAME>_enhanced
   └── compress_<PRIOR_PDF_NAME>_enhanced_<NB_COMPRESSED_REPLICAS>_output.dat



Adiabatic minimization
----------------------


Since compressing from an enhanced set could be difficult due to the limitation of the minimization
algorithm, it is possible to perfrom an adiabatic minimization by setting ``existing_enhanced`` to
`True` in the runcard. In this case, the minimization is perfromed in two steps: (1) a standard
compression of the prior, (2) a compression using the enhanced set but using as a starting point
the space in which the best from the standard compression was generated.



PDF grid and Validation plot
============================


To generate the reduced Monte Carlo set of PDF replicas, simply run:


.. code-block:: bash

   get-grid -i <PRIOR_PDF_NAME>/compressed_<PDF_NAME>_<NB_COMPRESSED>_output.dat


Note that if the compression is done from an enhanced set, the output folder will be append by **_enhanced**.

Finally, to check that the reduced Monte Carlo set indeed faithfully reproduces the statistics of the
prior, ERF plots for each of the estimator can be generated and compared to a random selection. To generate
the ERF validation plots, enter in the ``erfs_output`` folder and run the following:


.. code-block:: bash

   validate --random erf_randomized.dat --reduced erf_reduced.dat



Controlling the parallelization
===============================

The backend of pycompressor is the JIT compiler [numba](https://numba.pydata.org) and it is numba who controls the parallelization of the calculations within the code.
The number of cores to be used can be controlled with the appropiate settings to the following environmental variables:

.. code-block:: bash

    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export NUMBA_NUM_THREADS=4

An interface to control the numba number of threads is also provided as the command line argument ``threads``.
Note that in no case can ``threads`` be greater than the environmental variable (if given) ``NUMBA_NUM_THREADS``.

.. code-block:: bash

  pycomp runcards/runcard.yml --threads 4
