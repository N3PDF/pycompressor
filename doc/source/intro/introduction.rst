Why pyCompressor?
=================


The two new main features of the `pyCompressor` package compared to the C++ implementation are:

    - A new minimization algoirthm, **Covariance Matrix Adapation Strategy (CMA-ES)**. In addition to the
      **Genetic Algorithm (GA)**, there is now the possibility to choose as a minimizer the CMA-ES.
    - A **Generative Adversarial (GANs) Strategy** that can be used to enhance the statistices of the prior.
      This relies on another python package called `ganpdfs` which is a standalone code based on tensorflow
      designed specifically to learn the probability distribution of the a prior dataset (PDF replicas) and
      generate new ones that follow the same distribution.


The new desing flow of the `pyCompressor` package is schematically described in the following diagram.


 .. figure:: ../img-src/diagram.png
    :align: center


 
The prior Monte Carlo set of replica of size :math:`N` is given as input from which a PDF grid of 
:math:`(N, NF, X)`-points (where :math:`NF` and :math:`X` resp. denote the total number of flavors and the size of
the :math:`x`-grid) is computed. If the choice to enhance the statistics is activated, the prior PDF set goes through 
the GANs. This will generate a new Monte Carlo PDF set :math:`\tilde{N}` (with :math:`N < \tilde{N}`). A subset sample 
of the PDF replicas (prior or ehancend) with size :math:`\bar{N}` is then extracted in order to compute the Error Function 
(ERF). The minimization of the ERF is then performed either using the GA or the CMA which can be defined in the input 
runcard. The minimizer basically tries to find a candidate sample that leads to the smallest value of ERF. Once a 
convergence is achieved after a given iteration, the agorithm stops and outputs a compressed Monte Carlo set of PDF
replicas with size :math:`\bar{N}` (with :math:`\bar{N} < N < \tilde{N}`).



Benchmarks
==========


The following benchmarks agains the C++ code is done without enhancing the statistics of the prior (i.e. truning
off the GANs). The input PDF set chosen here is the NNPDF31_nnlo_as_0118_1000 with 1000 replicas. The mahine used
for testing is a Ryzen 5 2600 (12 threads) with 16 GB of RAM and a GTX 1060.


 .. figure:: ../img-src/speed_benchmark.png
    :align: center


 
As shown below, the performance of the pyCompressor is comparable to the C++ implementation with as slightly
faster convergence.


 .. figure:: ../img-src/erf_iterations.png
    :align: center
