Statistical Estimators
======================

The error function (ERF) that assesses the goodness of the compression by measuring the
distance between the prior and the compressed distributions is defined as


.. math::
   \text{ERF}_{(ES)} = \frac{1}{N_{ES}} \sum\limits_{i} \left( \frac{C_i^{(ES)} - O_i^{(ES)}}{O_i^{(ES)}} \right)^2


where :math:`N_{ES}` is the normalization factor for a given estimator :math:`ES`, 
:math:`O_i^{(ES)}` represents the value of that estimator computed at a generic 
point :math:`i` (which could be a given value of :math:`(x,Q)` in the PDFs), and 
:math:`C_i^{(ES)}` is the corresponding value of the same estimator in the compressed set.

The total value of ERF is then given by


.. math::
   \mathrm{ERF}_{\mathrm{TOT}} = \frac{1}{N_{\mathrm{est}}} \sum\limits_{k} \text{ERF}^{(ES)}


where :math:`k` runs over the number of statistiacal estimators used to quantify the distance
between the original and compressed distributions, and :math:`N_{\mathrm{est}}` is the total number
of statistical estimators.


The estimators taken into account in the computation of the ERF are the following:

1. The first four moments of the distribution including: central value, standard deviation,
   skewness and kurtosis

2. The kolmogorov-Smirnov test. This ensures that the higher moments are automatically
   adjusted.

3. The correlation between the mutpiple PDF flavours at different x points. This information
   is important to ensure that PDF-induced correlations in physical cross-sections are 
   succesfully maintained.


Mean Value, Variance, Higher Moments
------------------------------------

Let's denote by :math:`g_{i}^{(k)}\left(x_{j},Q_{0}\right)` and :math:`f_{i}^{(r)}\left(x_{j},Q_{0}\right)` 
respectively the prior and the compressed sets of replicas for a flavor :math:`i` at the position :math:`j` 
of the :math:`x`-grid containing :math:`N_{x}` points. :math:`N_{\text {rep }}` is the number of required 
compressed replicas. We then define the contribution to the ERF from the distances between central values 
of the prior and compressed distributions as follows

.. math::
   \mathrm{ERF}_{\mathrm{CV}}=\frac{1}{N_{\mathrm{CV}}} \sum_{i=-n_{f}}^{n_{f}} \sum_{j=1}^{N_{x}}\left(\frac{f_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right)-g_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right)}{g_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right)}\right)^{2}

where :math:`N_{\mathrm{CV}}` is the normalization factor for this estimator. We only include in the sum 
those points for which the denominator satisfies :math:`g_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right) \neq 0`. 
As usual, central values are computed as the average over the MC replicas, for the compressed set

.. math::
   f_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right)=\frac{1}{N_{\text {rep }}} \sum_{r=1}^{N_{\text {rep }}} f_{i}^{(r)}\left(x_{j}, Q_{0}\right)

while for the prior set we have

.. math::
   g_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right)=\frac{1}{\widetilde{N}_{\text {rep }}} \sum_{k=1}^{\tilde{N}_{\text {rep }}} g_{i}^{(k)}\left(x_{j}, Q_{0}\right)

Let us also define :math:`r_{i}^{t}\left(x_{j}, Q_{0}\right)` as a random set of replicas extracted from the prior 
set, where :math:`t` identifies an ensemble of random extractions. The number of random extraction of random sets
is denoted by :math:`N_{\text {rand }}`. Now, the normalization factors are extracted for all estimators as the lower 
:math:`68 \%` confidence-level value obtained after :math:`N_{\text {rand }}` realizations of random sets. In 
particular for this estimator we have

.. math::
   N_{\mathrm{CV}}=\left.\frac{1}{N_{\text {rand }}} \sum_{d=1}^{N_{\text {rand }}} \sum_{i=-n_{f}}^{n_{f}} \sum_{j=1}^{N_{x}}\left(\frac{r_{i}^{d, \mathrm{CV}}\left(x_{j}, Q_{0}\right)-g_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right)}{g_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right)}\right)^{2}\right|_{68 \% \text { lower band }}

For the contribution to the ERF from the distance between standard deviation, skewness and kurtosis, we can built 
expressions analogous to the above equation  by replacing the central value estimator with the suitable expression 
for the other statistical estimators, which in a Monte Carlo representation can be computed as

.. math::
   f_{i}^{\mathrm{STD}}\left(x_{j}, Q_{0}\right)=\sqrt{\frac{1}{N_{\mathrm{rep}}-1} \sum_{r=1}^{N_{\mathrm{rep}}}\left(f_{i}^{(r)}\left(x_{j}, Q_{0}\right)-f_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right)\right)^{2}}

.. math::
   f_{i}^{\mathrm{SKE}}\left(x_{j}, Q_{0}\right)=\frac{1}{N_{\mathrm{rep}}} \sum_{r=1}^{N_{\mathrm{rep}}}\left(f_{i}^{(r)}\left(x_{j}, Q_{0}\right)-f_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right)\right)^{3} /\left(f_{i}^{\mathrm{STD}}\left(x_{j}, Q_{0}\right)\right)^{3}

.. math::
   f_{i}^{\mathrm{KUR}}\left(x_{j}, Q_{0}\right)=\frac{1}{N_{\mathrm{rep}}} \sum_{r=1}^{N_{\mathrm{rep}}}\left(f_{i}^{(r)}\left(x_{j}, Q_{0}\right)-f_{i}^{\mathrm{CV}}\left(x_{j}, Q_{0}\right)\right)^{4} /\left(f_{i}^{\mathrm{STD}}\left(x_{j}, Q_{0}\right)\right)^{4}

for the compressed set, with analogous expressions for the original prior set.

The normalization factors for these estimators are extracted using the same strategy as above by averaging over random 
extractions of :math:`N_{\text {rep }}` replicas, exchanging :math:`\mathrm{CV}` by :math:`\mathrm{STD}, \mathrm{SKE}` 
and :math:`\mathrm{KUR}` respectively.


Kolmogorov-Smirnov
------------------


We define the contribution to the total ERF from the Kolmogorov-Smirnov (KS) distance as follows

.. math::
   \mathrm{ERF}_{\mathrm{KS}}=\frac{1}{N_{\mathrm{KS}}} \sum_{i=-n_{f}}^{n_{f}} \sum_{j=1}^{N_{x}} \sum_{k=1}^{(r)}\left(\frac{F_{i}^{k}\left(x_{j}, Q_{0}\right)-G_{i}^{k}\left(x_{j}, Q_{0}\right)}{G_{i}^{k}\left(x_{j}, Q_{0}\right)}\right)^{2}

where :math:`F_{i}^{k}\left(x_{j}, Q_{0}\right)` and :math:`G_{i}^{k}\left(x_{j}, Q_{0}\right)` are the 
outputs of the test for the compressed and the prior set ofreplicas respectively. The output of the test 
consists in counting the number of replicas containedin the :math:`k` regions where the test is performed. 
We count the number of replicas which fall in eachregion and then we normalize by the total number of replicas 
of the respective set. Here we haveconsidered six regions defined as multiples of the standard deviation of 
the distribution for each

.. math::
   \left[-\infty,-2 f_{i}^{\mathrm{STD}}\left(x_{j}, Q_{0}\right),-f_{i}^{\mathrm{STD}}\left(x_{j}, Q_{0}\right), 0, f_{i}^{\mathrm{STD}}\left(x_{j}, Q_{0}\right), 2 f_{i}^{\mathrm{STD}}\left(x_{j}, Q_{0}\right),+\infty\right]

where the values of the PDFs have been subtracted from the corresponding central value.

In this case, the normalization factor is determined from the output of the KS test for randomsets of replicas
extracted from the prior, denoted :math:`R_{i}^{k}\left(x_{j}, Q_{0}\right)` as follows

.. math::
   N_{\mathrm{KS}}=\frac{1}{N_{\text {rand }}} \sum_{d=1}^{N_{\text {rand }}} \sum_{i=-n_{f}}^{n_{f}} \sum_{j=1}^{N_{x}} \sum_{k=1}^{6}\left(\frac{R_{i}^{k}\left(x_{j}, Q_{0}\right)-G_{i}^{k}\left(x_{j}, Q_{0}\right)}{G_{i}^{k}\left(x_{j}, Q_{0}\right)}\right)^{2}

and we include in the sum those points for which the denominator satisfies :math:`G_{i}^{k}\left(x_{j}, Q_{0}\right) \neq 0`.


PDF Correlation
---------------

In addition to all the moments of the prior distribution, a sensible compression should also main-tain the 
correlations between values of :math:`x` and between flavours of the PDFs. In order to achieve this, correlations 
are taken into account in the ERF by meansof the trace method. We define acorrelation matrix :math:`C` for 
any PDF set as follows:

.. math::
   C_{i j}=\frac{N_{\text {rep }}}{N_{\text {rep }}-1} \cdot \frac{\langle i j\rangle-\langle i\rangle\langle j\rangle}{\sigma_{i} \cdot \sigma_{j}}

where it is defined that

.. math::
   \langle i\rangle=\frac{1}{N_{\text {rep }}} \sum_{r=1}^{N_{\text {rep }}} f_{i}^{(r)}\left(x_{i}, Q_{0}\right), \quad\langle i j\rangle=\frac{1}{N_{\text {rep }}} \sum_{r=1}^{N_{\text {rep }}} f_{i}^{(r)}\left(x_{i}, Q_{0}\right) f_{j}^{(r)}\left(x_{j}, Q_{0}\right)

For each flavornfwe define :math:`N_{x}^{\text{corr}}` points distributed in :math:`x` where the correlations arecomputed. 
The trace method consists in computing the correlation matrix :math:`P` for the prior set and then store its inverse
:math:`P^{−1}`. For :math:`n_{f}` flavours and :math:`N^{\text{corr}}_{x}` points we obtain:

.. math::
   g=\operatorname{Tr}\left(P \cdot P^{-1}\right)=N_{x}^{\text {corr }} \cdot\left(2 \cdot n_{f}+1\right)

After computing the correlation matrix for prior set, for each compressed set a matrix :math:`C` iscomputed and the 
trace is determined by

.. math::
   f=\operatorname{Tr}\left(C \cdot P^{-1}\right)

The compression algorithm then includes the correlation ERF by minimizing the quantity:

.. math::
   \mathrm{ERF}_{\mathrm{Corr}}=\frac{1}{N_{\mathrm{Corr}}}\left(\frac{f-g}{g}\right)^{2}

where :math:`N^{\text{Corr}}` is computed as usual from the random sets.
