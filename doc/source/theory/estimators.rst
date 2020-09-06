Statistical Estimators
======================

The error function (ERF) that assesses the goodness of the compression by measuring the
distance between the prior and the compressed distributions is defined as


.. math::
   \text{ERF}^{(ES)} = \frac{1}{N_{ES}} \sum\limits_{i} \left( \frac{C_i^{(ES)} - O_i^{(ES)}}{O_i^{(ES)}} \right)^2


where :math:`N_{ES}` is the normalization factor for a given estimator :math:`ES`, 
:math:`O_i^{(ES)}` represents the value of that estimator :math:`ES` computed at a generic 
point :math:`i` (which could be a given value of :math:`(x,Q)` in the PDFs), and 
:math:`C_i^{(ES)}` is the corresponding value of the same estimator in the compressed set.

The total value of ERF is then given by


.. math::
   \text{ERF}_{TOT} = \frac{1}{N_{est}} \sum\limits_{k} \text{ERF}^{(ES)}


where :math:`k` runs over the number of statistiacal estimators used to quantify the distance
between the original and compressed distributions, and :math:`N_{est}` is the total number
of statistical estimators.


The estimators taken into account in the computation of the ERF are the following:

1. The first four moments of the distribution including: central value, standard deviation,
   skewness and kurtosis

2. The kolmogorov-Smirnov test. This ensures that the higher moments are automatically
   adjusted.

3. The correlation between the mutpiple PDF flavours at different x points. This information
   is important to ensure that PDF-induced correlations in physical cross-sections are 
   succesfully maintained.


Mean Value
----------

Let us denote by :math:`g_i^{(k)}(x_j, Q_0)` and :math:`f_i^{(k)}(x_j, Q_0)` respectively the prior 
and the compressed sets for a flavor :math:`i` at the position :math:`j` of the :math:`x`-grid
containing :math:`N_x` points. The central values are defined as


.. math::
   f_i^{CV}(x_j, Q_0) = \frac{1}{N_{rep}} \sum\limits_{r=1}^{N_{rep}} f_i^{(r)} (x_j, Q_0)

.. math::
   g_i^{CV}(x_j, Q_0) = \frac{1}{\tilde{N}_{rep}} \sum\limits_{r=1}^{\tilde{N}_{rep}} g_i^{(r)} (x_j, Q_0)


Therefore, :math:`\text{ERF}^{(ES)}` is defined as


.. math::
   \text{ERF}_{CV} = \frac{1}{N_{CV}} \sum\limits_{i=-nf}^{nf} \sum\limits_{j=1}^{N_x} \left( \frac{f_i^{CV}(x_j, Q_0) - g_i^{CV}(x_j, Q_0)}{g_i^{CV}(x_j, Q_0)} \right)^2


where the normalization factor :math:`N_{CV}` is given by


.. math::
   N_{CV} = \frac{1}{N_{rand}} \sum\limits_{d=1}^{N_{rand}} \sum\limits_{i=-nf}^{nf} \sum\limits_{j=1}^{N_x} \left( \frac{r_i^{d,CV}(x_j, Q_0) - g_i^{CV}(x_j, Q_0)}{g_i^{CV}(x_j, Q_0)} \right)^2




.. note::

   Implement documentation for other estimators and the computation of the normalizations
