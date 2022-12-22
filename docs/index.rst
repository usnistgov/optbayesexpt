.. OptBayesExpt documentation master file, created by
   sphinx-quickstart on Mon Jul 29 16:15:56 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


OptBayesExpt: Optimmal Bayesian Experiment Design
=================================================

:Author: R.\  D.\  McMichael
:Email: rmcmichael@nist.gov
:Affiliation: National Institute of Standards and Technology
:Version: 1.2.0
:Date: December 22, 2022

Overview
--------

This package offers an implementation of sequential Bayesian experiment
design, an adaptive strategy for controlling experiments.

The ``optbayesexpt`` package is avaiable on github.
https://www.github.com/usnistgov/optbayesexpt.

In publications using ``optbayesexpt``, please cite

   Robert D. McMichael, Sean M. Blakley, and Sergey Dushenko,
   *Optbayesexpt: Sequential Bayesian Experiment Design for Adaptive
   Measurements*,
   Journal of Research of National Institute of Standards and Technology,
   **126**, 126002 (2021),   https://doi.org/10.6028/jres.126.002.

What is it for?
^^^^^^^^^^^^^^^

It's for making smart setting choices in measurements that determine
the parameters of a model. It is for cases with

 - a known parametric model, i.e. an equation that relates unknown parameters
   and experimental settings to measurement predictions. Fitting functions used
   in least-squares fitting are good examples of parametric models.
 - an experiment (possibly computational) that uses a set-measure-repeat
   sequence with opportunities to change settings between measurements.

The benefit of these methods is that they provide settings choices
that have the best chance of making the parameter estimates more precise.
This feature is very helpful in situations where the measurements are
expensive.

It is not primarily designed for analyzing existing data, but some of the
code could be used with existing data to do Bayesian inference of parameter
values.

Note that *Bayesian optimization* addresses a different problem: finding a
maximum or minimum of an unknown function.

What does it do?
^^^^^^^^^^^^^^^^

It chooses measurement settings "live" based on accumulated data.

The sequential Bayesian experimental design algorithms play the role of an
impatient experimenter who monitors data from a running experiment and
changes the measurement settings in order to get better, more meaningful
data. Note the two steps here. The first step, looking at the data, is
really an act of extracting meaning from the numbers, learning something
about the system from the existing measurements. The second step, a
decision-making step, is using that knowledge to improve the measurement
strategy.

In the "looking at the data" role, the method uses Bayesian inference to
extract and update information about model parameters as new measurement
data arrives.  Then, in the
"decision making" role, the methods use the updated parameter knowledge
to select settings that have the best chance of refining the parameters.

The most important role is the responsibility of the user. As delivered, the
BayesOptExpt is ignorant of the world, and it's the user's responsibility
to describe the world in terms of a reliable model, reasonable parameters, and
reasonable experimental settings. As with most computer programs, "the
garbage in, garbage out" rule applies.

Documentation
-------------

.. :caption: Contents:

.. toctree::
   :maxdepth: 4

   quickstart
   manual
   manual_demos
   versions
   optbayesexpt

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
