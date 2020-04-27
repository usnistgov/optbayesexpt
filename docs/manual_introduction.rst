Introduction
------------

This manual describes an implementation of optimal Bayesian experimental
design methods to control measurement settings in order to efficiently
determine model parameters. In situations where parametric models would
conventionally be fit to measurement data in order to obtain model
parameters, these methods offer an adaptive measurement strategy capable
of reduced uncertainty with fewer required measurements. These methods
are therefore most beneficial in situations where measurements are
expensive in terms of money, time, risk, labor or other cost. The price
for these benefits lies in the complexity of automating such
measurements and in the computational load required. It is the goal of
this package to assist potential users in overcoming at least the
programming hurdles.

Optimal Bayesian experimental design is not new, at least not in the
statistics community. A review paper from 1995 by `Kathryn Chaloner and
Isabella Verinelli <https://projecteuclid.org/euclid.ss/1177009939>`__
reveals that the basic methods had been worked out in preceding decades.
The methods implemented here closely follow `Xun Huan and Youssef M.
Marzouk <http://dx.doi.org/10.1016/j.jcp.2012.08.013>`__ which
emphasizes simulation-based experimental design. Optimal Bayesian
experimental design is also an active area of research.

There are at least three important factors that encourage application of
these methods today. First, the availability of flexible, modular
computer languages such as Python. Second, availability of cheap
computational power. Most of all though, an increased awareness of the
benefits of code sharing and reuse is growing in scientific communities.

.. include:: manual_philosophy.rst
.. include:: manual_requirements.rst