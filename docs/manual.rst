
Optimal Bayesian Experimental Design
====================================

| R. D. McMichael rmcmichael@nist.gov
| National Institute of Standards and Technology
| Gaithersburg, MD USA March 29, 2019

Contents
--------

-  `Introduction <#introduction>`__
-  `Philosophy and goals <#philosophy>`__
-  `Requirements <#requirements>`__
-  `Demos <#demos>`__
-  `Theory of Operation <#theory>`__

## Introduction

This manual describes an implementation of optimal Bayesian experimental
design methods to control measurement settings in order to efficiently
determine model parameters. In situations where parametric models would
conventionally be fit to measurement data in order to obtain model
parameters, these methods offer an adaptive measurement strategy capable
of reduced uncertainty with fewer required measurements. These methods
are therefore most beneficial in situations where measurements are
expensive in terms of money, time, risk, labor or discomfort. The price
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

## Philosophy and goals

   If it sounds good, it is good > Duke Ellington

Jazz legend Duke Ellington was talking about music, where it’s all about
the sound. For this package, it’s all about being useful. The goals are
modest: to adapt some of the developments in optimal Bayeseian
experimental design research for practical use in laboratory settings,
giving users tools to make better measurements.

-  If it’s a struggle to use, it can’t run good.
-  If technical jargon is a barrier, it can’t run good
-  If the user finds it useful, it runs good.
-  If it runs good, it is good.

## Requirements for users

It takes a little effort to get this software up and running. Here’s
what a user will need to supply to get started.

1. Python 3.x with the ``numpy`` python package
2. An experiment that yields measurement results with uncertainty
   estimates.
3. A parametric model for the experiment, typically a function with
   parameters to be determined.
4. A working knowledge of Python programming, enough to follow examples
   and program a model function.
5. ``Matplotlib.pyplot`` and ``Matplotlib.gridspec`` modules for running
   demo programs.

## Demos

Locating a Lorentzian peak
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _images/demoLorentzfig1.png
   :alt: Comparison of measure-then-fit and OBED method

   Comparison of measure-then-fit and OBED method

Figure 1: A comparison of measure-then-fit (left) and optimal Bayesian
experimental design (right). Both methods measure the same “true” peak
with Gaussian noise added (:math:`\sigma = 1`) independently. The peak
parameters are selected randomly: center between 2 and 4, height between
2 and 5, background between -1 and 1 and peak width between 0.1 and 0.3.
On the left, 30 evenly-spaced “measurements” are made and fit using
``scipy.optimize.curve_fit()``. A curve using the best-fit parameters is
plotted for comparison with the true curve. The diagonal element of the
covariance matrix is taken as the square of the uncertainty. On the
right, the optimal Bayes experimental design method is used
sequentially. Iterations stop when the standard deviation of the ``x0``
peak center distribution is less than the uncertainty of the fit on the
left. Green curves are generated from random draws from the parameter
distribution at the stopping point. Typical runs of this example require
approximately 1/4 to 1/2 as many measurements as the measure-then-fit
method. See ``Demos/demoLorentzian.py``.

Measurement speedup
~~~~~~~~~~~~~~~~~~~

.. figure:: _images/rootN.png
   :alt: Mechanism of speedup

   Mechanism of speedup

Figure 2: A different view of the Lorentzian peak problem, contrasting
efficiency differences between methods. The left panel shows results
after 3000 measurements of a spectrum. The simulated experimental noise
is Gaussian with standard deviation :math:`\sigma = 1`. In the “average
& fit” method, 100 simulated measurements are averaged at each of 30
points, yielding a uniform uncertainty of :math:`\sigma_y = 0.1`. In the
OptBayesExpt method, the algorithm focuses attention on the sides of the
peak, as shown in the histogram. The symbol areas are proportional to
the weights of the data points, :math:`1/\sigma^2`. The largest points
correspond to :math:`\sigma_y \approx 0.035.`

The right panel plots the evolution of error and uncertainty in the peak
center :math:`x_0` with the number of accumulated measurements. Plotted
values summarize the results of 100 runs, each with 3000 OptBayesExpt
measurments and 15 000 average & fit measurements. Dark lines are the
rms measurement errors, i.e. differences between the known peak position
in the simulated measurement data, and the best-fit or distribution
means of the analyses. Light traces show the evolution of the estimated
uncertainties for the methods, either the uncertainty of the fit, or the
standard deviation of the parameter probability distribution. Both the
average & fit technique and the OptBayesExpt method scale like
:math:`\sigma_{x0} \propto 1/\sqrt{N}` (:math:`N` = measurement number)
for large :math:`N`, but the OptBayesExpt requires approximately 4 times
fewer measurements to achieve the same levels of uncertainty and error.

Details: In both methods, the peak center, baseline value and the peak
height were treated as unknowns, and the line width was fixed ad 0.15.
The average and fit method used ``scipy.optimize.curve_fit()``. See
``Demos/fit_vs_obe_makedata.py`` and ``Demos/fit_vs_obe_plots.py`` for
the calculation and plotting of this demo.

Tuning a :math:`\pi` pulse
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _images/pipulsefig.png
   :alt: measurements of transition probability vs pulse length and
   detuning

   measurements of transition probability vs pulse length and detuning

Figure 3: A :math:`\pi` pulse is a method of inverting spins that is
frequently used in nuclear magnetic resonance (NMR and MRI) and pulsed
electron paramagnetic resonance (EPR). In order to be accurate, the
duration and frequency of the radio-frequency pulse must be tuned. On
the left, the background image displays the model photon counts for
optically detected spin manipulation for different frequency detunings
and pulse lengths. White indicates the expected result for spin up and
black, spin down. Points indicate simulated measurement settings, with
sequence in order from white to dark red. Simulated measurements have
1\ :math:`\sigma` uncertainties of 100. The right panel displays the
evolution of the probability distribution function with the number “N”
of measurements. See ``Demos.pipulse.py``.

Slope Intercept
~~~~~~~~~~~~~~~

.. figure:: _images/slopeintercept.png
   :alt: Straight line measurement examples

   Straight line measurement examples

Figure 4: This demo uses a straight line model, a case where the “best”
measurement settings are known in advance; measurements at the ends of a
line are the most effective at reducing uncertainty in the slope and
intercept values. For reassurance that the straight line model is
appropriate, some measurements in the in the middle of the span might
also be desired. The OptBayesExpt class provides two methods for
flexibility in measurement selection. The ``opt_setting()`` method
selects the setting with the highest *utility* :math:`\max[U(x)]`. The
first panel shows that it behaves as expected, choosing measurements at
the ends of the line. The ``good_setting()`` method is more flexible,
selecting settings with a probability based on the *utility* and the
``pickiness`` parameter. The 2nd through 4th panels show that the
``good_setting()`` algorithm selects more diverse setting values as the
``pickiness`` is reduced. Note also that the standard deviations
increase from left to right as measurement resources are diverted away
from reducing uncertainty. Each run uses 40 points. See
``Demos/slopeintercept.py``.

## Theory of operation

The optimal Bayes experimental design method incorporates two main jobs,
which we can describe as “learning fast” and “making good decisions”

Learning fast
~~~~~~~~~~~~~

In terms of *when* new measurements become useful, there’s a sharp
contrast between conventional measure-then-fit strategy and optimal
Bayesian experimental design. Using the usual measure-then-fit strategy,
a predetermined sequence of measurements are made and a least-squares
fit is performed to extract model parameters. In this method, decisions
based on the new results only become possible after the fitting is done
in the last step. In contrast, the optimal Bayesian experimental design
method updates our parameter knowledge with each measurement result, so
that information-based decisions can be made as data is collected.

The process of digesting new data uses Bayesian inference, which frames
knowledge in terms of probability distributions. If this notion seems
unfamiliar, consider that the notation :math:`a\pm \sigma` is a
shorthand description of a probability distribution. So, accumulated
knowledge about model parameters :math:`\theta` is expressed as a
probability distribution function :math:`p(\theta)`. If
:math:`p(\theta)` is a broad distribution, then we have a lot of
uncertainty, and if :math:`p(\theta)` is a narrow distribution, the
uncertainty is small.

When new measurement results :math:`m` are available, we want to know
the new probability distribution :math:`p(\theta|m)` after :math:`m` is
taken into account. The vertical bar in the notation :math:`p(\theta|m)`
indicates a conditional probability, the distribution of :math:`\theta`
values given :math:`m`.

Bayes’ rule gives us

.. math::  p(\theta|m) = \frac{p(m|\theta) p(\theta)}{p(m)}. 

All of the terms here have technical names. The left side is the
*posterior* distribution, i.e. the distribution of parameters
:math:`\theta` after we include :math:`m`. On the right, distribution
:math:`p(\theta)` is the *prior*, representing what we knew about the
parameters :math:`\theta` before the measurement. In the denominator,
:math:`p(m)` is called the *evidence*. Because it has no :math:`\theta`
dependence, it’s not very important in this situation, and as wrong as
it sounds, we will ignore the *evidence*.

The term that we will focus on, :math:`p(m|\theta)`, is called the
*likelihood*. It’s the probability of getting measurement :math:`m`
given variable parameter values :math:`\theta`. Just as any conditional
probability :math:`p(a|b)` depends on both :math:`a` and :math:`b`,
:math:`p(m |\theta)` depends on both :math:`m` and :math:`\theta`. But
when we put it into practice, we’re going to have fixed measurement
results :math:`m_i` to “plug in” for :math:`m`. It’s important to keep
sight of the fact that :math:`p(m_i|\theta)` is still a function of
theta. Conceptually, we can try out different parameter values in our
model to produce a variety of measurement predictions. Some parameter
values (the more likely ones) will produce predictions closer to
:math:`m_i` and for other parameters (the less likely ones), model
predictions will be further away.

To go further, we need to specify what we mean by a measurement
:math:`m_i`. The measurement data includes the “value” :math:`y_i`
(which could be more than one number), but we also require it to include
measurement settings :math:`x_i` and an estimate of the uncertainties
:math:`\sigma_i`. Together, :math:`y_i` and :math:`\sigma_i` are more
than fixed numbers; they are statements about distributions
:math:`p(y|y_i, \sigma_i)` of other possible measurement outcomes
:math:`y` given a mean value :math:`y_i`. If this distribution is
symmetric, like a Gaussian, for example, then
:math:`p(y|y_i, \sigma_i) = p(y_i|y, \sigma_i)` the probability of
measuring :math:`y_i` given a mean value :math:`y` that’s provided by
the experimental model :math:`y=y(x_i,\theta)`.

.. math::  p(m_i|\theta) = \frac{1}{\sqrt{2\pi}\sigma_i} \exp\left[-\frac{[y_i - y(x_i, \theta)]^2 }{ 2\sigma_i^2 } \right]

.

Now we know how to update our “knowledge” of parameters :math:`\theta`
expressed as a probability distribution :math:`P(\theta)`. 1. Collect
measurement data including settings, :math:`x_i`, measurement values
:math:`y_i` and measurement uncertainties :math:`\sigma_i`. 2. For all
parameter combinations :math:`\theta` calculate the model’s prediction
of the mean measurement :math:`y(x_i, \theta)` 3. For all parameter
combinations :math:`\theta` multiply :math:`p(\theta)` by the likelihood
:math:`\exp[-(y_i-y(x, \theta))^2/2\sigma_i^2 ]`

As we make more measurements, we’ll update :math:`p(\theta|m_i)` to
:math:`p(\theta|m_i, m_j, \ldots)` and so on. In order to keep the
notation readable, we’ll adopt a convention that :math:`p(\theta)`
always represents the most up-to-date parameter distribution that we
have.

We just made several important assumptions: - That our model function
:math:`y(x, \theta)` is a good description of our system, and - that the
noise in our measurement is Gaussian with standard deviation
:math:`\sigma_i`.

On one hand we have to admit that these assumptions don’t allow us to
address all important cases. On the other hand, these are the same
assumptions we often make in doing least-squares curve fitting.

The method described above puts the responsibility for determining
measurement uncertainty on the experiment, allowing the model to be an
easy-to-program, deterministic, scalar function that’s equivalent to the
fit function in the more familiar measure-then-fit methods. The downside
of this choice is that the current arrangement doesn’t handle important
cases where the model *is* a probability distribution, such as when the
uncertainty itself is a parameter to be determined.

Making good decisions
~~~~~~~~~~~~~~~~~~~~~

The next important job in the process is figuring out the settings for
the next measurement that will best advance our goals. At least part of
our goal is to make the parameter probability distribution
:math:`p(\theta)` narrow while minimizing cost or time spent. The
challenge is to develop a *utility function* :math:`U(x)` that helps us
to predict and compare the relative benefits of measurements made with
different possible experimental settings :math:`x`.

The mechanism for choosing measurement values hinges on the model,
particularly on the connection between parameter values :math:`\theta`
and measurement results :math:`y`. If we try out different parameter
values as inputs, the model will predict different measurement outcomes.
Intuitively, if we want to constrain the parameter values, it would do
the most good to “pin down” the measurement by selecting the settings
:math:`x` where the predicted :math:`y` has the largest variations due
to parameter variations. The parameter distribution :math:`p(\theta)` is
used to focus attention on the relevant parts of parameter space. By
using draws from :math:`p(\theta)` as test parameter variations,
unlikely parameter sets are avoided, and measurements become
concentrated on refinement of :math:`p(\theta)`.

In broad strokes, our approach to making good decisions about
measurement settings goes like this: 1. For random draws
:math:`\theta_i` of parameters from the distribution :math:`p(\theta)`
Use the model to predict :math:`y_i = y(x,\theta_i)` for every possible
setting :math:`x`. 2. Calculate a measure of the spread in :math:`y_i`
values for every :math:`x` 3. Pick a measurement setting with a large
spread.

Estimate benefits
^^^^^^^^^^^^^^^^^

| To translate such a qualitative argument into code, a good place to
  start is to clarify the meaning of “doing the most good” in refining
  the parameter distribution :math:`p(\theta)`. Usually, the goal in
  determining model parameters is to get results with small uncertainty.
  But :math:`p(\theta)` is a possibly multidimensional distribution and
  parameters may have different units. Fortunately, information theory
  provides the information entropy as a way to quantify the sharpness of
  a probability distribution. The information entropy of a probability
  distribution :math:`p(a)` is defined as
| 

  .. math::  E = -\int da\; p(a)\; \ln[p(a)]. 
| Note that the integrand above is zero for both :math:`p(a) = 1` and
  :math:`p(a)=0`. It’s the intermediate values encountered in a
  spread-out distribution where the information entropy accumulates. For
  common distributions, like rectangular or Gaussian, that have
  characteristic widths :math:`w` the entropy goes like
  :math:`\ln(w) + C`.

| By adopting the information entropy as the measure of
  :math:`p(\theta)` sharpness, it is possible to estimate how much
  entropy change :math:`E`\ (*posterior*) - :math:`E`\ (*prior*) we
  might get for predicted measurement values :math:`y` at different
  settings :math:`x`. Actually, statisticians use something slightly
  different called the Kulback-Liebler divergence:

  .. math::  D^{KL}(y,x) = \int d\theta\; p(\theta |y,x)\ln \left[ \frac{p(\theta | y,x)}{p(\theta)}\right]. 
| In this expression :math:`p(\theta | y,x)` is a speculative parameter
  distribution we would get if we happened to measure a value :math:`y`
  using settings :math:`x`. By itself, :math:`D^{KL}(y,x)` doesn’t work
  as a utility function :math:`U(x)` because it depends on this
  arbitrary possible measurement value :math:`y`. So we need to average
  :math:`D^{KL}`, weighted by the probability of measuring :math:`y`.

  .. math::  U(x) = \int dy \int d\theta\; p(y|x) p(\theta |y,x)\ln \left[ \frac{p(\theta | y,x)}{p(\theta)}\right]. 

Two applications of Bayes rule and rearrangement ensue …

The resulting utility :math:`U(x)` for each potential setting :math:`x`
is the difference between two information entropy-like terms:

1. The information entropy of :math:`p(y|x)`, the distribution of
   measurement values expected at setting :math:`x`. An important
   property of :math:`p(y|x)` doesn’t appear in the notation: that it
   includes likely variations of :math:`\theta.` Explicitly,

   .. math::  p(y|x) = \int d\theta'\; p(\theta') p(y|\theta',x) 

2. In the other term, :math:`p(y|\theta,x)` is the distribution when
   :math:`\theta` and :math:`x` are fixed. The entropy of this
   distribution is averaged over :math:`\theta` values.

   .. math::  \int d\theta\; p(\theta) \int dy\; p(y|\theta,x) \ln [ p(y|\theta, x) ] 

Term 1 is the entropy of the :math:`\theta`-averaged :math:`y`
distribution and Term 2 is the :math:`\theta` average of the entropy of
the :math:`y` distribution. Loosely, Term 1 is a measure of the spread
in :math:`y` values due to both measurement noise and likely parameter
variations, while term 2 is (mostly) just the measurement noise.

Calculation of :math:`U(x)` turns out to be a big computation if you use
the expressions above. We would have to do integrals over all parameters
and all possible measurement outcomes, once for every possible setting.
So, in keeping with our “runs good” philosophy, let’s consider
approximations. What are the risks? All we require of our
decision-making algorithm is that is gives us smart, data-driven
decisions. Is it critical that we make the absolute best measurement
every single time? Probably not. We don’t need precise values, we just
need to know if there are values of :math:`x` where :math:`U(x)` is
comparatively large. Even if we don’t choose the absolute best setting,
a “pretty good” choice will do more good than an uninformed choice. The
only really bad possibility is the risk that the software will run too
slowly to be useful.

Since precise parameter decisions aren’t necessary, consider the case
where all of the distributions are normal (Gaussian). The information
entropy of the normal distribution has a term that goes like
:math:`\ln`\ (width). Term 1 from above is a convolution of the
measurement noise distribution (width = :math:`\sigma_y` and the
distribution of model :math:`y` values (width =
:math:`\sigma_{y,\theta}`) that reflects the connection to the parameter
distribution. A property of normal distributions is that a convolution
of normal distributions is another normal distribution with width =
:math:`\sqrt{\sigma_{y,\theta}^2 + \sigma_y^2}`. Under the assumption of
normal distributions, we now have an approximate utility function

.. math::

    U^*(x) \approx \ln(\sqrt{\sigma_\theta^2 + \sigma_y^2}) - \ln(\sigma_y) 
            = \frac{1}{2}\ln\left[\frac{\sigma_{y,\theta}(x)^2}{\sigma_y(x)^2}+1\right]

| This approximation has some reasonable properties. The dependence on
  :math:`\sigma_{y,\theta}` matches our initial intuition that
  high-utility parameters are those where measurements vary a lot due to
  parameter variations. The dependence on measurement noise
  :math:`\sigma_y` also has an intuitive interpretation: that it’s less
  useful to make measurements at settings :math:`x` where the
  instrumental noise is larger. This approximate utility function is
  also positive, i.e. more data helps narrow a distribution.
| Finally, :math:`U^*(x)`, which is an approximate information entropy
  change, has the property that the parameter distribution width
  :math:`\sigma_\theta` behaves asymptotically like :math:`N^{-1/2}`
  after :math:`N` iterations when noise dominates. At least in one
  dimension,

  .. math::  d \ln(\sigma_\theta) = \frac{1}{2}\frac{\sigma_{y,\theta}(x)^2}{\sigma_y(x)^2}dN. 

  The left side is the approximate entropy change in one iteration
  (:math:`dN = 1`). If the parameter variations are small enough,
  :math:`\sigma_{y,\theta} \approx dy/d\theta\; \sigma_\theta`. Then the
  differential equation implied above has the solution

  .. math::  \sigma_\theta \propto N^{-1/2} 

  which is typical “beating down the noise” behavior of long-term
  averaging.

Estimate the costs
^^^^^^^^^^^^^^^^^^

There are two very important questions that we have left unresolved: 1.
What if some settings are more difficult/time consuming/expensive than
others? 2. When should I quit measuring?

Missing pieces
--------------

A. What if the output of the model is a probability distribution instead
of a number, and the likelihood distribution :math:`p(m|\theta)` is
spread out by more than measurement uncertainty? Quantum mechanics for
example. How do we handle those cases?

B. There is room for computational efficiency improvements. Currently,
the likelihood calculation evaluates the model for every possible
parameter combination. Sequential Monte Carlo or “particle” methods
could speed this up. Also, the Utility function currently evaluates the
model for every possible setting combination times a number of draws
from the parameter distribution. It might be more efficient to use a
conjugate gradient method or even Bayesian optimization (!) to find the
max utility setting with fewer

C. How do we handle situations with multiple measurements at once, like
voltage and current, with different scales, units and uncertainties?
