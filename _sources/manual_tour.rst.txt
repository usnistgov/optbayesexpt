Guided Tour
-----------

Where the section above treats the theory of Bayesian experimental design,
this section provides an introduction to the algorithms and methods in the
``OptBayesExpt``
class that perform the "learn fast" and "make good decisions" tasks.
An additional subsection describes how the parameter probability
distribution is implemented.

Learn fast routine
~~~~~~~~~~~~~~~~~~

.. sidebar:: Measurement inputs

    .. image:: _images/obe_pdf_update_flowchart.png
       :scale: 35 %

To input measurement data, the user script invokes
``pdf_update(measurement_result)``.  The events that follow are illustrated
in the sidebar.  ``OptBaesExpt`` requires the user to provide a
``measurement_result`` that represents a complete measurement record with
the settings, the value and the uncertainty. The settings are required since
actual settings may differ from suggested settings. The settings are used to
evaluate the model function for every parameter sample in the
parameter probability distribution to produce ``y_model_data``. For each
parameter sample the difference between the model value and the reported
measurement value is used to calculate the likelihood of that parameter
sample, assuming a Gaussian noise distribution.

The uncertainty deserves some attention. By requiring the measurement
uncertainty, OptBayesExpt places the burden of determining the uncertainty
on the experimenter. Bluntly, the reason for this requirement is that it makes
the programming easier. But often, the noise is one of the things one would
like to learn from an experiment.

To include the raw measurement uncertainty as an unknown, it is
convenient to create a child class that inherits from OptBayesExpt.
``ChildClass`` may include the uncertainty alongside the
actual model parameters in the ``parameter_samples`` tuple, as suggested by
the dashed arrow in the sidebar. In this arrangement, the
uncertainty parameter is not used in the
model function, but it does enter the likelihood calculation.  After all,
some values of uncertainty explain the data better than others. For examples of
this approach, see

    - ``demos/obe_line_plus_noise.py``
    - ``demos/obe_lockin.py``
    - ``demos/obe_sweeper.py``

``ParticlePdf.bayesian_update()`` adjusts the parameter distribution
(including the uncertainty part) based on the likelihood.

Make good decisions routine
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. sidebar:: Setting selection

    .. image:: _images/utility.png
       :scale: 35 %

Measurement settings are selected by ``opt_setting()`` and ``good_setting``,
and both of these rely on ``utility()``.  The sidebar illustrates the inner
workings of ``utility()``, where the most time-consuming part is
``yvar_from_parameter_draws()``.  For each of ``N_DRAWS`` samples from
the parameter distribution the model function is evaluated for every
possible setting combination, yielding a set of outputs for each setting.  The
variance of these outputs is compared to the model experimental noise which
is a constant by default.  The examples listed above for unknown
uncertainties include the mean of the variance as a model of experimental
noise.

In the deoniminator, the cost of a potential experiment may depend on
settings, or predicted value or distance to the next setting.  There are
many possibilities.  In ``OptBayesExpt`` the cost is constant by default but
customized cost models can be programmed into child classes.
In ``demos/obe_lockin.py`` for example, the cost model includes the additional
cost of a settling time if the setting is changed from its current value.  In
``demos/obe_sweeper.py``, the cost model includes a cost of setting up a
sweep plus a cost proportional to the length of the sweep.

Once the utility is computed, ``opt_setting()`` and ``good_setting()``
offer different strategies for selecting settings for the next measurement.
The ``opt_setting()`` method implements a "greedy" strategy, always
selecting the setting that generates the maximum utility.  The
resulting measurements tend to cluster strongly around a few settings. The
``good_setting()`` method uses the utility as a weighting function for a
random setting selection. To tilt the odds toward selecting settings
with higher utility, the contrast between high and low utility is stretched
by raising utility to a power, ``pickiness``.  The contrasting behavior of
``opt_settting()`` and ``good_setting()`` with different ``pickiness``
values are illustrated by ``demos/line_plus_noise.py``

Probability distribution class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. sidebar:: Probability distributions

    .. image:: _images/particleFilter.png
       :scale: 50 %

The probability distribution over the model parameters is at the core of
``OptBayesExpt``.  As of version 1.0, the optbayesexpt package uses the
``particlePDF`` class to implement the distribution function for *N*
parameters as a cloud or swarm of particles in *N*-dimensional parameter
space.  This approach has been given several names, but "sequential Monte
Carlo" seems to be gaining traction.  Each point in the cloud has
coordinates corresponding to parameter values, and also a weight, normalized
so that the sum of the weights is 1. The
density of particles in a small region and the weights of those particles
together provide a flexible representation of the probability density.  The
figure in the sidebar illustrates how this works for a 1-D normal (Gaussian)
distribution. Panel (a) plots 100 draws from the normal distribution, each
with a weight of 0.01. The probability distribution is represented only
by the density of points in this case. Panel (b) represents the same normal
distribution using only the weights.  The points are 100 draws from a
*uniform* distribution, but the weights represent the normal distribution.

In the learning fast stage, the distribution is modified based on new data.
The likelihood is calculated for every point in
the distribution cloud, and then ParticlePdf.bayesian_update() multiplies
the particle's weight by its likelihood.  So the distribution is easily
modified by adjusting weight values.

In the decision making stage, random draws of parameters from the
probability distribution are used to calculate utility.  Draws are made by
random selection of points with probability determined by weights.  In panel
(a) values at the center are more likely to be chosen because the density is
higher there.  In panel (b), values at the center are more likely to be
chosen because the weights are higher there.

``ParticlePdf`` also performs a self-maintenance function, ``resample()``.
As the incoming data is used to modify the
distribution, some regions of parameter space may develop very low
probabilities. The points near the ends of the plot in panel (b) illustrate
the issue. These low-weight points will almost never be chosen in
random draws, but they consume computational resources.  In resampling, *N*
draws are taken from an *N*-particle distribution, so some high-weight
particles will be chosen more than once and low-weight particles may not be
chosen at all. Each particle is then given a small random nudge to separate
copies of the same original particle, the cloud of points is shrunk
slightly to compensate for the one-step diffusion, and the weight is divided
evenly between particles as shown in panel (c). Resampling relocates
low-weight particles to higher-weight neighborhoods so that as measurements
narrow the distributions of parameter values, the representation adapts.
