Theory of operation
-------------------
The optimal Bayes experimental design method incorporates two main jobs,
which we can describe as “learning early” and “making good decisions”

Learning early
~~~~~~~~~~~~~~

By interpreting measurement data as soon as it becomes available, sequential Bayesian
experimental design gains a critical advantage over the traditional measure-then-fit design.
With a measure-then-fit strategy, we get information about parameters only at the very end of the
process, after all the measurements and fitting are done. In contrast, the optimal Bayesian
experimental design method updates our parameter knowledge with each measurement result, so
that information-based decisions can be made as data is collected.

The process of digesting new data is Bayesian inference, which frames
parameter knowledge in terms of a probability distribution :math:`p(\theta)` for an array of
parameters :math:`\theta = [ \theta_0, \theta_1, ...]`. The familiar notation :math:`a\pm \sigma`
is often a shorthand description of a Gaussian probability distribution. A broad distribution
:math:`p(\theta)` corresponds to large uncertainty, and if :math:`p(\theta)` is a narrow
distribution, the uncertainty is small.

When new measurement results :math:`m` are taken in to account, there will be a new,
revised probability distribution :math:`p(\theta|m)`. The vertical bar in the notation
:math:`p(\theta|m)` indicates a conditional probability, so :math:`p(\theta|m)` is the
distribution of :math:`\theta` values *given* :math:`m`.

Bayes’ rule provides

.. math::  p(\theta|m) = \frac{p(m|\theta) p(\theta)}{p(m)}.

All of the terms here have technical names. The left side is the
*posterior* distribution, i.e. the distribution of parameters
:math:`\theta` after we include :math:`m`. On the right, distribution
:math:`p(\theta)` is the *prior*, representing what we knew about the
parameters :math:`\theta` before the measurement. In the denominator,
:math:`p(m)` is called the *evidence*, but because it has no :math:`\theta`
dependence, it functions just a normalizing constant in this situation.
As wrong as this sounds, we will ignore the *evidence*.

The term that requires attention is in the numerator; :math:`p(m|\theta)` is called the
*likelihood*. It’s the probability of getting measurement :math:`m`
given variable parameter values :math:`\theta`.  Less formally, the *likelihood* answers the
question: "How well does the model explain the measured value
:math:`m`, when the model uses different parameter values :math:`\theta`?"

In practice, :math:`m_i` will be a fixed measurement result to “plug in” for :math:`m`. It’s
important to keep sight of the fact that :math:`p(m_i|\theta)` is still a function of
theta. Conceptually, we can try out different parameter values in our
model to produce a variety of measurement predictions. Some parameter
values (the more likely ones) will produce model values closer to
:math:`m_i` and for other parameters, (the less likely ones), model
model value will be further away.

The ``optbayesexpt.OptBayesExpt`` class requires the user to report a measurement record
:math:`m_i` that includes the measurement settings :math:`x_i`, the “value” :math:`y_i`, and
uncertainty :math:`\sigma_i`. Together, :math:`y_i` and :math:`\sigma_i` are more than fixed
numbers; they
are shorthand for a probability that a noise-free measurement would yield a value :math:`y`.
:math:`y` given a mean value :math:`y_i`. If this distribution is symmetric, like a Gaussian, for
example, then :math:`p(y|y_i, \sigma_i) = p(y_i|y, \sigma_i)`, the probability of measuring
:math:`y_i` given a mean value :math:`y` that’s provided by the experimental model :math:`y=y
(x_i,\theta)`.

.. math::  p(m_i|\theta) = \frac{1}{\sqrt{2\pi}\sigma_i}
            \exp\left[-\frac{[y_i - y(x_i, \theta)]^2 }{ 2\sigma_i^2 } \right].

With this *likelihood*, Bayes theorem provides the updated  :math:`p(\theta|m_i)`.
Then, another measurement :math:`m_j` can update :math:`p(\theta|m_i)` to
:math:`p(\theta|m_j, m_i, \ldots)` and so on. In order to keep the
notation readable, we’ll adopt a convention that :math:`p(\theta)`
always represents the most up-to-date parameter distribution that we
have.

This approach assumes that our model function :math:`y(x, \theta)` is a good description of our
system, and that the measurement noise is Gaussian with standard deviation
:math:`\sigma_i`. On one hand we have to admit that these assumptions don’t allow us to
address all important cases. On the other hand, these are the same
assumptions we often make in doing least-squares curve fitting.

The method described above puts the responsibility for determining
measurement uncertainty on the measurement reporter, but as an experimental result, uncertainty is
generally a measurement output, not an input.  If uncertainty is a parameter to be determined, it
enters the process through the likelihood function given above, but it is not part of the model
function :math:`y(x_i, \theta)`. See ``demos/line_plus_noise.py`` for an example.

Making good decisions
~~~~~~~~~~~~~~~~~~~~~

The next important job in the process is figuring out good measurement
settings. The goal is to make the parameter probability distribution
:math:`p(\theta)` narrow while minimizing cost. More formally, the
challenge is to develop a *utility function* :math:`U(x)` that helps us
to predict and compare the relative benefits of measurements made with
different possible experimental settings :math:`x`.

Qualitatively, the mechanism for choosing measurement values hinges on the model's connection
between parameter values :math:`\theta` and measurement results :math:`y`.
Consider a sampling of several sets of parameter values :math:`\theta_i`.  With these parameter
sets, the model can produce a collection of output curves :math:`y_i(x)`, and generally these
curves will be closer together for some settings :math:`x` and further apart for others.
Intuitively, little would be learned by measuring at :math:`x` values where the curves
agree.  Instead, it would do the most good to “pin down” the results with a measurement at an
:math:`x` where the predicted :math:`y_i(x)` curves disagree.

By drawing samples from the updated parameter distribution :math:`p(\theta)` the mechanism above
focuses attention on the relevant parts of parameter space, and through the model to relevant
settings. Or, stated slightly differently, using an updated parameter distribution helps to avoid
wasting measurement resources on low-impact measurements.

Estimate benefits
^^^^^^^^^^^^^^^^^

To translate such a qualitative argument into code, "doing the most good"
must be defined more precisely in terms of desired changes in
the parameter distribution :math:`p(\theta)`. Usually, the goal in
determining model parameters is to get unambiguous results with small uncertainty. The *information
entropy* provides a measure of something like a probability distribution. The information entropy
of a probability distribution :math:`p(a)` is defined as

.. math::  E = -\int da\; p(a)\; \ln[p(a)].

Note that the integrand above is zero for both :math:`p(a) = 1` and
:math:`p(a)=0`. It’s the intermediate values encountered in a
spread-out distribution where the information entropy accumulates. For
common distributions, like rectangular or Gaussian, that have
characteristic widths :math:`w` the entropy goes like :math:`\ln(w) + C` with :math:`C` values
depending on the shape of the distribution.

Now we can define "doing the most good" in terms of how much
entropy change :math:`E`\ (*posterior*) - :math:`E`\ (*prior*) we
might get for predicted measurement values :math:`y` at different
settings :math:`x`. Actually, we use something slightly
different called the Kulback-Liebler divergence:

.. math::  D^{KL}(y,x) = \int d\theta\; p(\theta |y,x)
    \ln \left[ \frac{p(\theta | y,x)}{p(\theta)}\right].

In this expression :math:`p(\theta | y,x)` is a speculative parameter
distribution we would get if we happened to measure a value :math:`y`
using settings :math:`x`. By itself, :math:`D^{KL}(y,x)` doesn’t work
as a utility function :math:`U(x)` because it depends on this
arbitrary possible measurement value :math:`y`. So we need to average
:math:`D^{KL}`, weighted by the probability of measuring :math:`y`.

.. math::  U(x) \propto \int dy \int d\theta\; p(y|x) p(\theta |y,x)
    \ln \left[ \frac{p(\theta | y,x)}{p(\theta)}\right].

With two applications of Bayes rule and some rearrangement this expression for
:math:`U(x)` can be rewritten as the difference between two information entropy-like terms:

:Term 1: The information entropy of :math:`p(y|x)`, the distribution of
        measurement values expected at setting :math:`x`. Importantly this distribution
        includes likely variations of :math:`\theta.` Explicitly,

        .. math::  p(y|x) = \int d\theta'\; p(\theta') p(y|\theta',x)

        Qualitatively, this term is the information entropy of the predicted measurement values
        including both measurement noise and the effects of parameter uncertainty.

:Term 2: The other term is the information entropy of :math:`p(y|\theta,x)` the measurement value
        distribution when :math:`\theta` and :math:`x` are fixed, i.e. the entropy of just the
        measurement noise distribution. The entropy of this distribution is averaged over
        :math:`\theta` values.

        .. math::  \int d\theta\; p(\theta) \int dy\; p(y|\theta,x) \ln [ p(y|\theta, x) ]

Term 1 is the entropy of the :math:`\theta`-averaged :math:`y`
distribution and Term 2 is the :math:`\theta` average of the entropy of
the :math:`y` distribution. Loosely, Term 1 is a measure of the spread
in :math:`y` values due to both measurement noise and likely parameter
variations, while term 2 is (mostly) just the measurement noise.

An accurate calculation of :math:`U(x)` is a big job, requiring integrals over all
parameter space and also all possible measurement outcomes, once for every possible setting.
Fortunately, in keeping with the “runs good” project philosophy, accuracy is not required.
All we require of an approximate utility function is that provides a guide for non-stupid
decisions. It is not critical that the absolute best measurement choice is made
every single time.  It is only necessary to know if there are values of :math:`x`
where :math:`U (x)` is large compared to other
:math:`x`.  Even if we don’t choose the absolute best setting,
a “pretty good” choice will do more good than an uninformed choice.

Consider an approximate calculation of :math:`U^*(x)`
where all of the distributions are assumed to be normal (Gaussian). The information
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

    U^*(x) \propto \approx \ln(\sqrt{\sigma_\theta^2 + \sigma_y^2}) - \ln(\sigma_y)
            = \frac{1}{2}\ln\left[\frac{\sigma_{y,\theta}(x)^2}{\sigma_y(x)^2}+1\right]

This approximation has some reasonable properties. The dependence on
:math:`\sigma_{y,\theta}` matches our initial intuition that
high-utility parameters are those where measurements vary a lot due to
parameter variations. The dependence on measurement noise
:math:`\sigma_y` also has an intuitive interpretation: that it’s less
useful to make measurements at settings :math:`x` where the
instrumental noise is larger. This approximate utility function is
also positive, i.e. more data helps narrow a distribution.

Estimate the costs
^^^^^^^^^^^^^^^^^^

The utility as described above focuses on the information entropy change,
but if the question is about making efficient measurements, the cost of the
measurements is fully half of the problem.  In the simplest situations, the
time spent measuring is the only cost.  In more "interesting" situations,
there may be a cost associated with changing settings, the cost of a
measurement may depend on the likelihood of damage. So in contrast to the
general approach to predicting information entropy change, the cost closely
associated with local conditions.
