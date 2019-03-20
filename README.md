
This repo offers an implementation of optimized Bayesian experimental design (OBED).  It is for
cases where we have confidence in a mathematical model of a system, and we merely want to determine
parameters of the model.  Least-squares fitting is the usual approach in this situation. Examples
include linear models with slope and intercept parameters, peaks with positions, heights, and
widths, and also oscillations with amplitudes and frequencies.  The real benefit of OBED methods is
that they direct measurements toward settings that have the best chance of making our parameter
estimates more precise.  This feature is very heplful in situations where the measurements are
expensive.

Note that Bayesian optimization is "something completely different" that has proven to be useful for
machine learning.  Typically, those are situations where the system isn't well understood.

Using a common measurement strategy, we might measure "y" for a series of "x" values and then, when
the data's all collected, do least-squares fit that determines best-fit parameters for a model
function.  Or maybe we look at the data and decide to run the measurements again, only with "x"
values more focused on some region of interest.

The Bayesian Optimized Experimental Design plays the role of the experimenter who looks at the data
and decides to change the measurement settings in order to get better, more meaningful data.  Note
the two steps here.  The first step, looking at the data, is really an act of extracting meaning
from the numbers, learning something about the system from the existing measurements.  The second
step is using that knowledge to decide what to do next.  But instead of waiting for the measurements
 to finish, the OBED algorithms evaluate the data constantly and make measurement setting decisions
 "on the fly."

In Bayesian optimized experimental design, our knowledge of the system parameters is represented by
a joint probability distribution function.  The learning step is performed using Bayesian inference
to refine this distribution for each measurement value.  Narrow distributions correspond to
precisely "measured" parameters.

The decision step uses the model function and the the parameter distribution to suggest settings for
the next measurement.  This step looks at fluctuations in the model's predicted measurement values
when the parameters are drawn randomly from their probability distribution.  A good choice for the
next measurement will "pin down" fluctuations where they are large, so we choose settings where the
model predicts large fluctuations.

The most important step is left to the user, however.  As delivered, the BayesOptExpt is ignorant of
the world, and it's the user's responsibility to describe the world in terms of a model, reasonable
parameters and reasonable experimental settings.  As with any computer program, "the garbage in,
garbage out" rule applies.

The package incorporates a few Python classes that provide the basic fuctionality.

    - ProbDistFunc_class.py implements the probability distribution function.  Its methods are used
      to define the parameters and their respective ranges, perform basic mathematical functions,
      to supply random draws, basic statistics of the distribution, and the distribution itself.

    - ExptModel_class.py provides methods to define the experimental settings and to evaluate a
      model function.  The model function itself must be provided by the user, preferably by adding
      it to an instance of the BayesOptExpt class.

    - OptBayesExpt.py provides methods for the learning and deciding steps described above.  This
      class inherits all methods and data from both ProbDistFunc and ExptModel, and it is the only
      class that a user will need to interact with directly.

    - sequentialLorentzian.py  is an example script that demonstrates how to incorporate a simple
      model of a Lorentzian peak into a BayesOptExpt.  A simulation takes the place of a real
      measurement, supplying noisy "measurement results" that the BayesOptExpt uses to locate and
      focus on a randomly placed peak.




