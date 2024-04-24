
# OptBayesExpt Overview

R. D. McMichael
`rmcmichael@nist.gov`  
National Institute of Standards and Technology  
revision: April 24, 2024

## What is it for?

Optimal Bayesian Experiment Design is for making smart setting choices in
measurements. The `optbayesexpt` python package is for cases with

 - a known parametric model, i.e. an equation that relates unknown parameters
   and experimental settings to measurement predictions. Fitting functions used
   in least-squares fitting are good examples of parametric models.
 - an experiment (possibly computational) that uses a set-measure-repeat
   sequence with opportunities to change settings between measurements.

The benefit of these methods is that they choose settings
that have a good chance of making the parameter estimates more precise.
This feature is very helpful in situations where the measurements are
expensive.

It is not primarily designed for analyzing existing data, but some of the
code could be used for Bayesian inference of parameter values.

Note that *Bayesian optimization* addresses a different problem: finding a
maximum or minimum of an unknown function.

## What does it do?

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

## What's next?

Documentation is offered at
[this project's web page](https://pages.nist.gov/optbayesexpt).  The website
includes a manual, a quick start guide, a gallery of demo programs, and the
API documentation.

## Legal stuff

### Disclaimer
Certain commercial firms and trade names are identified in this document in
order to specify the installation and usage procedures adequately. Such
identification is not intended to imply recommendation or endorsement by the
[National Institute of Standards and Technology](http://www.nist.gov), nor
is it intended to imply that related products are necessarily the best
available for the purpose.

### Terms of Use
This software was developed by employees of the National Institute of
Standards and Technology (NIST), an agency of the Federal
Government and is being made available as a public service. Pursuant to
title 17 United States Code Section 105, works of NIST employees are not
subject to copyright protection in the United States. This software may be
subject to foreign copyright. Permission in the United States and in
foreign countries, to the extent that NIST may hold copyright, to use,
copy, modify, create derivative works, and distribute this software and its
documentation without fee is hereby granted on a non-exclusive basis,
provided that this notice and disclaimer of warranty appears in all copies.

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE. IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING,
BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
