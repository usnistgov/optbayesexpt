
# OptBayesExpt Overview

R. D. McMichael
`rmcmichael@nist.gov`  
National Institute of Standards and Technology  
May 6, 2019


## What is it for?

It's for making smarter measurements. This repository offers an implementation of optimized Bayesian experimental design (OBED). It is an experimental control strategy that chooses measurement settings based on accumulated data.  It is for cases with

 - an experiment (possibly computational) 
   - that yields measurements and uncertainty estimates, 
   - and that can be controlled on the fly by one or more experimental settings, and
 - a parametric model, i.e. an equation that relates unknown parameters and experimental settings to measurement predictions.  If you would normally fit a function to the data to get fit parameters, that's the parametric model.
 
The real benefit of OBED methods is that they direct measurements toward settings that have the best chance of making our parameter estimates more precise. This feature is very heplful in situations where the measurements are expensive.

It is not for fitting existing data, but if you're thinking about making more measurements, you might be interested.

Note that Bayesian optimization is 'something completely different' that has proven to be useful for machine learning. Typically, those are situations where the system isn't well understood.

## What does it do?

The optimal Bayesian experimental design algorithms play the role of an impatient experimenter who monitors data from a running experiment and changes the measurement settings in order to get better, more meaningful data. Note the two
steps here. The first step, looking at the data, is really an act of extracting meaning from the numbers, learning something about the system from the existing measurements. The second step is using that knowledge to improve the measurement strategy.

In the "looking at the data" role, the OBED routines use a user-supplied model function and Bayesian inference to extract and update information about model parameters as new measurement data arrives.  Then, in the decision making role, the OBED routines use the updated parameter knowledge to select settings that have the best chance of refining the parameters.

The most important role is the responsibility of the user, however. As delivered, the BayesOptExpt is ignorant of the world, and it's the user's responsibility to describe the world (OK, just the experiment really.) in terms of a model, reasonable parameters, and reasonable experimental settings. As with most computer programs, "the garbage in, garbage out" rule applies.

## What's included?

### Core files
 
The package incorporates a few Python classes that provide the basic
fuctionality.

* **OptBayesExpt.py** provides methods for the learning and deciding steps described above. This class inherits all methods and data from both ProbDistFunc and ExptModel, and it is the only class that a user will need to interact with directly.  Manual: [[notebook]](Docs/OptBayesExpt.ipynb)[[html]](Docs/OptBayesExpt.html)
  
* **ProbDistFunc_class.py** implements the probability distribution function. Its methods are used to define the parameters and their respective ranges, perform basic mathematical functions, to supply random draws, basic statistics of the distribution, and the distribution itself. Manual: [[notebook]](Docs/ProbDistFunc_class.ipynb)[[html]](Docs/ProbDistFunc_class.html)

* **ExptModel_class.py** provides methods to define the experimental settings and to evaluate a model function. The model function itself must be provided by the user, preferably by adding it to an instance of the BayesOptExpt class. Manual: [[notebook]](Docs/ExptModel_class.ipynb)[[html]](Docs/ExptModel_class.html)
  
* **OBETCP.py** provides way to use this software with programs written in other languages. OBETCP incorporates
  an OptBayesExpt class with a server that communicates with external programs via TCP socket connections. Configuration commands, measurement settings and measurement data are sent as JSON-formatted strings over the TCP sockets.
  
## What's next?

### Tutorials

* **Docs/sequentialLorentzian.ipynb** [[notebook]](Docs/sequentialLorentzian.ipynb) [[html]](Docs/sequentialLorentzian.html) provides a tutorial introduction to the OptBayesExpt software including setup and use of `OptBayesExpt` in a simulated "measurement" of a Lorentzian peak.  The `ipynb` file is live code that can be run from inside the jupyter notebook environment.  See also **sequentialLorentzian.py**.

* **Docs/manual.ipynb** [[notebook]](Docs/manual.html)[[html]](Docs/manual.html) outlines the "if it works good, it is good" philosophy of the project and provides a tutorial-level description of the theory behind optimal Bayesian experimental design.

### Demos

A brief discussion of these demos is included in the manual. 
[[notebook]](Docs/manual.ipynb) [[html]](Docs/manual.html)

* **demoLorentzian.py**  demonstrates how to incorporate a simple model into a BayesOptExpt. A simulation takes the place of a real measurement, supplying noisy "measurement results" that the BayesOptExpt uses to locate and measure a randomly placed Lorentzian peak.  One setting, several model parameters.  Also see **Docs/sequentialLorentzian.ipynb** [[notebook]](Docs/sequentialLorentzian.ipynb)[[html]](Demos/sequentialLorentzian.html) is a jupyter notebook with a step-by-step walk-through of the `sequentialLorentzian.py` code.  

* **demoLorentzian2.py** uses the Lorentzian peak experimenttal model to demonstrate 10 $\times$ improved measurement efficiency of BayesOptExpt relative to an average & fit method.

* **pipulse.py** is a slightly more complicated demo featuring multiple (two!) experimental settings and the process for including pre-existing information, a _prior_.

* **slopeIntercept.py** demonstrates measurement of straight lines, $y = m x + B$.  The demonstration presents options in the decision-making part of the code.

* **LabView/OBE_ODMR_demo.vi** demonstrates how OBETCP.py can be used from Labview. Like the sequentialLorentzian.py example, it's about finding and measuring a Lorentzian peak.

## Legal stuff

### Disclaimer
Certain commercial firms and trade names are identified in this document in order to specify the installation and usage procedures adequately. Such identification is not intended to imply recommendation or endorsement by the [National Institute of Standards and Technology](http://www.nist.gov), nor is it intended to imply that related products are necessarily the best available for the purpose.

### Terms of Use
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal
Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States. This software may be subject to foreign copyright. Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use,
copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE. IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING,
BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.


```python

```
