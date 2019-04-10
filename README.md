
# OptBayesExpt Overview

R. D. McMichael
`rmcmichael@nist.gov`  
National Institute of Standards and Technology  
March 28, 2019


## What is it for?

It's for automating measurements to efficiently determine parameters of a pre-determined model. 
This repository offers an implementation of optimal Bayesian experimental
design (OBED). It is geared toward cases where the "normal" method would determine model parameters by fitting data after measurements are completed.  In contrast, OBED provides an iterative measurement strategy, suggesting measurement settings based on updated measurement data.  By selecting settings that are predicted to yield the most meaningful information, the OBED method makes efficient use of measurement resources. 

Please note that Bayesian optimization is "something completely different." Bayesian optimization has proven to be useful for machine learning. Typically, those are situations where the model isn't known ahead of time.


## What does it do?

The optimal Bayesian experimental design algorithms play the role of an impatient experimenter who monitors data from a running experiment and changes the measurement settings in order to get better, more meaningful data. Note the two
steps here. The first step, looking at the data, is really an act of extracting meaning from the numbers, learning something about the system from the existing measurements. The second step is using that knowledge to improve the measurement strategy.

In the "looking at the data" role, the OBED routines use a user-supplied model function and Bayesian inference to extract and update information about model parameters as new measurement data arrives.  Then, in the decision making role, the OBED routines use the updated parameter knowledge to select settings that have the best chance of refining the parameters.

The most important role is the responsibility of the user, however. As delivered, the BayesOptExpt is ignorant of the world, and it's the user's responsibility to describe the world (OK, just the experiment really.) in terms of a model, reasonable parameters, and reasonable experimental settings. As with any computer program, "the garbage in, garbage out" rule applies.

## What's included?

### Core files
 
The package incorporates a few Python classes that provide the basic
fuctionality.

* **ProbDistFunc_class.py** implements the probability distribution
  function. Its methods are used to define the parameters and their
  respective ranges, perform basic mathematical functions, to supply
  random draws, basic statistics of the distribution, and the
  distribution itself.

* **ExptModel_class.py** provides methods to define the experimental
  settings and to evaluate a model function. The model function itself
  must be provided by the user, preferably by adding it to an instance
  of the BayesOptExpt class.

* **OptBayesExpt.py** provides methods for the learning and deciding steps
  described above. This class inherits all methods and data from both
  ProbDistFunc and ExptModel, and it is the only class that a user will
  need to interact with directly.
  
### Examples

* **sequentialLorentzian.py** is an example script that demonstrates how to incorporate a simple model of a Lorentzian peak into a BayesOptExpt. A simulation takes the place of a real measurement, supplying noisy "measurement results" that the BayesOptExpt uses to locate and focus on a randomly placed peak.  Features include:
  * Step-by-step documentation
  * multiple "fit" parameters

* **pipulse.py** uses the OptBayesExpt class to simultaneously find the resonance frequency of a spin and the Rabi frequency of an rf field by "measuring" a spin's response to rf pulses.  Features include
  * multiple (two!) settings
  * Including existing information in a non-default _prior_ parameter distribution.

### Add-ons

Extra code for extra functionality

* **OBETCP.py** provides way to use optimized Bayesian experimental design
  software with programs written in other languages. OBETCP incorporates
  an OptBayesExpt class with a server that communicates with external
  programs via TCP socket connections. Configuration commands,
  measurement settings and measurement data are sent as json-ified
  strings over the TCP sockets.
  
* **LabView/OBE_ODMR_demo.vi** demonstrates how OBETCP.py can be used from
  Labview. Like the sequentialLorentzian.py example, it's about finding
  and measuring a Lorentzian peak.

## What's next?

* **Docs/manual.html** and 
* **Docs/manual.ipynb** provides a gallery of examples and describes the statistical methods used in OptBayesExpt.


* **Docs/sequentialLorentzian.html** and 
* **Docs/sequentialLorentzian.ipynb** provide a gentle introduction to the OptBayesExpt software in `html` and jupyter notebook `ipynb` formats.  They include explanations of the setup and use of `OptBayesExpt` in an example simulated "measurement" of a Lorentzian peak.  The `ipynb` file is live code that can be run from inside the jupyter notebook environment. Both formats are derived from **sequentialLorentzian.py**.


