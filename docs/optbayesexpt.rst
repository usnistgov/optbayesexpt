optbayesexpt API
================

Overview
--------

* :obj:`OptBayesExpt` is the core class that performs Bayesian inference
  and selects measurement settings. Typically, it is the only
  class that a user will need to interact with directly.

* :obj:`OptBayesExptNoiseParameter` is similar to :obj:`OptBayesExpt` but
  is designed for cases where the measurement uncertainty is a parameter to
  be estimated.

* :obj:`ParticlePDF` is inherited by :obj:`OptBayesExpt` to handle the
  duties of a probability distribution functions.

* :obj:`OBE_Server` class provides communication with other processes
  through a mini-language of label-value commands.

* :obj:`Socket` class is inherited by :obj:`Server` to handle TCP connections
  and message encoding/decoding.

* The **obe_utils.py** file provides

    * A :obj:`MeasurementSimulator` class that uses "true value" parameters
      and added noise to simulate experimental outputs.
    * For post-processing, a :obj:`trace_sort()` function sorts measurement
      data by measurement setting and combines all measurements with settings
      in common.
    * A :obj:`differential_entropy()` function to calculate information entropy from
      samples of a distribution.


OptBayesExpt class
------------------

.. automodule:: optbayesexpt.obe_base
   :members:
   :undoc-members:
   :show-inheritance:

OptBayesExptNoiseParam class
----------------------------

.. automodule:: optbayesexpt.obe_noiseparam
   :members:
   :undoc-members:
   :show-inheritance:

OBE_Server class
----------------

.. automodule:: optbayesexpt.obe_server
   :members:
   :undoc-members:
   :show-inheritance:

OBE_Socket class
----------------

.. automodule:: optbayesexpt.obe_socket
   :members:
   :undoc-members:
   :show-inheritance:

ParticlePDF class
-----------------

.. automodule:: optbayesexpt.particlepdf
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: optbayesexpt.obe_utils
   :members:
   :undoc-members:
   :show-inheritance:
