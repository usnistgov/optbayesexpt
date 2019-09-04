
Quick Start
===========

Installation
------------

Clone the ``optbayesexpt`` repository from github. Starting in an empty
diretory,

::

     git clone https://www.github.com/usnistgov/optbayesexpt .
     python setup.py build
     python setup.py install

Scripting
---------

Setting up
~~~~~~~~~~

.. code:: ipython3

    import numpy as np
    from optbayesexpt import OptBayesExpt
    
    myOBE = OptBayesExpt()          # initiate a class

Specify the experimental settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, we specify a setting called ``knob`` to vary between 1 and
10. Continuous variables must be discretized, and here we set 100
values.

.. code:: ipython3

    knob = np.linspace(1, 10, 100)
    myOBE.sets = (knob, ) 

``OptBayesExpt.sets`` must be a tuple of numerical lists as
``(list_1, [list_2], ...)``.

Specify parameter space
~~~~~~~~~~~~~~~~~~~~~~~

Suppose there’s a ``foo_phase`` and a ``bar_coeff`` that the
measurements are to determine. Specify ranges and discretization.
Importantly, specifying a parameter range asserts that there is zero
probability that the true value will be outside the range.

.. code:: ipython3

    foo_phase = np.arange(-np.pi/2, np.pi/2, 180)
    bar_coeff = np.arange(1, 10, 50)
    myOBE.pars = (foo_phase, bar_coeff)

``OptBayesExpt.pars`` must be a tuple of numerical lists as
``(list_a, [list_b], ...)``.

Specify constants
~~~~~~~~~~~~~~~~~

An optional definition for parameters that are held constant for the
duration of an experiment, but that might change between runs.

.. code:: ipython3

    temperature = 19   # degrees C
    myOBE.cons = (temperature, )

OptBayesExpt.cons must be a tuple of numerical values.

Configure
~~~~~~~~~

.. code:: ipython3

    myOBE.config()

The ``config`` function creates numpy arrays from the ``sets`` and
``pars`` tuples, For :math:`n` settings in ``sets``, there will be
:math:`n` of :math:`n`-dimensional arrays to describe all possible
settings. For :math:`m` parameters, the probability distribution will be
an :math:`m`-dimensional array. Array sizes in each dimension correspond
to lengths of arrays in ``sets`` and ``pars``.

Specify the model
~~~~~~~~~~~~~~~~~

BayesOptExpt requires a ``model_function()`` to be supplied by the user.
BayesOptExpt will call this function with three tuple arguments.
OptBayesExpt will iterate over all possible combinations of setting
values, and separately, over all possible parameters.

.. code:: ipython3

    def my_model_function(settings, parameters, constants):
        """
        User-supplied code
        :param settings:   either a tuple of setting values 
                             -or- a tuple of 1D numpy arrays of settings
        :param parameters: either a tuple of numpy arrays of parameters 
                             -or- a tuple of parameter values
        :param contants:   a tuple of constants
        :return:           a measurement prediction
        """
        # user-defined code
        # ...
    
    # Incorporate the model into myOBE
    myOBE.model_function = my_model_function

Using ``numpy`` arrays, iterations over array elements are handled
automatically using broadcasting. See the Numpy User Guide for
information on
`broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__.

Running
-------

BayesOptExpt participates at two stages in the measurement loop as shown
in the following pseudocode.

.. code:: ipython3

    while still_measuring:
        
        # myOBE picks settings - there's a choice of methods
        # settings = myOBE.opt_setting()
        settings = myOBE.good_setting(pickiness=a_value_between_1_and_10)
        
        # The experiment makes a measurement and returns results
        new_result = measurement_results_determined_using(settings)
        
        # myOBE uses the new results to update the parameter 
        # probability distribution function (pdf)
        myOBE.pdf_update(settings, new_result, uncertainties)
    
    # end while loop
    
    # get the results
    parameter_distribution = myOBE.get_pdf()
