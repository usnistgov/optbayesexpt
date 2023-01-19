Quick Start
===========

Software Requirements
---------------------

python3.X with the following packages:
    - numpy
    - numba is recommended for faster execution
    - scipy is recommended
    - matplotlib.pyplot (for demos)
    - pytest (for testing)

Download
--------

Using ``git``,
    1. Create an empty directory/folder to hold the source files.  The
       directory name isn't important, but we'll call it ``obe_source`` here.
    2. Move into the source directory
        - ``> cd obe_source``  on linux
        - On Windows, click into the new folder and then File --> Open
          Windows Power Shell

    3. From the command line,
        - ``> git clone https://github.com/usnistgov/optbayesexpt.git .``

       Note the trailing '.'

Using zip,

    1. Point your browser to the optbayesexpt github page,
       https://github.com/usnistgov/optbayesexpt.
    2. From the green Code button, select Download zip from the drop down
       menu.
    3. Unzip the optbayesexpt-master.zip file.
    4. Move into the unzipped directory/folder. This is your source directory.
        - ``> cd optbayesexpt-master``  on linux
        - On Windows, click into unipped folder, e.g. ``optbayesexpt-master``
          and then File --> Open Windows Power Shell.

Installation
------------

Now install the optbayesexpt modules from the command line by one of these
methods:

::

    > python -m pip install .

--or--

::

     > python setup.py build
     > python setup.py install

Some systems may use ``python3`` instead of ``python``.

Testing
-------

Optional testing for basic functionality requires the
pytest module. From the source directory:

::

    > python -m pytest

Note that ``test_zinference.py`` script is a statistical test, and is not
expected to pass every time.  An error message of the form ``"AssertionError:
We ran 100 inference tests ..."`` does not necessarily indicate a problem.

Scripting
---------

The ``*.py`` files in the ``demos/`` folder offer several examples of working
scripts that may be adapted for different applications. This *Quick Start*
offers an overview of the essentials.

Setting up
~~~~~~~~~~

In the python script, import the necessary modules.

::

    import numpy as np
    from optbayesexpt import OptBayesExpt

"Out of the box," OptBayesExpt is ignorant, and
it must be educated about the little universe where it will do its work.
The "Specify ..." sections below show how this education process is done.

Specify the model
^^^^^^^^^^^^^^^^^

OPtBayesExpt requires a ``model_function()`` to describe the relationships
between experimental controls, parameters and measurement results. The
model_function is trusted to tell the truth, the whole truth and nothing but
the truth about how the experiment will behave, so it's important for the
model function to allow for any real but extraneous "features" of the data,
e.g. sloped backgrounds, extra peaks, etc.

The model function is required to accept three tuples as arguments:
``settings``, ``parameters`` and ``constants``, representing
experimental controls, parameters to be determined, and infrerquently changed
values. When this function is called, either ``settings`` or ``parameters``
will be a tuple of numpy arrays.  The other two arguments will be tuples of
floats.

::

 def my_model_function(settings, parameters, constants):
    """Example model function

    The ``(settings, parameters, constants)`` argument structure is required
    Args:
        settings (tuple or tuple of array(s)): knob settings
        parameters (tuple of arrays, or tuple): parameter distribution sample(s)
        constants (tuple): infrequently changed values

    Returns: a noise-free model value
    """
    # Unpack the arguments.  See the "Specify ..." sections in the text.
    knob, = settings
    phase, delay = parameters
    temperature, = constants

    # This is where the model calculation goes.  It could be defined as a separate
    # function as suggested here, or the raw math expressions could go here.
    model_result = my_model_calculation(knob, phase, delay, temperature)

    return model_result

Hint: The ``numpy`` arrays offer convenience and computation speed.
Also, iIterations over numpy array elements are handled automatically using
broadcasting. See the Numpy User Guide for information on
`broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__.

The ``(settings, parameters, constants)`` arguments are required for
compatibility, and the example code above shows how they are unpacked. The
following sections describe what these arguments should contain.

Specify the allowed experimental settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generally, there can be more than one setting, so the convention is that
settings are always part of a tuple., i.e. ``(setting_1_values,
[setting_2_values, [ ... ]])`` with array-like lists of values for each
setting. Continuous settings must be discretized. The arrays in the settings
tuple may have different lengths. This example specifies a single setting,
a ``knob`` that goes to 11 with a resolution of 0.1.  In choosing settings,
``optbayesexpt`` will evaluate the experimental model function several times
for every combination of setting values included in the
``setting_N_values`` arrays.

::

    knob = np.linspace(0, 11, 111)
    setting_values = (knob, )

The first line here creates an array called ``knob`` that contains possible
knob settings, and the second line above packs ``knob`` as the first item in
a one-item tuple.

Specify the model parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters are specified by describing their initial or *prior* probability
distributions, each described by a generous sampling of draws from the
distribution.  It's easier to demonstrate than to explain.  For example,
suppose there’s a *phase* parameter that the measurements are to determine.
Let's say that the phase could be anywhere between :math:`-\pi/2` and
:math:`\pi/2` but that values outside this range are forbidden.  We
represent this *prior* knowledge with samples from a uniform distribution.

::

    n_samples = 50000
    phase = np.random.uniform(-np.pi/2, np.pi/2, n_samples)

Suppose also that there is an unknown *delay* parameter, and that there is
*prior* information that *delay* is 3, more or less\ :math:`^*`, but there
aren't and hard limits.  We might represent this *prior* using a normal
distribution with a width of 2.0.

::

    limit = np.random.normal(3.0, 2.0, n_samples)
    parameter_samples = (phase, limit)

The 2nd line above packs the parameter samples in a tuple.

Opinion:

    In order to generate independent results, I find it helpful to think of
    the *prior* as a generous expression of willingness to consider, rather
    than as a concise summary of preconceived notions. Narrow *prior*
    distributions risk biasing the results. A minimally biased result can
    always be compared and combined with independent results afterwards.

Specify constants
^^^^^^^^^^^^^^^^^

A definition for values (settings or parameters) that are constant for the
duration of an experiment, but that might change at some later time.

::

    temperature = 19   # degrees C
    constants = (temperature, )



Configure
^^^^^^^^^

The final part of preparation is to create an instance of the OptBayesExpt class

::

    my_obe = OptBayesExpt(model_function, setting_values, parameter_samples, constants)


Running
-------

BayesOptExpt participates at two stages in the measurement loop as shown
in the following **pseudocode**.

::

    while still_measuring:
        
        # (1) my_obe picks a single combination of settings - there's a choice of methods.
        # settings = my_obe.opt_setting()
        #   --  or --
        settings = my_obe.good_setting(pickiness=a_value_between_1_and_10)
        
        # The experiment makes a measurement using settings and returns a result
        # (Machine goes "bing!")
        # measurement results are reported as tuples
        measurement = (actual_settings, result, uncertainty)
        # (2) report the measurement
        my_obe.pdf_update(measurement)

    # end while loop
    
    # get results from the parameter distribution
    #
    mean_values = my_obe.mean()
    std_deviaion_values = my_obe.std()
    covariance_matrix = my_obe.covariance()


Footnote:

:math:`^*` Previous work has suggested that a *delay* parameter value of 2 is
transitional and that 5 is "right out."