Version Notes
=============

Version 1.1.0
-------------

May 21, 2021

* Support for multi-channel measurements has been added to the OptBayesExpt
  class. As a result, ``demos/lockin/obe_lockin.py`` is no longer needed,
  and it has been removed.

* Support for noise parameter estimation is provided by a new component of
  the optbayesespt package, ``OptBayesExptNoiseParam``, which takes a
  ``noise_parameter_index=(int)`` argument to identify a parameter as
  measurement noise. These demos  now use ``OptBayesExptNoiseParam``.

    - ``demos/line_plus_noise/line_plus_noise.py``,
    - ``demos/lockin/lockin_of_coil.py``, and
    - ``demos/sweeper/sweeper.py``

* Added support for ``**kwargs`` arguments to OptBayesExpt. Attribute values
  for OptBayesExpt, parent class ParticlePDF and OptBayesExpt child classes can
  now be set at instantiation.  Keyword arguments ``a_param``,
  ``resample_threshold``, ``auto_resample`` and ``scale`` are passed to
  ParticlePdf to tune resampling behavior.  ``OptBayesExpt`` uses ``choke``,
  and ``OptBayesExptNoiseParam`` uses ``noise_parameter_index``.

May 11, 2021

Plans for version 1.1.0

* Accelerate computation-heavy functions using numba on systems where the
  numba package is installed.

Version 1.0.1
-------------

June 2, 2020

* Provides backwards compatibility to numpy.random usage for numpy versions pre-1.17.0.

* Fixes some plotting problems

* Adds 'ready' command to OBE_Server.run() for communication checks.

Version 1.0.0
-------------

April 27, 2020

Version 1.0.0 represents an overhaul of the optbayesexpt python package.  It
is not compatible with earlier versions, but only minor changes are needed to
adapt script to use the new version.
The most significant changes are briefly described here. Please consult the
documentation at https://pages.nist.gov/optbayesexpt for more detail.

Probability Distribution Function:
    Starting with V.1.0.0, the probability distribution function over
    parameter values is implemented using a sequential
    Monte Carlo scheme in ``ParticlePDF()``, replacing the
    N-dimensional array representation used in ``ProbDistFunc()``. This
    change boosts speed and allows more parameters in the model function.

Experiment Model:
    Starting with V.1.0.0, the ``ExptModel`` class is no longer used. Methods
    of the ``ExptModel`` class are incorporated into ``OptBayesExpt``.

OptBayesExpt class:
    The OptBayesExpt class has been rewritten with reuse in mind.
    As much as possible, the calculations have been split out into separate
    methods.  The goal was to make is easier to determine how to create
    customized child classes for different applications.

    Creation of a functioning ``OptBayesExpt`` object has been simplified
    by including the model function, settings, parameters and constants as
    arguments to ``__init__()``.  In earlier versions, the object was created
    and then configured in separate steps.

Server:
    The ``OBE_Server class`` has been redesigned to be a caretaker and TCP
    communication interface for OptBayesExpt objects.  With this new design
    a OBE_Server object can initialize a series of OptBayesExpt objects
    with different configurations, e.g. for a series of measurement runs.



