__author__ = 'Bob McMichael'

import numpy as np


class ExptModel:
    """
    A class defining an experimental model and its evaluation.  Inherited by OptBayesExpt.

    Important:

        The :obj:`model_function()` method must be redefined to incorporate a model
        that's relevant to the application.

    Attributes:
        constants (:obj:`list`, :obj:`tuple`, :obj:`array`): Constant parameters in the model
            function.
        allsettings (:obj:`tuple` of :obj:`ndarray`): All possible combinations of setting values.
        allparams (:obj:`tuple` of :obj:`ndarray`): All possible combinations of parameter values.
    """
    def __init__(self):
        pass

    def model_config(self, settings, parameters, constants):
        """
        Creates arrays for model evaluation.

        Args:
            settings (:obj:`tuple` of :obj:`ndarray`): Each array in the settings tuple
                represents the allowed values of an experimental control "knob."
            parameters (:obj:`tuple` of :obj:`ndarray`): Each array in the parameters tuple
                represents the allowed values of a model parameter.
            constants (:obj:`tuple` of :obj:`float`): Values corresponding to constants in the
                model
        """
        self.constants = constants
        self.allsettings = np.meshgrid(*settings, indexing='ij')
        self.allparams = np.meshgrid(*parameters, indexing='ij')

    def model_function(self, settings, parameters, constants):
        """Defines the core parametric model

        "Out of the box," this function is a stub, and it must be redefined by the
        user.  The model_function is called in two ways: In the Bayesian inference phase,
        model_function is called for one set of measurement settings, and many sets of
        parameters

        Args:
            settings (:obj:`tuple` of :obj:`float`): One set of measurement settings.
            parameters (:obj:`tuple` of :obj:`ndarray`): Combinations of parameter values.
            constants (:obj:`tuple` of :obj:`float`): Model constants.

        Returns:
            A numpy array with dimensions of the parameters arrays.

        In the setting optimization phase, **model_function** is called for many sets of
        measurement settings and one set of parameters

        Args:
            settings (:obj:`tuple` of :obj:`ndarrayfloat`): Combinations of measurement settings.
            parameters (:obj:`tuple` of :obj:`float`):  One set of parameter values.
            constants (:obj:`tuple` of :obj:`float`): Model constants.

        Returns:
            A numpy array with dimensions of the settings arrays.


        A simple way to handle this polymorphic complication is to use pass tuples of numpy
        arrays as arguments to the model_function.  The *broadcasting* feature of numpy arrays
        allows simple code.

        Example:

            A function definition specifying the model for an experiment that generates a
            Lorentzian peak::

                def lorentz_model(settings, parameters, constants):
                # unpack settings
                #   frequency is the first element in a one-element tuple.
                frequency, = settings

                # unpack parameters
                #   center frequency, amplitude and line width from a 3-element tuple
                f0, amplitude, lw = parameters

                # calculate the model function
                return amplitude / (((frequency - f0) * 2 / lw)**2 + 1)

            Typically, the user will create an OptBayesExpt instance, which inherits EptModel.
            The user-defined model, (:code:`lorentz_model` in this case), can be incorporated by
            redefining the
            :obj:`model_function` method like this::

                obe_one = OptBayesExpt()
                obe_one.model_function = lorentz_model

        """

        pass

    def eval_over_all_parameters(self, onesettingset):
        """
        Evaluate the experimental model

        Evaluates the model for all possible model parameters and one set of
        measurement settings.  Used by OptBayesExpt in calculating likelihood of a measurement
        result see OptBayesExtp.update_pdf()

        Args:
            onesettingset (:obj:`tuple` of :obj:`float`): a single set of measurement settings

        Returns:
            (:obj:`ndarray`) array of model values with dimensions of :obj:`allparams.shape`.
        """
        return self.model_function(onesettingset, self.allparams, self.constants)

    def eval_over_all_settings(self, oneparamset):
        """
        Evaluate the experimental model

        Evaluates the model for all possible measurement settings one set of
        parameters.  Used by OptBayesExpt in predicting the utility of future measurement settings.

        Args:
            onesettingset (:obj:`tuple` of :obj:`float`): a single set of model parameters.

        Returns:
            (:obj:`ndarray`) array of model values with dimensions of :obj:`allsettings.shape`.
        """
        return self.model_function(self.allsettings, oneparamset, self.constants)


