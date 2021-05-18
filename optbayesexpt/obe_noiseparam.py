from optbayesexpt import OptBayesExpt
import numpy as np


class OptBayesExptNoiseParameter(OptBayesExpt):
    """Creates an obe class for a linear model with noise as parameter

    This class demonstrates three things:

        - Using the OptBayesExptBase class
        - Incorporating a noise characteristic as an unknown

    Often, the experimental uncertainty is an unknown, and it would be
    useful to have the "sigma" of the measurement noise as one of the
    parameters.

    Args:
        model_function (function): the experimental model.  Function
            must accept (settings, parameters, constants) arguments and
            return an array of 2 floats or of 2 arrays.
        setting_values (tuple of arrays): the allowed setting values.
        parameter_samples (tuple of arrays): random draws of each model
            parameter representing the prior probability distribution.
        constants (tuple): settings or parameters that are assumed constant
            for the duration of the measurement.

    Attributes:
        noise_parameter_index (int): identifies which parameter
            array contains the measurement sigma parameter.
    """

    def __init__(self, model_function, setting_values, parameter_samples,
                 constants):
        OptBayesExpt.__init__(self, model_function, setting_values,
                                  parameter_samples, constants)
        # identify the measurement noise parameter.
        self.noise_parameter_index = 3

    def pdf_update(self, measurement_record, y_model_data=None):
        """
        Adds the noise parameter array as the measurement_record[3]
        :param measurement_record:
        :param y_model_data:
        :return:
        """

        onesetting, y_meas = measurement_record[:2]
        # package sigma parameter array as uncertainty
        sigma = (self.parameters[self.noise_parameter_index],)
        new_record = (onesetting, y_meas,
                      (self.parameters[self.noise_parameter_index],))

        return super().pdf_update(new_record, y_model_data)

    def enforce_parameter_constraints(self):
        """
        All of the coil parameters and noise values must be > 0.  Assign
        zero probability to any violators.
        """
        changes = False
        param = self.parameters[self.noise_parameter_index]
        bad_ones = np.argwhere(param < 0).flatten()
        if len(bad_ones) > 0:
            changes = True
            for violator in bad_ones:
                # effective death penalty.  Next resample will remove
                self.particle_weights[violator] = 0

        if changes is True:
            # rescale the particle weights
            self.particle_weights = self.particle_weights \
                                    / np.sum(self.particle_weights)

    def y_var_noise_model(self):
        """Calculates the mean variance for noise as a parameter

        The self.opt_setting() and self.good_setting() methods calculate
        utility, which depends on the relative magnitudes of variance in
        measurement outcomes due to parameter distribution and variance due
        to measurement noise.  Here, we assume that the noise is independent
        of setting values and values of other parameters.

        Returns: (float) average variance.

        """
        # square the sigma samples
        sigma_squared = self.parameters[self.noise_parameter_index] ** 2
        # weighted average
        return np.average(sigma_squared, weights=self.particle_weights)
