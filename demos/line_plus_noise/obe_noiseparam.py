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

    def likelihood(self, y_model, measurement_result):
        """Calculate the likelihood with measurement uncertainty as a
        parameter.

        Assumes that experimental noise is Gaussian noise and that sigma is
        "packed" as the third parameter.
        Args:
            y_model: modeled measurement mean values for all parameter
                combos. The result of a self.evaluate_ove_all_parameters()
                call.
            measurement_result: A tuple of (settings, measured values). Note
                that the user's script must call self.pdf_update() with an
                argument that fits this format.

        Returns: likelihood values for all parameters

        """
        # unpacking
        sigma = self.parameters[self.noise_parameter_index]
        setting, yvalue = measurement_result
        # calculate likelihood assuming Gaussian noise.
        likyhd = np.exp(-(y_model - yvalue) ** 2 / (2 * sigma ** 2)) / np.abs(
            sigma)
        return likyhd

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
