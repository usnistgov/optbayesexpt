from optbayesexpt import OptBayesExpt
import numpy as np


class OptBayesExptNoiseParameter(OptBayesExpt):
    """Sequential Bayesian experiment design for with measurement
    uncertainty as random parameters to estimate.

    OptBayesExptNoiseParameter inherits attributes from OptBayesExpt,

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
                 constants, noise_parameter_index=None):
        OptBayesExpt.__init__(self, model_function, setting_values,
                                  parameter_samples, constants)

        # identify the measurement noise parameter.
        self.noise_parameter_index = noise_parameter_index

    def pdf_update(self, measurement_record, y_model_data=None):
        """
        Refines the parameters' probability distribution function given a
        measurement result.

        Incorporates a measurement result, allowing for measurement
        uncertainty as random variable to be estimated.  Packages
        measurement_record with the noise parameter array as the third
        element of the  measurement_record and calls
        OptBayesExpt.pdf_update() to calculate likelihood and generate a
        *posterior* parameter distribution.

        Args:
            measurement_record (:obj:`tuple`): A record of the measurement
                containing at least the settings and the measured value(s).
                The first element of ``measurement_record`` gets passed as a
                settings tuple to ``evaluate_over_all_parameters()`` The
                entire ``measurement_result`` tuple gets forwarded to
                ``likelihood()``.

            y_model_data (:obj:`ndarray`): The result of
                :code:`self.eval_over_all_parameters()` This argument allows
                model evaluation to run before measurement data is
                available, e.g. while measurements are being made. Default =
                ``None``.

        Returns:

        """

        onesetting, y_meas = measurement_record[:2]
        # package sigma parameter array as uncertainty
        sigma = (self.parameters[self.noise_parameter_index],)
        new_record = (onesetting, y_meas,
                      (self.parameters[self.noise_parameter_index],))

        # With a repackaged measurement record, use the pdf_update from the
        # parent class, ``OptBayesExpt``, which is invokes using ``super()``.
        return super().pdf_update(new_record, y_model_data)

    def enforce_parameter_constraints(self):
        """Constrains the noise parameter to be positive. Negative
        uncertainties lead to negative likelihoods, negative particle
        weights and other abominations.

        Returns:
        """

        changes = False
        param = self.parameters[self.noise_parameter_index]
        # np.nonzero identifies negative values
        bad_ones, = np.nonzero(param <= 0)
        if len(bad_ones) > 0:
            changes = True
            self.particle_weights[bad_ones] = 0
        #
        #  other parameter checks may be added here
        #
        # renormalize ``particle_weights`` if weights have been changed.
        if changes is True:
            # rescale the particle weights
            self.particle_weights = self.particle_weights \
                                    / np.sum(self.particle_weights)

    def y_var_noise_model(self):
        """Calculates the mean variance for noise as a parameter

        The self.opt_setting() and self.good_setting() methods calculate
        utility, which depends on the relative magnitudes of variance in
        measurement outcomes due to parameter distribution and variance due
        to measurement noise.

        Returns: (float) average variance.

        """
        # square the sigma samples
        sigma_squared = self.parameters[self.noise_parameter_index] ** 2
        # weighted average
        return np.average(sigma_squared, weights=self.particle_weights)
