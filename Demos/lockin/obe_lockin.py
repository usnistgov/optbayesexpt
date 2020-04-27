from optbayesexpt import OptBayesExpt
import numpy as np


class OptBayesExptLockin(OptBayesExpt):
    """ An OptBayesExpt object designed for a lockin amplifier

    - Multiple (two) simultaneous measurement channels.  See
      ``self.VALUES_PER_MEASUREMENT``, ``yvar_from_parameter_draws()``, and
      ``Likelihood()``.
    - Setting-dependent cost function. Extra time is required for settling
      when the setting is changed.  See ``cost_estimate()``,
      ``self.cost_of_changing_setting``
    - Unknown noise level.  The standard deviation of the (Gaussian)
      measurement noise is one of the parameters.  See
      ``likelihood()`` and ``y_var_noise_model()``.  The noise is assumed to
      be the same for both channels.

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
        cost_of_changing_setting (float):  The cost of changing the setting
            and making a measurement relative to repeating a measurement at
            the current setting.  Changing the excitation frequency often
            produces a transient, and the rule of thumb is to wait 5 time
            constants for the output to settle.  Default 5.

        VALUES_PER_MEASUREMENT (int): The number of measurement channels,
            used by ``yvar_from_parameter_draws()`` the ``likelihood()`` to
    """

    def __init__(self, model_function, setting_values, parameter_samples,
                 constants):

        OptBayesExpt.__init__(self, model_function, setting_values,
                                  parameter_samples, constants)

        self.VALUES_PER_MEASUREMENT = 2
        self.cost_of_changing_setting = 5.
        self.noise_parameter_index = 3


    def likelihood(self, y_model, measurement_record):
        """
        Calculates the likelihood of a measurement result.

        Args:
            y_model (:obj:`ndarray`): ``model_function()`` results evaluated
                for all parameters.
            measurement_record (:obj:`tuple`): The measurement conditions
                and results, supplied by the user to ``update_pdf()``. The
                elements of ``measurement_record`` are:

                    - settings (tuple)
                    - measurement value (float)
                    - std uncertainty (float)

        Returns:
            an array of probabilities corresponding to the parameters in
            :code:`self.allparameters`.
        """
        # unpack the measurement_record
        onesetting, y_meas = measurement_record

        noise = self.parameters[self.noise_parameter_index]
        sigma = np.array((noise, noise))
        # Assuming Gaussian noise with sigma as a parameter.

        # Compute the likelihood for each of the measurement channels and
        # multiply them together..
        x_llh = np.exp(-((y_model[0] - y_meas[0]) / sigma[0]) ** 2 / 2) \
            / (sigma[0])
        y_llh = np.exp(-((y_model[1] - y_meas[1]) / sigma[1]) ** 2 / 2) \
            / (sigma[1])

        likelihood_result = x_llh * y_llh

        return likelihood_result

    def yvar_from_parameter_draws(self):
        """Models the measurement variance solely due to parameter
        distributions.

        Evaluates the effect ov the distribution
        of parameter values on the distribution of model outputs for every
        setting combination. This calculation is done as part of the
        *utility* calculation as an approximation to the information
        entropy. For each of ``self.N_DRAWS`` samples from the parameter
        distribution, this method models a noise-free experimental output
        for all setting combinations and returns the variance of the model
        values for each setting combination.

        Returns:
            :obj:`ndarray` with shape of a member of all_settings
        """
        paramsets = self.randdraw(self.N_DRAWS).T
        # make space for model results
        ycalc = np.zeros((self.N_DRAWS, self.VALUES_PER_MEASUREMENT)
                         + self.allsettings[0].shape)
        # the default for the default number of draws is set in __init__()

        # fill the model results for each drawn parameter set
        for i, oneparamset in enumerate(paramsets):
            ycalc[i] = self.eval_over_all_settings(oneparamset)

        # Evaluate how much the model varies at each setting
        # calculate the variance of results for each setting
        return np.var(ycalc, axis=0)

    def y_var_noise_model(self):
        """
        Models the measurement variance solely due to measurement noise.

        A model of measurement variance (noise) as a function of settings,
        averaged over parameters if parameter-dependent.  Used in the
        *utility* calculation.

        In general, the measurement noise could depend on both settings and
        parameters, and the model would require evaluation of the noise
        model over all parameters, averaged over draws from the parameter
        distribution.  Measurement noise that depends on the measurement
        value, like root(N), Poisson-like counting noise is an example of
        such a situation. Fortunately, this noise estimate only affects the
        utility function, which only affects setting choices, where the
        "runs good" philosophy of the project allows a little approximation.

        Returns:
            If measurement noise is independent of settings, a :obj:`float`,
            otherwise an :obj:`ndarray` with the shape of an
            element of `allsettings`
        """
        # square the sigma samples
        sigma_squared = self.parameters[self.noise_parameter_index] ** 2
        # weighted average
        return np.average(sigma_squared, weights=self.particle_weights)

    def cost_estimate(self):
        """
        Estimate the cost of measurements, depending on settings

        The denominator of the *utility* function allows measurement
        resources (e.g. setup time + data collection time) to be entered
        into the utility calculation.

        Returns:
            :obj:`float`, otherwise an :obj:`ndarray` describing how
                measurement variance depends on settings.
        """
        setting = self.last_setting_index
        cost = np.ones_like(self.allsettings[0]) * \
            self.cost_of_changing_setting
        cost[setting] = 1.0

        return cost

    def utility(self):
        """
        Estimate the utility as a function of settings.

        Used in selecting measurement settings. The *utility* is the
        predicted benefit/cost ratio of a new measurement where the benefit
        is given in terms of a change in the information entropy of the
        parameter distribution.  Here we use an approximation that assumes
        Gaussian distributions.

        Returns:
            utility as an :obj:`ndarray` with dimensions of a member of
            :code:`allsettings`
        """
        var_p = self.yvar_from_parameter_draws()
        var_n = self.y_var_noise_model()
        cost = self.cost_estimate()

        util = np.log(1 + (var_p[0] + var_p[1]) / var_n) / cost

        return util
