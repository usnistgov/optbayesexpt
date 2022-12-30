from optbayesexpt.obe_base import OptBayesExpt

import numpy as np


class OptBayesExptNoiseParameter(OptBayesExpt):
    """OptBayesExpt adding measurement uncertainty as an unknown parameter.

    OptBayesExptNoiseParameter is designed for cases where the experimental
    uncertainty / standard deviation of the measurement noise is an unknown.
    The standard deviation of the measurement
    noise is included as one of the parameters.
    The methods rely on several assumptions:

        - The noise in measurement data is well-described by additive,
          normally-distributed (Gaussian) noise.
        - The noise is independent of settings and other parameters.

    Arguments:
        measurement_model (function): A function accepting (settings,
            parameters, constants) arguments and returning model values of
            experiment outputs. See the ``model_function`` arg of
            ``OptBayesExpt``.
        setting_values (tuple of arrays): Arrays of allowed values for each
            setting.  See the ``setting_values`` arg of ``OptBayesExpt``.
        parameter_samples (tuple of arrays): random draws of each model
            parameter representing the prior probability distribution.
            See the ``parameter_samples`` arg of ``OptBayesExpt``. One of the
            parameter_sample arrays must be the standard deviation of the
            measurement noise, idendified by the ``noise_parameter_index`` arg.
        constants (tuple): settings or parameters that are assumed constant.
            See the ``constants`` arg of ``OptBayesExpt``.
        noise_parameter_index (int or tuple): identifies which of the arrays
            in the ``paramter_samples`` input are uncertainty parameters.
            For multi-channel measurements, the tuple identifies uncertainty
            parameters corresponding to measurement channels. In cases where
            channels have the same noise characteristics, indices may be
            repeated in the tuple.

        \*\*kwargs: Keyword arguments passed to the parent classes.

    **Attributes:**
    """

    def __init__(self, measurement_model, setting_values, parameter_samples,
                 constants, noise_parameter_index=None, **kwargs):
        OptBayesExpt.__init__(self, measurement_model, setting_values,
                              parameter_samples, constants, **kwargs)

        # identify the measurement noise parameter.
        #: int: Stores the noise_parameter_index argument
        self.noise_parameter_index = np.atleast_1d(noise_parameter_index)
        if len(self.noise_parameter_index) != self.n_channels:
            raise RuntimeError(f'noise_parameter_index is not compatible with'
                               f' {self.n_channels} measurement channels')

    def enforce_parameter_constraints(self):
        """Detects and nullifies disallowed parameter values

        Constrains the noise parameter to be positive. Negative
        uncertainties lead to negative likelihoods, negative particle
        weights and other abominations. Overwrites the ``OptBayesExpt`` stub
        method.
        """
        changes = False
        # np.nonzero identifies negative values
        for param in self.parameters[self.noise_parameter_index]:
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

    def likelihood(self, y_model, measurement_record):
        """
        Calculates the likelihood of a measurement result.

        For each parameter combination, estimate the probability of
        obtaining the results provided in :code:`measurement_result`.

        Under these assumptions, and model values :math:`y_{model}` as a
        function of parameters, the likelihood is a Gaussian function
        proportional to :math:`\sigma^{-1} \exp [-(y_{model} - y_{meas})^2
        / (2 \sigma^2)]`.

        Args:
            y_model (:obj:`ndarray`): ``model_function()`` results evaluated
                for all parameters.
            measurement_record (:obj:`tuple`): The measurement conditions
                and results, supplied by the user to ``update_pdf()``. The
                first two elements of ``measurement_record`` are:

                    - settings (tuple)
                    - measurement value (float or tuple)

                further entries in the measurement_record are ignored.

        Returns:
            an array of probabilities corresponding to the parameters in
            :code:`self.allparameters`.
        """
        # unpack the measurement_record
        onesetting, y_meas = measurement_record[:2]
        lky = 1.0
        sigma = self.parameters[self.noise_parameter_index]
        for y_m, y, s in zip(y_model,
                             np.atleast_1d(y_meas), sigma):
            lky *= self._gauss_noise_likelihood(y_m, y, s)

        if self.choke is not None:
            return np.power(lky, self.choke)
        else:
            return lky

    def yvar_noise_model(self):
        """Calculates the mean variance from the noise parameter.

        Overwrites OptBayesExpt method, replacing :code:`default_noise_std **
        2` with the mean variance calculated from the noise parameter. Used
        in calculating utility.

        Returns: (float) average variance.
        """
        # square the sigma samples
        sigma_squared = self.parameters[self.noise_parameter_index] ** 2
        # weighted average
        yvars = np.average(sigma_squared, weights=self.particle_weights,
                          axis=1)
        return yvars.reshape((self.n_channels, 1))
