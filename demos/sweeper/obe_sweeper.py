import numpy as np
from optbayesexpt import OptBayesExptNoiseParameter
try:
    rng = np.random.default_rng()
except AttributeError:
    rng = np.random
import matplotlib.pyplot as plt

class OptBayesExptSweeper(OptBayesExptNoiseParameter):
    """ An OptBayesExpt class for instruments that sweep a parameter

    This class provides (start, stop) pairs as measurement settings,
    and interprets arrays of data representing sweeps of a setting.

    Args:

        model_function (:obj:`function`):
            The ``model_function`` will be evaluated for either a single
            combination of settings and many combinations of parameters or
            vise-versa.  The different argument types can be accommodated by
            the broadcasting properties of numpy arrays.

        setting_values (:obj:`tuple` of :obj:`ndarray`):
            Each array in the :code:`setting_values` tuple contains the
            allowed discrete values of a measurement setting.  The first
            setting array is the setting that will be swept; start values
            and stop values will be selected from this first array.  See
            ``start_stop_subsample``.

        parameter_samples (:obj:`tuple` of :obj:`ndarray`):
            Each array in the :code:`parameter_samples` tuple contains the
            possible values of a model parameter.  In a simple example
            model, :code:`y = m * x + b`, the parameters are :code:`m` and
            :code:`b`.  As with the :code:`setting_values`,
            :code:`parameter_samples` arrays should be kept few and small.
            Parameters that can be assumed constant belong in the
            :code:`constants` array. Discretization should only be fine
            enough to support the needed measurement precision. The
            parameter ranges must also be limited: too broad, and the
            computation will be slow; too narrow, and the measurement may
            have to be adjusted and repeated.

        constants (:obj:`tuple` of :obj:`float`):
            Model constants.  Examples include experimental settings that
            are rarely changed, and model parameters that are well-known
            from previous measurement results.

    Attributes:

        cost_of_new_sweep (:obj:`float`):
            The cost of a sweep is modeled as::

                cost = (stop_index - start_index) + cost_of_new_sweep.

            ``cost_of_new_sweep`` is relative to the cost of one measurement
            in a sweep.

        start_stop_subsample (:obj:`int`): Allows the start and stop values
            to have a lower resolution than the swept parameter. In the
            experimental design phase, the utility considers all viable
            combinations of start and stop, which is O(N**2).  This
            computational load can is reduced by a factor of order
            ``start_stop_subsample**2``.
    """

    def __init__(self, model_function, setting_values, parameter_samples,
                 constants, **kwargs):
        OptBayesExptNoiseParameter.__init__(self, model_function,
                setting_values, parameter_samples, constants, **kwargs)
        self.sweep_settings = setting_values[0]
        self.start_stop_subsample = 3
        self.start_stop_indices = self._generate_start_stop_indices()
        self.start_stop_choice_indices = np.arange(len(self.start_stop_indices), dtype=int)
        self.start_stop_values = self.sweep_settings[self.start_stop_indices]

        self.noise_parameter_index = 3
        self.cost_of_new_sweep = 5.

    def pdf_update(self, measurement_record, y_model_data=None):
        """Performs Bayesian inference on swept measurement results.

        Args:
            measurement_record (:obj:`tuple`): A record of the measurement
                containing an array of the settings used in the sweep and
                the corresponding measurement values.
            y_model_data (): arrays of model values for all parameters,
                one array for each setting value in a sweep.  Possibly
                calculate while the experiment is working.
        """
        # settings are always packaged in tuples
        (setting_values,), result_values = measurement_record
        if y_model_data is None:
            # Iterate through the settings and results, calculating y_model
            for setting, result in zip(setting_values, result_values):
                # # calculate the model values for all parameters
                # # model expects setting value in a tuple
                # y_model_data = self.eval_over_all_parameters((setting,))
                # package the noise parameter as a sigma
                sigma = (self.parameters[self.noise_parameter_index],)
                measurement_package = ((setting,), result, sigma)
                super().pdf_update(measurement_package)

    def cost_estimate(self):
        # Assume that pointwise costs are uniform for all swept settings.
        return 1.0

    def sweep_cost_estimate(self):
        """
        Estimate the cost of measurements, depending on settings

        The denominator of the *utility* function allows measurement
        resources (e.g. setup time + data collection time) to be entered
        into the utility calculation.

        Returns:
            :obj:`float`, otherwise an :obj:`ndarray` describing how
                measurement variance depends on settings.
        """

        return self.start_stop_indices[:, 1] - self.start_stop_indices[:, 0] \
            + self.cost_of_new_sweep

    def sweep_utility(self):
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
        # var_p = self.yvar_from_parameter_draws()
        # var_n = self.y_var_noise_model()
        cost = self.sweep_cost_estimate()
        #
        point_utility = self.utility()

        # print(point_utility)
        # indefinite integral along sweep

        proto_utility = np.cumsum(point_utility)
        # evaluate indefinite integral at ends
        ends = proto_utility[self.start_stop_indices]
        # utility integral between start and stop
        utility_s = (ends[:, 1] - ends[:, 0]) / cost
        return utility_s

    def opt_setting(self):
        """Find the setting with maximum utility

        Calls :code:`utility()` for an estimate of the benfit/cost ratio for
        all allowed settings (i.e. start, stop combinations), and returns the
        settings corresponding to the maximum utility.

        Returns:
            A settings tuple.
        """
        # Find the settings with the maximum utility
        # argmax returns an array of indices into the flattened array
        bestindex = np.argmax(self.sweep_utility())
        return self.start_stop_indices[bestindex]

    def good_setting(self):
        """
        Calculate a setting with a good probability of refining the pdf

        ``good_setting()`` selects settings using a weighted random
        selection using the utility function to calculate a weight.  The
        weight function is ``utility( )`` raised to the ``pickiness`` power.
        In comparison to the ``opt_setting()`` method, where the
        measurements select only the very best setting, ``good_setting()``
        yields a more diverse series of settings.

        Args:
            pickiness (float): A setting selection tuning parameter.
                Pickiness=0 produces random settingss.  With pickiness
                values greater than about 10 the behavior is similar to
                :code:`opt_setting()`.

        Returns:
            A settings tuple.
        """

        util = self.sweep_utility() ** self.pickiness
        weight = util / np.sum(util)
        # the exponent 'pickiness' is a tuning parameter

        # choose a start, stop pair based on weight
        return rng.choice(self.start_stop_indices, p=weight)

    def random_setting(self):
        """
        Pick a random setting for the next measurement

        ``random_setting()`` randomly selects a setting from all possible
        setting combinations.

        Returns:
            A settings tuple.
        """
        index_pair = rng.choice(self.start_stop_indices)
        return index_pair

    def _generate_start_stop_indices(self):
        """Creates valid [start, stop] index combinations

        Returns:  list of [start, stop] indices
        """
        raw_length = len(self.sweep_settings)
        index_subsamples = list(np.arange(0, raw_length,
                                          self.start_stop_subsample))
        # tack on the last setting index
        last_index = raw_length - 1
        if last_index != index_subsamples[-1]:
            index_subsamples.append(last_index)

        # generate start, stop pairs for stop > start
        index_pairs = []
        for i, istart in enumerate(index_subsamples[:-1]):
            [index_pairs.append([istart, istop]) for istop in
                index_subsamples[(i + 1):]]
        return np.array(index_pairs)
