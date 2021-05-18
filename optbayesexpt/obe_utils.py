import numpy as np
try:
    rng = np.random.default_rng()
except AttributeError:
    rng = np.random
    
class MeasurementSimulator():
    """
    Provides simulated measurement data

    Evaluates the model function and adds noise.

    Args:
        model_function (func): Generally the same as the function used
            by OptBayesExpt
        true_params (tuple): Parameter values, typically the "true values"
            of the simulated experiment.
        cons (tuple): The constants.
        noise_level (float): standard deviation of the added noise.

    """
    def __init__(self, model_function, true_params, cons, noise_level):

        self.model_function = model_function
        self.params = true_params
        self.cons = cons
        self.noise_level = noise_level

    def simdata(self, setting, params=None, noise_level=None):
        """ Simulate a measurement

        Args:
            setting (tuple of floats): The setting values
            params (tuple of floats): if not ``None``, temporarily used
                instead of the initial values. (opt)
            noise_level (float): if not ``None``, temporarily used instead of
                the initial value. (opt)

        Returns:
            Simulated measurement value(s)
        """
        if params is None:
            params = self.params

        if noise_level is None:
            noise_level = self.noise_level

        y = np.array(self.model_function(setting, params, self.cons))
        tmpnoise = rng.standard_normal(y.shape) * noise_level
        yn = y + tmpnoise

        return yn


def trace_sort(settings, measurements):
    """Combine measurements at identical settings values

    Analyzes input arrays of setttings and corresponding measurement
    values, data where settings values may repeat, i. e. more than one
    measurement was done at some of the settings.  The function
    bins the measurements by setting value and calculates some statistics
    for measurments in each bin.

    Args:
        settings: (ndarray) Setting values
        measurements: (ndarray) measurement values

    Returns:
        A tuple, (sorted_settings, m_average, m_std, n_of_m)
            - sorted_settings (list): setting values (sorted, none repeated)
            - m_average (list): average measurement value at each setting
            - m_sigma (list): standard deviation of measurement values at
              each setting
            - n_of_m (list): number of measurements at each setting.

    """

    # Sort the arrays by the setting values
    sortindices = np.argsort(settings)
    sarr = np.array(settings)[sortindices]
    marr = np.array(measurements)[sortindices]

    oldx = sarr[0]
    sorted_settings = []
    m_average = []
    m_std = []
    n_of_m = []
    m_list = []

    for x, y in zip(sarr, marr):
        # accumulate batches having the same x
        # check if the new x value is different
        if x != oldx:
            # new x value, so batch is complete
            # process the accumulated data for the old x value
            sorted_settings.append(oldx)
            m_average.append(np.mean(np.array(m_list)))
            m_std.append(np.std(m_list)/np.sqrt(len(m_list)))
            n_of_m.append(len(m_list))
            # reset accumulation & start a new batch
            oldx = x
            m_list = [y, ]
        else:
            # same setting value, so just accumulate the y value
            m_list.append(y)
    # process the last accumulated batch
    sorted_settings.append(oldx)
    m_average.append(np.mean(np.array(m_list)))
    n_of_m.append(len(m_list))
    m_std.append(np.std(np.array(m_list))/np.sqrt(len(m_list)))

    return sorted_settings, m_average, m_std, n_of_m
