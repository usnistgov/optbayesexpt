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


def differential_entropy(values, window_length=None, base=None,
                         axis=0, method='auto'):
    """Given a sample of a distribution, estimate the differential entropy.

    This code is copied from scipy.stats with reformatted docstrings.  When the module is
    loaded, __init__.py attempts to import ``differential_entropy()`` from scipy.stats, and loads
    this version from obe_utils.py if an ``ImportError`` is raised.

    Several estimation methods are available using the `method` parameter. By
    default, a method is selected based the size of the sample.

    Args:
        values (:obj:`sequence`): Samples from a continuous distribution.

        window_length (:obj:`int`, optional): Window length for computing Vasicek estimate.
            Must be an integer between 1 and half of  the sample size. If ``None``
            (the default), it uses the heuristic value

            .. math::
                \left \lfloor \\sqrt{n} + 0.5 \\right \\rfloor

            where :math:`n` is the sample size. This heuristic was originally
            proposed in [2]_ and has become common in the literature.

        base (:obj:`float`, optional)
            The logarithmic base to use, defaults to ``e`` (natural logarithm).

        axis (:obj:`int`, optional)
            The axis along which the differential entropy is calculated.
            Default is 0.

        method : {'vasicek', 'van es', 'ebrahimi', 'correa', 'auto'}, optional
            The method used to estimate the differential entropy from the sample.
            Default is ``'auto'``.  See Notes for more information.

    Returns:
        entropy (:obj:`float`):
            The calculated differential entropy.

    Notes:
        This function will converge to the true differential entropy in the limit

        .. math::
            n \\to \\infty, \\quad m \\to \\infty, \\quad \\frac{m}{n} \\to 0

        The optimal choice of ``window_length`` for a given sample size depends on
        the (unknown) distribution. Typically, the smoother the density of the
        distribution, the larger the optimal value of ``window_length`` [1]_.
        The following options are available for the `method` parameter.

        * ``'vasicek'`` uses the estimator presented in [1]_. This is one of the
          first and most influential estimators of differential entropy.
        * ``'van es'`` uses the bias-corrected estimator presented in [3]_, which
          is not only consistent but, under some conditions, asymptotically normal.
        * ``'ebrahimi'`` uses an estimator presented in [4]_, which was shown
          in simulation to have smaller bias and mean squared error than
          the Vasicek estimator.
        * ``'correa'`` uses the estimator presented in [5]_ based on local linear
          regression. In a simulation study, it had consistently smaller mean
          square error than the Vasiceck estimator, but it is more expensive to
          compute.
        * ``'auto'`` selects the method automatically (default). Currently,
          this selects ``'van es'`` for very small samples (<10), ``'ebrahimi'``
          for moderate sample sizes (11-1000), and ``'vasicek'`` for larger
          samples, but this behavior is subject to change in future versions.

        All estimators are implemented as described in [6]_.

    References:

        .. [1] Vasicek, O. (1976). A test for normality based on sample entropy.
               Journal of the Royal Statistical Society:
               Series B (Methodological), 38(1), 54-59.
        .. [2] Crzcgorzewski, P., & Wirczorkowski, R. (1999). Entropy-based
               goodness-of-fit test for exponentiality. Communications in
               Statistics-Theory and Methods, 28(5), 1183-1202.
        .. [3] Van Es, B. (1992). Estimating functionals related to a density by a
               class of statistics based on spacings. Scandinavian Journal of
               Statistics, 61-72.
        .. [4] Ebrahimi, N., Pflughoeft, K., & Soofi, E. S. (1994). Two measures
               of sample entropy. Statistics & Probability Letters, 20(3), 225-234.
        .. [5] Correa, J. C. (1995). A new estimator of entropy. Communications
               in Statistics-Theory and Methods, 24(10), 2439-2449.
        .. [6] Noughabi, H. A. (2015). Entropy Estimation Using Numerical
               Methods. Annals of Data Science, 2(2), 231-241.
               https://link.springer.com/article/10.1007/s40745-015-0045-9

    """
    values = np.asarray(values)
    values = np.moveaxis(values, axis, -1)
    n = values.shape[-1]  # number of observations

    if window_length is None:
        window_length = int(np.floor(np.sqrt(n) + 0.5))

    if not 2 <= 2 * window_length < n:
        raise ValueError(
            f"Window length ({window_length}) must be positive and less "
            f"than half the sample size ({n}).",
        )

    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    sorted_data = np.sort(values, axis=-1)

    methods = {"vasicek": _vasicek_entropy,
               "van es": _van_es_entropy,
               "correa": _correa_entropy,
               "ebrahimi": _ebrahimi_entropy,
               "auto": _vasicek_entropy}
    method = method.lower()
    if method not in methods:
        message = f"`method` must be one of {set(methods)}"
        raise ValueError(message)

    if method == "auto":
        if n <= 10:
            method = 'van es'
        elif n <= 1000:
            method = 'ebrahimi'
        else:
            method = 'vasicek'

    res = methods[method](sorted_data, window_length)

    if base is not None:
        res /= np.log(base)

    return res


def _pad_along_last_axis(X, m):
    # Pad the data for computing the rolling window difference.
    # scales a  bit better than method in _vasicek_like_entropy
    shape = np.array(X.shape)
    shape[-1] = m
    Xl = np.broadcast_to(X[..., [0]], shape)  # [0] vs 0 to maintain shape
    Xr = np.broadcast_to(X[..., [-1]], shape)
    return np.concatenate((Xl, X, Xr), axis=-1)


def _vasicek_entropy(X, m):
    # Compute the Vasicek estimator as described in [7] Eq. 1.3.
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m)
    differences = X[..., 2 * m:] - X[..., : -2 * m:]
    logs = np.log(n/(2*m) * differences)
    return np.mean(logs, axis=-1)


def _van_es_entropy(X, m):
    # Compute the van Es estimator as described in [7].
    # No equation number, but referred to as HVE_mn.
    # Typo: there should be a log within the summation.
    n = X.shape[-1]
    difference = X[..., m:] - X[..., :-m]
    term1 = 1/(n-m) * np.sum(np.log((n+1)/m * difference), axis=-1)
    k = np.arange(m, n+1)
    return term1 + np.sum(1/k) + np.log(m) - np.log(n+1)


def _ebrahimi_entropy(X, m):
    # Compute the Ebrahimi estimator as described in [7].
    # No equation number, but referred to as HE_mn
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m)

    differences = X[..., 2 * m:] - X[..., : -2 * m:]

    i = np.arange(1, n+1).astype(float)
    ci = np.ones_like(i)*2
    ci[i <= m] = 1 + (i[i <= m] - 1)/m
    ci[i >= n - m + 1] = 1 + (n - i[i >= n-m+1])/m

    logs = np.log(n * differences / (ci * m))
    return np.mean(logs, axis=-1)


def _correa_entropy(X, m):
    # Compute the Correa estimator as described in [7].
    # No equation number, but referred to as HC_mn
    n = X.shape[-1]
    X = _pad_along_last_axis(X, m)

    i = np.arange(1, n+1)
    dj = np.arange(-m, m+1)[:, None]
    j = i + dj
    j0 = j + m - 1  # 0-indexed version of j

    Xibar = np.mean(X[..., j0], axis=-2, keepdims=True)
    difference = X[..., j0] - Xibar
    num = np.sum(difference*dj, axis=-2)  # dj is d-i
    den = n*np.sum(difference**2, axis=-2)
    return -np.mean(np.log(num/den), axis=-1)
