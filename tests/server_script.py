import optbayesexpt as obe
import numpy as np

# create a server
nanny = obe.OBE_Server(port=61982)

# define functions to customize nanny's behavior


def linear_model(settings, parameters, constants):
    # unpack settings - pulsetime is the first in a 1-element tuple
    x = settings
    # unpack b1, scale and offset from a 3-element tuple
    m, b = parameters
    # calculate the model function
    return m * x + b


def newrun_near_startpoint(datatuple):
    """
    configures settings, parameters
    :param centerfreq:
    :return:
    """
    centerfreq = datatuple[0]
    span = .2e9
    f0vals = np.linspace(centerfreq-span/2, centerfreq+span/2, 400)
    amplitude = np.linspace(.01, 0.1, 50)
    lw = np.linspace(.003, .050, 50)*1e9

    nanny.sets = (f0vals,)
    nanny.pars = (f0vals, amplitude, lw)
    nanny.cons = ()

    nanny.config()


def pretty_mean_value():
    mean_sig = nanny.get_mean(0)
    return {'mean': mean_sig[0], 'sigma': mean_sig[1]}


# connect the customized functions by redefining defaults
nanny.model_function = linear_model
nanny.newrun = newrun_near_startpoint
nanny.getmean = pretty_mean_value

nanny.run()
