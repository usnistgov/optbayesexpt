"""
A demonstration server script using the optbayesexpt.OBE_Server object.

The script is intended to run as a separate process, receiving commands and
performing sequential Bayesian experimental design tasks for an external
experiment.

This example server is configured with a Lorentzian model, and unknown
parameters: center frequency, amplitude, and linewidth.
"""

import optbayesexpt as obe
import numpy as np


def lorentz_model(settings, parameters, constants):
    """A Lorentzian model function

    Args:
        settings (tuple): the experimental controls, just frequency here
        parameters (tuple): the model parameters
        constants: (tuple): not used here, but an argument required by
        :obj:`OptBayesExpt`

    Returns: float
    """
    # unpack settings from a tuple, array, etc.
    frequency, = settings
    # unpack parameters tuple
    f0, amplitude, lw = parameters
    # unpack constants.  Nevermind. Not used here.
    # calculate the model function & return
    return 1 - amplitude / (((frequency - f0) * 2 / lw) ** 2 + 1)


# In this next section, describing the possible settings and creating the
# prior distribution of
# parameters.

# an initial array for settings.
fsets = np.linspace(2.4, 2.6, 1000)

# Generate a *prior* distribution of parameters
n_samples = 10000
f0vals = np.random.uniform(1.5, 3.5, n_samples)
amplitude = np.random.uniform(.01, .1, n_samples)
lw = np.random.uniform(.003, .05, n_samples)

# packaging - settings, parameters must be iterables of numpy arrays.
settings = (fsets,)
parameters = (f0vals, amplitude, lw)
# Constants must be an iterable of floats
constants = ()


# create a version of OBE_Server with custom newrun() method.
class CustomServer(obe.OBE_Server):
    """
    Inherits all the properties of optbayesexpt.OBE_Server, but overwrites
    the newrun method to process messages sent from magnetometer_demo.py
    """

    def __init__(self, initial_args=(), ip_address='127.0.0.1', port=61981,
                 **kwargs):
        obe.OBE_Server.__init__(self, initial_args=initial_args,
                                ip_address=ip_address, port=port, **kwargs)

    def newrun(self, message):
        """ set up for a new measurement

        Args:
            message(dict): A decoded message as a dict from the client

        In this example, the client sends limits and steps for the settings
        of a new run. This function decodes messages of the form:
        message = {'command': 'newrun', 'lolim': low_limit,
        'hilim': high_limit, 'steps': steps}
        """
        # create the new setting array using the message elements.
        setvalues = np.linspace(message['lolim'], message['hilim'],
                                message['steps'])
        new_settings = (setvalues,)  # settings is a tuple

        # Arrange information to build an OptBayesExpt class
        model, sets, params, consts = self.initial_args
        class_args = (model, new_settings, params, consts)
        kwargs = self.initial_kwargs
        # create and attach a new obe engine
        self.make_obe(obe.OptBayesExpt, class_args, **kwargs)


args = (lorentz_model, settings, parameters, constants)

nanny = CustomServer(initial_args=args, scale=False)

# nanny has no obe_engine at this point, but a call to newrun() will create
# one.

# Start serving.
nanny.run()
# wait for commands over TCP & respond until a 'done' command is received.
