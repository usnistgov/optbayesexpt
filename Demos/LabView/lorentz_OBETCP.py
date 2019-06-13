import sys

sys.path.append()
from OBETCP import BOE_Server

# define a model function
def lorentzian_model(self, settings, parameters, constants):
    # unpack our input tuples
    # experimental settings
    x = settings[0]
    # model parameter
    x0 = parameters[0]  # peak center
    A = parameters[1]  # peak amplitude
    B = parameters[2]  # background
    # constants
    d = constants[0]

    # and now our model 'fitting function' for the experiment
    return B + A / (((x - x0) / d) ** 2 + 1)
    # OK, this is just a one-liner, but the model could be much more complex.


# create a server
nanny = BOE_Server()
# connect the model
nanny.model_function = lorentzian_model

# wait for commands over TCP & respond until a 'done' command is received.
nanny.run()
