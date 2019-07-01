from .OptBayesExpt import OptBayesExpt
from .obe_socket import Socket


class OBE_Server(Socket, OptBayesExpt):
    def __init__(self, ip_address='127.0.0.1', port=20899):
        Socket.__init__(self, 'server', ip_address=ip_address, port=port)
        OptBayesExpt.__init__(self)

        self.noise = 1.0
        self.Ndraws = 100

    def run(self):
        print()
        print('SERVER READY')
        while True:
            # use the Socket.receive() method to get the incoming message from Labview
            message = self.receive()
            # the messages we get from Labview are Labview clusters encoded as json objects
            # Decoded, they are python dicts.

            # manipulate coinfiguration arrays for settings, params, consts.
            # clear commands
            if 'clrset' in message['command']:
                self.clrsets()
                self.send('OK')
            elif 'clrpar' in message['command']:
                self.clrpars()
                self.send('OK')
            elif 'clrcon' in message['command']:
                self.clrcons()
                self.send('OK')

            # get commands request arrays
            elif 'getset' in message['command']:
                self.send(self.sets)
            elif 'getpar' in message['command']:
                self.send(self.pars)
            elif 'getcon' in message['command']:
                self.send(dumps(self.cons))

            # add arrays
            elif 'addset' in message['command']:
                self.addsets(message['array'])
                self.send('OK')
            elif 'addpar' in message['command']:
                self.addpars(message['array'])
                self.send('OK')
            elif 'addcon' in message['command']:
                self.addcon(message['value'])
                self.send('OK')

            # Finish configuration
            elif 'config' in message['command']:
                # self.config(self.sets, self.pars, self.cons)
                self.config()
                self.send('OK')

            # run-time commands
            elif 'optset' in message['command']:
                self.send(self.opt_setting())
            elif 'goodset' in message['command']:
                if 'pickiness' in list(message):
                    self.send(self.good_setting(pickiness=message['pickiness']))
                else:
                    self.send(self.good_setting())
            elif 'newdat' in message['command']:
                self.pdf_update((message['x'],), message['y'], message['s'])
                self.send('OK')
            elif 'getpdf' in message['command']:
                self.send(list(self.PDF))
            elif 'maxpdf' in message['command']:
                self.send(self.max_params())

            elif 'done' in message['command']:
                self.send('OK')
                break
            else:
                pass


if __name__ == '__main__':
    # define a model function
    def lorentzian_model(settings, parameters, constants):
        # unpack our input tuples
        # experimental settings
        x = settings[0]
        # model parameter
        x0 = parameters[0]  # peak center
        A =  parameters[1]  # peak amplitude
        B =  parameters[2]  # background
        # constants
        d = constants[0]

        # and now our model 'fitting function' for the experiment
        return B + A / (((x - x0) / d) ** 2 + 1)
        # OK, this is just a one-liner, but the model could be much more complex.

    # create a server
    nanny = OBE_Server()
    # connect the model
    nanny.model_function = lorentzian_model

    # wait for commands over TCP & respond.
    nanny.run()
