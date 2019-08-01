from .obe import OptBayesExpt
from .obe_socket import Socket


class OBE_Server(Socket, OptBayesExpt):
    """
    A TCP socket interface for OptBayesExpt using TCP communications.

    This class provides a way to use OptBayesExpt from programs written in other languages.  A
    typical use would be to run OptBayesExpt from external instrument control software.  The
    external program must

        1. Launch a server script that configures and runs an :code:`OBE_Server`,
        2. Format command messages as JSON strings,
        3. Send and receive messages via TCP sockets,
        4. Close the server script when done.

    Server Script:
        The server script configures an instance of OptBayesExpt and then runs in the background,
        listening for commands and responding until a :code:`done` command is received.

        Example:

            This server script is written for an application where we want to make several
            measurement runs, and we want to supply a setting range for each run.  The script
            creates an :code:`OBE_server` (:code:`nanny`, to take care of things).  The
            :code:`lorentz_model()` function defines the parametric model.  The other function,
            :code:`newrun_near_startpoint()`, customizes the behavior of the "newrun" TCP command
            in the :code:`OBE_server.run()` method.::

                import optbayesexpt as obe
                import numpy as np

                # create a server
                nanny = obe.OBE_Server()


                # define the parametric model
                def lorentz_model(settings, parameters, constants):
                    # model for a Lorentzian peak
                    # unpack settings - pulsetime is the first in a 1-element tuple
                    frequency, = settings
                    # unpack b1, scale and offset from a 3-element tuple
                    f0, amplitude, lw = parameters
                    # calculate the model function
                    return amplitude / (((frequency - f0) * 2 / lw)**2 + 1)


                # customized *newrun* behavior
                def newrun_near_startpoint(datatuple):
                    #  configures settings, parameters

                    centerfreq, span = datatuple
                    f0vals = np.linspace(centerfreq-span/2, centerfreq+span/2, 400)
                    amplitude = np.linspace(.01, 0.1, 50)
                    lw = np.linspace(.003, .050, 50)*1e9

                    nanny.sets = (f0vals,)
                    nanny.pars = (f0vals, amplitude, lw)
                    nanny.cons = ()

                    nanny.config()


                # connect the customized functions by redefining default stubs
                nanny.model_function = lorentz_model
                nanny.newrun = newrun_near_startpoint

                # wait for commands over TCP & respond until a 'done' command is received.
                nanny.run()


    Command Language
        The messages sent from client to this server are JSON object strings, each prependend by the
        string length formatted as a 10-digit number.  See the obe_socket module docs for details.
        Unless otherwise stated the server will reply with :code:`'0000000004"OK"'` upon
        successful completion of the command.

        Setup commands
            These commands are used in defining the settings, parameters and constants,
            and in using these definitions to configure.

            clrset
                clears the :code:`sets` tuple, :code:`OptBayesExpt.sets = ()`.
                    :code:`{"command": "clrset"}`

            clrpar
                clears the :code:`pars` tuple, :code:`OptBayesExpt.cons = ()`.
                    :code:`{"command": "clrpar"}`

            clrcon
                clears the :code:`cons` tuple, :code:`OptBayesExpt.cons = ()`.
                    :code:`{"command": "clrcon"}`

            addset
                appends a setting array to the :code:`OptBayesExpt.sets` tuple.
                    :code:`{"command": "addset", "array": <JSON list>}`

            addpar
                appends a parameter array to the :code:`OptBayesExpt.pars` tuple.
                    :code:`{"command": "addpar", "array": <JSON list>}`

            addcon
                appends a constant to the :code:`OptBayesExpt.cons` tuple.
                    :code:`{"command": "addcon", "value": <number>}`

            getset
                Reports the :code:`OptBayesExpt.sets` attribute

                    :code:`{"command": "getset"}`

                Reply: a list of setting value lists.

            getpar
                Reports the :code:`OptBayesExpt.pars` attribute

                    :code:`{"command": "getset"}`

                Reply: a list of parameter value lists.

            getcon
                Reports the :code:`OptBayesExpt.cons` attribute

                    :code:`{"command": "getcon"}`

                Reply: a list of model constants.

            config
                configures the :code:`OptBayesExpt` instance using the setting, parameter and
                constant tuples. Invokes :code:`OptBayesExpt.config()`

                    :code:`{"command": "config"}`

            newrun
                a stub to run the :code:`newrun` method.
                    :code:`{"command": "newrun", "array": <JSON list>}`


        Run time commands

            These  commands are used in every measurement cycle, generating measurement
            settings and processing measurement results.

            optset
                Genrates measurement settings.  Calculates the utility of possible settings and
                reports the settings having the maximum utility.
                Invokes :code:`OptBayesExpt.opt_setting()`

                    :code:`{"command": "optset"}`

            goodset
                Generated measurement settings. Calculates the utility of possible settings and
                reports a random selection that is weighted by the utility.
                Invokes :code:`OptBayesExpt.good_setting()`

                    :code:`{"command": "goodset"[, "pickiness", <integer>]}`

            newdat
                Processes measurement results. Refines the parameter probability distribution
                based on the *likelihood* of measurement results reported in the command.
                Invokes :code:`OptBayesExpt.pdf_update()`.  Required components include
                * "x": a tuple containing the measurement settings
                * "y": a measurement mean value
                * "s": uncertainty of the measurement as a standard deviation

                    :code:`{"command": "newdat", "x": <settings tuple>, "y": <float>, "s": <float>}`


        Information requests

            These commands report cumulative measurement results based on the parameter
            probability distribution.

            getpdf
                Reports the entire probability distribution array

                    :code:`{"command": "getpdf"}`

                Reply: Probability distribution function array expressed as a JSON array [of arrays].

            maxpdf
                Reports the parameters corresponding to the maximum of the probability
                distribution. Invokes :code:`OptBayesExpt.max_params()`

                    :code:`{"command": "maxpdf"}`

                Reply: A single set of parameters packaged as a JSON array.

            getmean
                Reports a parameter mean value and standard deviation.  The parameter is
                selected by the "index" argument, which corresponds to the parameter's position in
                the :code:`OptBayesExpt.pars` tuple  The default "index" is :code:`0`.

                    :code:`{"command": "getmean"[, "index": <int> ]}`

                Reply: :code:`{"mean": <float>, "sigma": <float>}`


        Cleaning up

            done
                Stops the :code:`OBE_server` and allows the server script to complete.
                    :code:`{"command": "done"}`


    Warning:

        It may be important to issue the :code:`done` command. If the server is allowed to run,
        it will continue to use the TCP socket it was assigned during initialization.  Later
        instances of :code:`OBE_Server` may conflict.

    """

    def __init__(self, ip_address='127.0.0.1', port=61981):
        Socket.__init__(self, 'server', ip_address=ip_address, port=port)
        OptBayesExpt.__init__(self)

        self.Ndraws = 100

    def newrun(self, *args):
        """
        Stub to allow customized behavior of the :code:`"newrun"` TCP command

        The intent is to support a user-defined function that can reset :code:`OptBayesExpt.sets`
        and :code:`OptBayesExpt.params` arrays and to allow an :code:`OptBayesExpt` instance to
        be recycled for repeated measurement runs.
        """
        pass

    def run(self):
        """
        Listen and respond to TCP messages

        Enters a continuous loop, responding to TCP command messages until a "done" command is
        received.
        """
        print()
        print('SERVER READY')
        while True:
            # use the Socket.receive() method to get the incoming message from client
            message = self.receive()
            # the messages we get from Labview are Labview clusters encoded as json objects
            # Decoded, they are python dicts.

            # manipulate coinfiguration arrays for settings, params, consts.
            # clear commands
            if 'clrset' in message['command']:
                self.sets = ()
                self.send('OK')
            elif 'clrpar' in message['command']:
                self.pars = ()
                self.send('OK')
            elif 'clrcon' in message['command']:
                self.cons = ()
                self.send('OK')

            # get commands request arrays
            elif 'getset' in message['command']:
                self.send(self.sets)
            elif 'getpar' in message['command']:
                self.send(self.pars)
            elif 'getcon' in message['command']:
                self.send(self.cons)

            # add arrays as new tuple elements
            elif 'addset' in message['command']:
                self.sets += (message['array'],)
                self.send('OK')
            elif 'addpar' in message['command']:
                self.pars += (message['array'],)
                self.send('OK')
            elif 'addcon' in message['command']:
                self.cons += (message['value'],)
                self.send('OK')

            # Finish configuration
            elif 'config' in message['command']:
                self.config()
                self.send('OK')

            # run-time commands
            elif 'newrun' in message['command']:
                self.newrun(message['array'])
                self.send('OK')

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

            # report
            elif 'getpdf' in message['command']:
                self.send(list(self.PDF))
            elif 'maxpdf' in message['command']:
                self.send(self.max_params())
            elif 'getmean' in message['command']:
                if 'index' in list(message):
                    mean, std = self.get_mean(message['index'])
                else:
                    mean, std = self.get_mean(0)
                self.send(self.getmean())

            elif 'done' in message['command']:
                self.send('OK')
                break
            else:
                pass
