from .obe import OptBayesExpt
from .obe_socket import Socket

class OBE_Server(Socket, OptBayesExpt):
    def __init__(self, ip_address='127.0.0.1', port=61981):
        Socket.__init__(self, 'server', ip_address=ip_address, port=port)
        OptBayesExpt.__init__(self)

        self.noise = 1.0
        self.Ndraws = 100

    def newrun(self, *args):
        # stub for a python script using data passed from client
        pass

    def getmean(self):
        # stub for a python script handling data request from client
        return self.getmean(0)

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
                self.sets += (message['array'], )
                self.send('OK')
            elif 'addpar' in message['command']:
                self.pars += (message['array'], )
                self.send('OK')
            elif 'addcon' in message['command']:
                self.cons += (message['value'], )
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
            elif 'getpdf' in message['command']:
                self.send(list(self.PDF))
            elif 'maxpdf' in message['command']:
                self.send(self.max_params())
            elif 'getmean' in message['command']:
                self.send(self.getmean())

            elif 'done' in message['command']:
                self.send('OK')
                break
            else:
                pass
