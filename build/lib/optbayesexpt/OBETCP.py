from json import dumps, loads
from socket import socket, AF_INET, SOCK_STREAM
from optbayesexpt import OptBayesExpt


class OBE_Socket():
    def __init__(self, role, ip_address='127.0.0.1', port=20899):
        """
        Create a simplified TCP socket which can act as a server or client.
        :param role: 'server' tells the socket to wait and listen for someone to connect.
        'client' tells the socket to connect to a server.
        :param ip_address: Which computer do you want to talk to? Specify its IP address.
        The default of 127.0.0.1 means "the same computer I'm on". Sometimes you just have to
        talk to yourself.
        :param port: The server will listen on this TCP port for communication. The client will
        connect to this port.  ZIP of the NIST campus in MD: 20899.
        """
        self.role = role
        self.ip_address = ip_address
        self.port = port
        self.connection = None
        if self.role == 'client':
            pass
            # Client will connect as needed.
        elif self.role == 'server':
            self.server = socket(AF_INET, SOCK_STREAM)
            self.server.bind((self.ip_address, self.port))
            self.server.listen(1)
        else:
            raise Exception('Invalid role. Valid choices are client or server.'.format(role))

    def send(self, contents):
        """
        Send a message to the other computer. A server may only call this function after
        receive() is called.
        The message has a 10 character header describing length
        """
        json = dumps(contents).encode()
        jdatalen = '{:0>10d}'.format(len(json)).encode()
        message = jdatalen + json
        print(message)
        self.connection.sendall(message)

    def receive(self):
        """
        Receive a message from the other computer. If a connection to another computer does not
        exist then the function will wait until a connection is established.
        :return: Returns a message received from the other computer.
            This function will block until a message is received.
        """
        gulp = 1024
        while True:
            if self.role == 'server':
                self.connection, address = self.server.accept()
            bitcount = b''
            bytes_recd = 0
            while bytes_recd < 10:
                chunk = self.connection.recv(10 - bytes_recd)
                if chunk == b'':
                    raise RuntimeError("socket connection broken")
                bitcount += chunk
                bytes_recd = bytes_recd + len(chunk)
            message_len = int(bitcount)

            raw_message = b''
            bytes_recd = 0
            while bytes_recd < message_len:
                chunk = self.connection.recv(min(message_len - bytes_recd, gulp))
                if chunk == b'':
                    raise RuntimeError("socket connection broken")
                raw_message += chunk
                bytes_recd = bytes_recd + len(chunk)
            return loads(raw_message.decode())

    def close(self):
        """
        Close the communication connection. Only clients need to close connections once they're
        done communicating.
        No need to call this for servers.
        """
        self.connection.close()
        self.connection = None

    def tcpcmd(self, command):
        self.connection = socket(AF_INET, SOCK_STREAM)
        self.connection.connect((self.ip_address, self.port))
        self.send(command)
        reply = self.receive()
        self.connection.close()
        return(reply)

class OBE_Server(OBE_Socket, OptBayesExpt):
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
    def lorentzian_model(self, settings, parameters, constants):
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
