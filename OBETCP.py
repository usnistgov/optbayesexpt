from json import dumps, loads
from socket import socket, AF_INET, SOCK_STREAM
from OptBayesExpt import OptBayesExpt

class Socket:
    def __init__(self, role, ip_address='127.0.0.1', port=31415):
        """
        Create a simplified TCP socket which can act as a server or client.
        :param role: 'server' tells the socket to wait and listen for someone to connect. 'client' tells the
        socket to connect to a server.
        :param ip_address: Which computer do you want to talk to? Specify its IP address.
        The default of 127.0.0.1 means "the same computer I'm on". Sometimes you just have to talk to yourself.
        :param port: The server will listen on this TCP port for communication. The client will connect to this port.
        """
        self.role = role
        self.ip_address = ip_address
        self.port = port
        self.connection = None
        if role == 'client':
            self.connection = socket(AF_INET, SOCK_STREAM)
            self.connection.connect((self.ip_address, self.port))
        elif role == 'server':
            self.server = socket(AF_INET, SOCK_STREAM)
            self.server.bind((self.ip_address, self.port))
            self.server.listen(1)
        else:
            raise Exception('Invalid role. Valid choices are client or server.'.format(role))

    def send(self, contents):
        """
        Send a message to the other computer. A server may only call this function after receive() is called.
        The message has a 12 character header:  " + 10 chars describing length + "
        """
        json = dumps(contents).encode()
        jdatalen = dumps('{:0>10d}'.format(len(json))).encode()
        message =  jdatalen + json
        print(message)
        self.connection.sendall(message)

    def receive(self):
        """
        Receive a message from the other computer. If a connection to another computer does not exist
        then the function will wait until a connection is established.
        :return: Returns a message received from the other computer.
            This function will block until a message is received.
        """
        gulp = 1024
        while True:
            if self.role == 'server':
                self.connection, address = self.server.accept()
            # our protocol includes 10 bytes of message size
            sizestr = self.connection.recv(10).decode()
            if sizestr != None:
                bytestoread = int(sizestr)
                raw_message = b''
                nextgulp = gulp
                while bytestoread > 0:
                    if bytestoread < gulp:
                        nextgulp = bytestoread
                    newchunk = self.connection.recv(nextgulp)
                    raw_message += newchunk
                    bytestoread -= gulp
                contents = loads(raw_message.decode())
                print(contents)
                return contents

    def close(self):
        """
        Close the communication connection. Only clients need to close connections once they're done communicating.
        No need to call this for servers.
        """
        self.connection.close()
        self.connection = None


class BOE_Server(Socket, OptBayesExpt):
    def __init__(self, ip_address='127.0.0.1', port=31415):
        Socket.__init__(self, 'server', ip_address=ip_address, port=port)
        OptBayesExpt.__init__(self)

        self.noise = 1.0
        self.Ndraws = 100

    def model_function(self, settings, parameters, constants):
        # unpack our input tuples
        # experimental settings
        x = settings[0]
        # model parameter
        x0 = parameters[0]   ;# peak center
        A =  parameters[1]   ;# peak amplitude
        B =  parameters[2]   ;# background
        # constants
        d = constants[0]

        # and now our model 'fitting function' for the experiment
        return B + A / ( ((x - x0) / d) ** 2 + 1 )
        # OK, this is just a one-liner, but the model could be much more complex.


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
    nanny = BOE_Server()
    nanny.run()




