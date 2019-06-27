from json import dumps, loads
from socket import socket, AF_INET, SOCK_STREAM

class Socket:
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
        return reply


