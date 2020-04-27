from json import dumps, loads
from socket import socket, AF_INET, SOCK_STREAM


class Socket:
    """Handles TCP communications

    The :code:`Socket` can be configured either as a 'server' or a 'client'.
    Server sockets wait for connections, receive messages and send replies.
    Client sockets initiate connections and receive replies.

    The message protocol uses messages formatted as JSON strings, each
    prependend by the string length as a zero-padded, 10-digit decimal
    number.  The general form is

        dddddddddd<JSON-formatted string>

    Command messages from the client use a JSON :obj:`object`:

        dddddddddd{"command": <command_str>[, <label_str>: <value_str>[, ...]].

    Example messages
        * :code:`0000000038{"command": "goodset", "pickiness": 4}`
        * :code:`0000000019{"command": "done"}`
        * :code:`0000000004"OK"`

    Args:
        role (str): either 'client' to configure the Socket to initiate
            communications or 'server' to listen and respond.

        ip_address (str): Identifies the computer host to communicate with.
            The default of '127.0.0.1' is the localhost,  enabling
            communications between processes on the host computer.

        port (int): the TCP port used for communications.  The default value
            61981 was chosen chosen randomly in the unassigned port range
            49152 to 65535.

    Attributes:
        server: for the 'server' role, a :code:`socket.socket` object
            configured to listen and accept connections
        connection: for the 'client' role, a :code:`socket.socket` object
            configured to initiate connections and send messages
    """

    def __init__(self, role, ip_address='127.0.0.1', port=61981):

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
            raise Exception(
                'Invalid role {}. Valid choices are \
                    client or server.'.format(role))

    def send(self, contents):
        """
        Formats and sends a message

        This method formats the :code:`contents` argument into the message
        format, opens a connection and sends the :code:`contents` as a message.

        Args:
            contents: Any JSON format-able object. Briefly, python's
                :obj:`str`, :obj:`int`, :obj:`float`, :obj:`list`,
                :obj:`tuple`, and :obj:`dict` objects.

        Important:
            json.dumps() is not able to format numpy arrays.  To send numpy
            arrays, the :code:`numpy.tolist()` method is a convenient way to
            list-ify a numpy array.  For example::

                mySocket.send(myarray.tolist())
        """

        if self.role == 'client':
            self.connection = socket(AF_INET, SOCK_STREAM)
            self.connection.connect((self.ip_address, self.port))

        json = dumps(contents).encode()
        jdatalen = '{:0>10d}'.format(len(json)).encode()
        message = jdatalen + json
        # print(message)
        self.connection.sendall(message)

    def receive(self):
        """Wait for and process messages on the TCP port

        Blocks until a connection is made, then reads the number of bytes
        specified in the first 10 characters.  Reads the connection until
        the full message is received, then decodes the messages string.

        Returns:
            The message string decoded and repackaged as a python object
        """

        gulp = 1024
        while True:
            if self.role == 'server':
                # accept() method blocks until a connection is made
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
                chunk = self.connection.recv(
                    min(message_len - bytes_recd, gulp))
                if chunk == b'':
                    raise RuntimeError("socket connection broken")
                raw_message += chunk
                bytes_recd = bytes_recd + len(chunk)
            return loads(raw_message.decode())

    def close(self):
        """Close the communication connection.

        Only clients need to close connections once they're done communicating.
        No need to call this for servers.
        """
        self.connection.close()
        self.connection = None

    def tcpcmd(self, command):
        """Sends a command and receives a response.

        Run from a client socket, this method sends a command message and
        receives a response. The connection is then closed.

        Args:
            command: a JSON-ifiable python object to be interpreted by the
            recipient.

        Returns:
            a pyhton object decoded from the reply message
        """
        if self.role == 'client':
            self.send(command)
            reply = self.receive()
            self.connection.close()
            return reply
