"""
This script is a simulated measurement program that communicates with an
OBE_Server via TCP sockets. The server is configured with a Lorentzian model,
and unknown parameters: center frequency, amplitude, and line width.

###################################################################.
# NOTE: sections where this script interacts with the server script
# are marked with #### - like this section.
###################################################################.
"""
import matplotlib.pyplot as plt
import numpy as np

from json import dumps, loads
from socket import socket, AF_INET, SOCK_STREAM

from time import sleep
from subprocess import Popen
import os

# defaults
default_ip_address = '127.0.0.1'
# the server must use the same port
default_port = 61981
# simulated noise level
sigma_y = 0.005


def main():
    """The main script of this demo

    Called at the end of this file after all of the other definitions
    """

    # Start the server script in a separate process
    ##################################################################
    server_script = os.path.join(os.getcwd(), "server_script.py")
    try:
        server_pipe = Popen(['python3', server_script])
    except FileNotFoundError:
        server_pipe = Popen(['python', server_script])
    ##################################################################

    while True:
        try:
            tcpcmd({'command': 'ready'})
            break
        except ConnectionRefusedError:
            sleep(.1)

    # measure & plot 3 runs
    print(
        "3 runs, each measuring a randomly located peak and specified "
        "measurement settings")
    print()
    print(
        "25 measurement settings between 2 GHz and 3 GHz. Notice the "
        "discrete settings")
    measurement_run(2, 3, 25)
    print()
    print(
        "Higher resolution now. 200 measurement settings between 2 GHz and 3 "
        "GHz.  ")
    measurement_run(2, 3, 200)
    print()
    print("A tricky one.  200 Measurement settings from 1 GHz to 4 GHz.")
    print(
        "The measurements probably won't extend all the way from 1 to 4, "
        "though.")
    print(
        "This is because the prior (see server script) only expects center "
        "values from 1.5 to 3.5")
    measurement_run(1, 4, 200)

    # End the server script
    ###############################################
    tcpcmd({'command': 'done'})
    ###############################################


# function to simulate measurements: a Lorentzian curve with noise.
def fake_measure(f, truef0=2.45, trueampl=.02, truelw=.040,
                 noiselevel=sigma_y):
    return 1 - trueampl / (((f - truef0) * 2 / truelw) ** 2 + 1) \
           + noiselevel * np.random.randn()


def measurement_run(low_limit, high_limit, steps):
    """ performs a simulated measurement run and plots the results
    """
    ##########################################################
    # configure the OBE_server for a new run
    # compose a command message
    message = {'command': 'newrun',
               'lolim': low_limit,
               'hilim': high_limit,
               'steps': steps}
    # send it to the server
    tcpcmd(message)
    ##########################################################

    # trails to collect accumulating data for plotting
    f_trail = []
    y_trail = []
    f0_trail = []
    s_trail = []

    # pick a random peak frequency for the simulated measurement"
    truef = (np.random.random() * .8 + 2.1)

    # perform 200 measurements using sequential Bayesian experimental design
    for _ in np.arange(200):
        # trial_f, = obe.tcpcmd({'command': 'goodset', 'pickiness': 9})

        # get a new recommended setting
        #####################################################################
        # f_setting, = tcpcmd({'command': 'optset'})
        f_setting, = tcpcmd({'command': 'goodset', 'pickiness': 6})
        #####################################################################

        # simulate a measurement result
        meas_y = fake_measure(f_setting, truef0=truef)

        # report measurement results to the OBE_Server
        ######################################################################
        tcpcmd({'command': 'newdat', 'x': (f_setting,), 'y': meas_y,
                's': sigma_y})
        #####################################################################

        f_trail.append(f_setting)
        y_trail.append(meas_y)

        # get statistics for plotting
        ###################################################################
        mean = tcpcmd({'command': 'getmean'})
        sigma = tcpcmd({'command': 'getstd'})
        ###################################################################
        f0_trail.append(mean[0])
        s_trail.append(sigma[0])

    # Simulated measurements are done.  The rest is plotting ...
    ftrail = np.array(f_trail)
    ytrail = np.array(y_trail)

    plt.figure(figsize=(4, 6))

    plt.subplot(311)
    plt.plot(ftrail, ytrail, 'k.')
    f = np.linspace(low_limit, high_limit, steps)
    plt.plot(f, fake_measure(f, truef0=truef, noiselevel=0))
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Signal')

    plt.subplot(312)
    plt.plot(f_trail, 'b.')
    plt.xlabel("Iteration")
    plt.ylabel("frequency setting (GHz)")

    snp = np.array(s_trail)
    err = np.abs(np.array(f0_trail) - truef)

    plt.subplot(313)
    iteration = np.arange(len(snp)) + 1
    plt.loglog(iteration, snp*1000, label='sigma')
    plt.loglog(iteration, err*1000, label='error')
    plt.ylim(.1, 1000)
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Center uncertainty (MHz)")
    plt.tight_layout()
    print('    "X" out to proceed.')
    plt.show()


# -----------------------------
# TCP communication routines
#

def connect(ip_address, port):
    """Start up a socket to communicate with the server

    Args:
        ip_address:
        port:

    Returns: the connected socket
    """
    connection = socket(AF_INET, SOCK_STREAM)
    connection.connect((ip_address, port))
    return connection


def send(connection, contents):
    """
    Format and send a message

    This method formats the :code:`contents` argument into the message format,
    opens a connection to a server and sends the message.

    Args:
        connection: a connected socket
        contents: Any JSON format-able object. Briefly, python's :obj:`str`,
        :obj:`int`,
            :obj:`float`, :obj:`list`, :obj:`tuple`, and :obj:`dict` objects.

    Important:
        json.dumps() is not able to format numpy arrays.  To send numpy
        arrays, the
        :code:`numpy.tolist()` method is a convenient way to list-ify a
        numpy array.  For
        example::

            mySocket.send(myarray.tolist())
    """
    json = dumps(contents).encode()
    jdatalen = '{:0>10d}'.format(len(json)).encode()
    message = jdatalen + json
    # print(message)
    connection.sendall(message)


def receive(connection):
    """Wait for and process messages on the TCP port

    Blocks until a connection is made, then reads the number of bytes
    specified in the first 10
    characters.  Reads the connection until the full message is received,
    then decodes the
    messages string.

    Returns:
        The message string decoded and repackaged as a python object
    """
    gulp = 1024
    # while True:
    # Read length of message encoded in first 10 chars.
    bytecount = b''
    bytes_recd = 0
    while bytes_recd < 10:
        chunk = connection.recv(10 - bytes_recd)
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        bytecount += chunk
        bytes_recd = bytes_recd + len(chunk)
    message_len = int(bytecount)
    # read the body of the message
    raw_message = b''
    bytes_recd = 0
    while bytes_recd < message_len:
        chunk = connection.recv(min(message_len - bytes_recd, gulp))
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        raw_message += chunk
        bytes_recd = bytes_recd + len(chunk)
    return loads(raw_message.decode())


def tcpcmd(command):
    global default_ip_address, default_port
    connection = connect(default_ip_address, default_port)
    send(connection, command)
    reply = receive(connection)
    connection.close()
    return reply


if __name__ == "__main__":
    main()
