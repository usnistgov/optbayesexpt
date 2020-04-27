import numpy as np
from numpy.testing import assert_array_equal

from optbayesexpt import Socket


myClient = Socket('client', port=60899)
myServer = Socket('server', port=60899)

def test_send_recieve_text():
    contents = 'test message'
    myClient.send(contents)
    received = myServer.receive()
    myClient.close()
    assert received == contents

def test_send_recieve_number():
    contents = 3.1415
    myClient.send(contents)
    received = myServer.receive()
    myClient.close()
    assert received == contents

def test_send_recieve_array():
    contents = [1, 2, 3]
    myClient.send(contents)
    received = myServer.receive()
    myClient.close()
    assert received == contents

def test_send_receive_dict():
    contents = {'message': 'Howdy!', 'pi': 3.1415, 'count': [1, 2, 3]}
    myClient.send(contents)
    received = myServer.receive()
    myClient.close()
    assert received == contents

"""
The tcpcmd() method is tested in helpme_test_server.py
"""
if __name__ == '__main__':
    test_send_recieve_text()
    test_send_recieve_number()
    test_send_recieve_array()
    test_send_receive_dict()
