import math
import struct
import socket
from time import sleep

import numpy as np


class UdpIkSender:
    def __init__(self):
        self.client_host = "127.0.0.1"
        self.client_port = 5015
        BUFFER_SIZE = 1024

        # 创建UDP套接字
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 绑定客户端地址和端口
        # client_socket.bind((client_host, client_port))

    def send(self, message):
        packed_data = b''.join([struct.pack('<f', num) for num in message])  # simulation >
        self.client_socket.sendto(packed_data, (self.client_host, self.client_port))
        # #print('send')


if __name__=="__main__":
    udp=UdpIkSender()
    msg=np.zeros((48,7),float)
    msg[:,3]=1
    msg[1, :] = 1
    t=0

    while True:
        t+=0.1
        msg[0, 0] = math.sin(t)*0.100+0.100
        udp.send(msg.ravel())
        sleep(0.1)