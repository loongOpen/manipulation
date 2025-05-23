#!/usr/bin/env python
'''
accept vuer vr support
adapt devices: quest3 and vision pro
drive QinLong robot

TODO: finger
TODO:
'''

import os

# printing environment variables
# print(os.environ)
import sys


# sys.path.append("..")

import asyncio
from enum import Enum
# from multiprocessing import shared_memory
# from queue import Queue

import cv2
import numpy as np
import rospy
import threading

import signal
import socket
import struct




class SubThread():
    def __init__(self):
        # start one thread to receive data

        t1 = threading.Thread(target=self.receive_data)
        t1.start()

    def receive_data(self):
        server_ip = "0.0.0.0"  # IP address to listen on
        server_port = 5015  # Port to listen on
        BUFFER_SIZE = 8888  # Buffer size for incoming messages

        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((server_ip, server_port))
        self.sock.settimeout(1.0)  # Set a timeout of 1 second

        # print("UDP receiver started")

        while not rospy.is_shutdown():
            try:
                # print("1111111")
                data_bytes, addr = self.sock.recvfrom(BUFFER_SIZE)  ####receive
                # print("2222222")
                # print(data_bytes)
                if not data_bytes:
                    print('not data_bytes')
                    break

                l = len(data_bytes)
                # print("1234455",l)
            except socket.timeout:
                # If no data is received within timeout, continue loop to check shutdown_event
                print("no data")
                continue
            except socket.error:
                print("socket.error")
                break
        
        self.sock.close()
        print("UDP receiver stopped")
        
        print("sub thread exit")




class VRMocapManager:
    def __init__(self):
        rospy.init_node('mocap_manager_node')


        # Publish the TF message at a rate of 10 Hz
        self.tf_publish_rate = rospy.Rate(60)
        self.sub_thread=SubThread()



    def run(self):


        # vp_control_pub = rospy.Publisher('/vp_control', VpControlData, queue_size=10)


        print("start main loop")

        while not rospy.is_shutdown():


            self.tf_publish_rate.sleep()


# def signal_handler(sig, frame):
#     print("cleaning up shared memory")
#     # shm.close()
#     sys.exit(0)


if __name__ == '__main__':
    # signal.signal(signal.SIGTERM, signal_handler)
    print("VRMocapManager start!")
    vrMocapManager = VRMocapManager()

    # from pynput import keyboard

    # listener = keyboard.Listener(
    #     on_release=on_release)
    # listener.start()

    vrMocapManager.run()
