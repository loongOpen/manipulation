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



sys.path.append("..")

import asyncio
from enum import Enum
from multiprocessing import shared_memory
from queue import Queue

import cv2
import numpy as np
import rospy


class VRMocapDataWrapper:
    def __init__(self):
        super().__init__()
        self.data_length = 222
        self.data = np.zeros(self.data_length)
        self.data[:16] = np.eye(4).flatten()
        self.data[16:32] = np.eye(4).flatten()
        self.data[32:48] = np.eye(4).flatten()
        self.data[198:] = 0



    def get_data(self):
        return self.data

    def set_image(self,frame):
        return 0






