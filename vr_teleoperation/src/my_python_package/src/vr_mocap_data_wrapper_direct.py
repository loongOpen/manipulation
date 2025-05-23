import asyncio
import struct
import threading
import time
from multiprocessing import shared_memory
from queue import Queue

import numpy as np

from mocap2robot_src.Mocap2Robot_Vuer2Qinlong import Vuer2Qinlong
from mocap2robot_src.TeleVision.teleop.TeleVision import OpenTeleVision,Vuer2Data

from mocap2robot.srv import MocapCalibrate, MocapCalibrateResponse

from vr_mocap_data_wrapper import VRMocapDataWrapper


class VRMocapDataWrapperDirect(VRMocapDataWrapper):
    def __init__(self):
        super().__init__()


        self.vuer2data = Vuer2Data()

        resolution = (720, 1280)
        crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
        crop_size_h = 270
        resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
        img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
        img_height, img_width = resolution_cropped[:2]  # 450 * 600
        shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)

        shm_name = shm.name
        self.img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)

        image_queue = Queue()
        toggle_streaming = asyncio.Event()
        # tv = OpenTeleVision(resolution_cropped, cert_file="../cert.pem", key_file="../key.pem")
        self.tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming,
                                 cert_file="/home/enpht/catkin_ws/src/body_map_vision_pro/src/mocap2robot_src/TeleVision/teleop/cert.pem",
                                 key_file="/home/enpht/catkin_ws/src/body_map_vision_pro/src/mocap2robot_src/TeleVision/teleop/key.pem")

    def set_data(self):
        self.data=self.vuer2data.process(self.tv, self.data)

    def get_data(self):
        return self.data