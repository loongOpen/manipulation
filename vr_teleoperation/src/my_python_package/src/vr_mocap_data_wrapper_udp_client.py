import struct
import threading
import time
from multiprocessing import shared_memory
from queue import Queue

import numpy as np
import asyncio

from mocap2robot_src.TeleVision.teleop.TeleVision import OpenTeleVision, Vuer2Data
import socket
import struct


class VRMocapDataWrapperUdpClient:
    def __init__(self):
        self.data_length = 222
        self.data = np.zeros(self.data_length)
        self.data[:16] = np.eye(4).flatten()
        self.data[16:32] = np.eye(4).flatten()
        self.data[32:48] = np.eye(4).flatten()
        self.data[198:] = 0

        self.vuer2data = Vuer2Data()

        resolution = (720, 1280)
        crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
        crop_size_h = 270
        resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
        img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
        img_height, img_width = resolution_cropped[:2]  # 450 * 600
        # shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)

        shm_name = "image_shared_memory"
        # self.img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)

        image_queue = Queue()
        toggle_streaming = asyncio.Event()
        # tv = OpenTeleVision(resolution_cropped, cert_file="../cert.pem", key_file="../key.pem")
        self.tv = OpenTeleVision(resolution_cropped, shm_name, image_queue, toggle_streaming,
                                 cert_file="/home/enpht/catkin_ws/src/body_map_vision_pro/src/mocap2robot_src/TeleVision/teleop/cert.pem",
                                 key_file="/home/enpht/catkin_ws/src/body_map_vision_pro/src/mocap2robot_src/TeleVision/teleop/key.pem")

        # Define the server address and port
        self.server_address = ('localhost', 10633)

        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_data(self):
        while True:
            self.data = self.vuer2data.process(self.tv, self.data)

            packed_data = struct.pack(f'{self.data_length}f', *self.data)
            sent = self.sock.sendto(packed_data, self.server_address)
            time.sleep(0.030)

    def receive_image(self):
        pass

    def run(self):
        data_send_thread = threading.Thread(target=self.send_data)
        receive_image_thread = threading.Thread(target=self.receive_image)

        data_send_thread.start()
        data_send_thread.join()
        while True:
            time.sleep(1)


if __name__=="__main__":
    app= VRMocapDataWrapperUdpClient()
    app.run()