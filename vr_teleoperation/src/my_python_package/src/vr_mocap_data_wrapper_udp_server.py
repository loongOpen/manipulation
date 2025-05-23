import struct
import socket
import threading
from multiprocessing import shared_memory

import cv2
import numpy as np
import rospy

from vr_mocap_data_wrapper import VRMocapDataWrapper


class VRMocapDataWrapperUdpServer(VRMocapDataWrapper):
    def __init__(self):
        super().__init__()


        resolution = (720, 1280)
        crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
        crop_size_h = 270
        resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
        self.img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
        img_height, img_width = resolution_cropped[:2]  # 450 * 600

        try:
            rospy.loginfo("try close shared memory image_shared_memory")
            self.shm = shared_memory.SharedMemory(name='image_shared_memory')
            self.shm.close()
            self.shm.unlink()
        except Exception as e:
            rospy.loginfo(e)
            rospy.loginfo("if image_shared_memory not found, it's ok because we need to create one")

        self.shm = shared_memory.SharedMemory(name='image_shared_memory',create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)

        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)

    def run(self):
        set_data_thread=threading.Thread(target=self.set_data)
        set_data_thread.start()

    def set_data(self):

        host = 'localhost'
        port = 10633
        # 创建 UDP 套接字
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind((host, port))

        print(f"UDP server started at {host}:{port}")

        while not rospy.is_shutdown():
            # 接收数据
            # 接收数据
            data, client_address = server_socket.recvfrom(1024)
            # print(f"Received message from {client_address}")

            float_array = struct.unpack(f'{len(data) // 4}f', data)
            # float_array = np.array(float_array)
            for i in range(self.data_length):
                self.data[i] = float_array[i]

            byte_data = struct.pack(f'{len(float_array)}f', *float_array)
            server_socket.sendto(byte_data, client_address)

    def set_image(self,frame):
        # shared memory way
        image = cv2.resize(frame, (self.img_shape[1], self.img_shape[0]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # cv2.imshow("canvas",frame)
        # cv2.waitKey(30)

        np.copyto(self.img_array, image)


        return 0


