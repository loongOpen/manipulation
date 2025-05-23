

import struct
import numpy as np


class Mocap2Robot:


    def __init__(self):
        self.head_pose_save= np.eye(4)
        self.head_pose_new= np.eye(4)
        self.tf_he_robot_to_l_s_robot = np.array([[1, 0, 0, -0.2],
                                   [0, 1, 0, -0.2],
                                   [0, 0, 1, -0.],
                                   [0, 0, 0, 1]])

        self.tf_he_robot_to_r_s_robot = np.array([[1, 0, 0, 0.2],
                                    [0, 1, 0, -0.2],
                                    [0, 0, 1, -0.],
                                    [0, 0, 0, 1]])
        # self.tf_he_robot_to_l_s_robot = np.array([[1, 0, 0, -0.2],
        #                                           [0, 1, 0, 0.9],
        #                                           [0, 0, 1, 0.2],
        #                                           [0, 0, 0, 1]])
        #
        # self.tf_he_robot_to_r_s_robot = np.array([[1, 0, 0, 0.2],
        #                                           [0, 1, 0, 0.9],
        #                                           [0, 0, 1, 0.2],
        #                                           [0, 0, 0, 1]])
        pass

    def calibrate_shoulder(self):
        self.head_pose_save = self.head_pose_new.copy()

    def get_hand_tf_quest_fixed_head(self, group_to_head_new, group_to_left_hand, group_to_right_hand):
        pass

    def handle_raw_data(self, data_bytes):
        '''
        if l == 4 * 49 * 7:
                       ...
                       fun(data_bytes)
        '''
        float_array = struct.unpack(f'{49 * 7}f', data_bytes)
        xyzqwqxqyqz = np.array(float_array).reshape((49, 7))
        return xyzqwqxqyqz

    def process(self, xyzqwqxqyqz):
        pass
