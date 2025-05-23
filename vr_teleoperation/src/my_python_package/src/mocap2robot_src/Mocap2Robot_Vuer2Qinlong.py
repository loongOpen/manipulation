import socket
import struct

import numpy as np


from .HelpFuction import rpy2rotation_matrix, fast_mat_inv
from .Mocap2Robot import Mocap2Robot
from .log_utils import print1


# 内旋转外旋（移动坐标系转固定坐标系）
def matrix3d_to_euler_angles_zyx(m3dr):
    beta_y = np.arctan2(m3dr[0, 2], np.sqrt(m3dr[0, 0] * m3dr[0, 0] + m3dr[0, 1] * m3dr[0, 1]))
    alpha_z = np.arctan2(-m3dr[0, 1] / np.cos(beta_y), m3dr[0, 0] / np.cos(beta_y))
    gamma_x = np.arctan2(-m3dr[1, 2] / np.cos(beta_y), m3dr[2, 2] / np.cos(beta_y))

    if np.abs(beta_y - np.pi / 2) < 10e-4:
        gamma_x = 0
        alpha_z = np.arctan2(m3dr[1, 0], m3dr[1, 1])

    if np.abs(beta_y + np.pi / 2) < 10e-4:
        gamma_x = 0
        alpha_z = np.arctan2(m3dr[1, 0], m3dr[1, 1])

    gamma_x = (gamma_x + np.pi) % (2 * np.pi) - np.pi
    beta_y = (beta_y + np.pi) % (2 * np.pi) - np.pi
    alpha_z = (alpha_z + np.pi) % (2 * np.pi) - np.pi

    return np.array([alpha_z, beta_y, gamma_x])


######################################################

def get_tf_by_rpy(r, p, y):
    tf = np.eye(4)
    # rt_l_hand_quest_2_l_hand_robot[:3,:3]=rpy2rotation_matrix(np.deg2rad(180),0,np.deg2rad(0))
    tf[:3, :3] = rpy2rotation_matrix(r, p, y)
    return tf



class UdpIkSender:
    def __init__(self):
        self.client_host = "192.168.1.196"
        self.client_port = 5005
        BUFFER_SIZE = 1024

        # 创建UDP套接字
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 绑定客户端地址和端口
        # client_socket.bind((client_host, client_port))

    def send(self, message):
        packed_data = b''.join([struct.pack('<f', num) for num in message])  # simulation >
        self.client_socket.sendto(packed_data, (self.client_host, self.client_port))
        # #print('send')

class Vuer2Qinlong(Mocap2Robot):
    def __init__(self):
        super().__init__()

        self.udp_ik_sender = UdpIkSender()

    def get_hand_tf(self,tf_base_robot_to_he_robot, tf_base_robot_to_l_ha_robot, tf_base_robot_to_r_ha_robot):


        # left
        tf_l_s_robot_to_l_s_sun = get_tf_by_rpy(np.deg2rad(-90), 0, np.deg2rad(90))

        tf_l_ha_robot_to_l_ha_sun = get_tf_by_rpy(np.deg2rad(180), np.deg2rad(-90), 0)

        # tf_he_robot_to_l_s_robot = np.array([[1.0000000, 0.0000000, 0.0000000, -0.2],
        #                                      [0.0000000, 1.0000000, 0.0000000, -0.2],
        #                                      [0.0000000, 0.0000000, 1.0000000, 0],
        #                                      [0, 0, 0, 1]])

        # right
        tf_r_s_robot_to_r_s_sun = get_tf_by_rpy(np.deg2rad(90), 0, np.deg2rad(90))

        tf_r_ha_robot_to_r_ha_sun = get_tf_by_rpy(np.deg2rad(0), np.deg2rad(-90), 0)

        # tf_he_robot_to_r_s_robot = np.array([[1.0000000, -0.0000000, 0.0000000, 0.2],
        #                                      [0.0000000, 1.0000000, -0.0000000, -0.2],
        #                                      [0.0000000, 0.0000000, 1.0000000, 0],
        #                                      [0, 0, 0, 1]])

        tf_l_s_sun_to_l_s_robot = fast_mat_inv(tf_l_s_robot_to_l_s_sun)
        tf_l_s_robot_to_he_robot = fast_mat_inv(self.tf_he_robot_to_l_s_robot)

        tf_r_s_sun_to_r_s_robot = fast_mat_inv(tf_r_s_robot_to_r_s_sun)
        tf_r_s_robot_to_he_robot = fast_mat_inv(self.tf_he_robot_to_r_s_robot)

        tf_he_robot_to_base_robot = fast_mat_inv(tf_base_robot_to_he_robot)

        tf_l_s_sun_to_l_ha_sun = tf_l_s_sun_to_l_s_robot @ tf_l_s_robot_to_he_robot @ tf_he_robot_to_base_robot @ tf_base_robot_to_l_ha_robot @ tf_l_ha_robot_to_l_ha_sun
        tf_r_s_sun_to_r_ha_sun = tf_r_s_sun_to_r_s_robot @ tf_r_s_robot_to_he_robot @ tf_he_robot_to_base_robot @ tf_base_robot_to_r_ha_robot @ tf_r_ha_robot_to_r_ha_sun

        zyx_left = matrix3d_to_euler_angles_zyx(tf_l_s_sun_to_l_ha_sun)
        zyx_right = matrix3d_to_euler_angles_zyx(tf_r_s_sun_to_r_ha_sun)

        return tf_l_s_sun_to_l_ha_sun, tf_r_s_sun_to_r_ha_sun, zyx_left, zyx_right

    def process(self, float_array):
        # print('begin processing!')

        head_data = np.array(float_array[:16])
        left_data = np.array(float_array[16:32])
        right_data = np.array(float_array[32:48])

        head_data = head_data.reshape((4, 4))
        left_data = left_data.reshape((4, 4))
        right_data = right_data.reshape((4, 4))

        # head_data[:3,:3]=np.eye(3)

        # left right positions

        head_data_t = head_data[:3, -1]
        left_data_t = left_data[:3, -1]
        right_data_t = right_data[:3, -1]
        # print(head_data)


        # head_data_rpy = euler_from_matrix(head_data[:3, :3])
        # left_data_rpy = euler_from_matrix(left_data[:3, :3])
        # right_data_rpy = euler_from_matrix(right_data[:3, :3])

        self.head_pose_new = head_data.copy()
        # head_position_save=group_to_head[:3,3]

        head_r_zyx = matrix3d_to_euler_angles_zyx(head_data[:3, :3])
        head_r_rpy_new = [0, head_r_zyx[1], 0]
        head_r_new = rpy2rotation_matrix(*head_r_rpy_new)
        self.head_pose_new[:3, :3] = head_r_new.copy()

        left_data_robot, right_data_robot, zyx_left_robot, zyx_right_robot = self.get_hand_tf(self.head_pose_save, left_data,
                                                                                         right_data)

        left_wrist_t = left_data_robot[:3, -1] * 1000
        right_wrist_t = right_data_robot[:3, -1] * 1000

        message = [
            #
            *zyx_left_robot,
            # -1.5708, 1.5708, 0,
            *left_wrist_t, 1.0,
            # -500.0, 300, 100.0, 0.5233,
            1.0,
            # right
            # 0.0, 1.5708, 0.0,
            # 100.0, -200, 500.0, 0.0,
            #  0.0True
            *zyx_right_robot,
            *right_wrist_t, -1.0,
            0.0,
            *float_array[198:]
        ]

        self.udp_ik_sender.send(message)
        print1(*left_wrist_t,*right_wrist_t,tag=1)

        # np.set_printoptions(formatter={'float': '{:.2f}'.format})
        # float_array=np.array(float_array)
        # print(float_array[198:198+12])

        left_wrist_t = left_data_robot[:3, -1] * 1000
        right_wrist_t = right_data_robot[:3, -1] * 1000
        # #print("left_theta_rad",self.left_theta_rad)
        # #print(left_data,left_wrist_t)

        tf_hand_left = left_data_robot
        tf_hand_right = right_data_robot
        joint_hand_left = float_array[198:210][[0, 2, 4, 6, 8, 10]]
        joint_hand_right = float_array[210:222][[0, 2, 4, 6, 8, 10]]
        print1(float_array[198:210])
        print1(joint_hand_left)
        return (tf_hand_left, tf_hand_right,
                joint_hand_left, joint_hand_right,
                head_data,left_data,right_data,
                zyx_left_robot,zyx_right_robot)
