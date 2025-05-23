#!/usr/bin/env python
'''
accept vuer vr support
adapt devices: quest3 and vision pro
drive QinLong robot

TODO: finger
TODO:
'''

import os
import time
# printing environment variables
# print(os.environ)
import sys

from mocap2robot_src.log_utils import print1

# sys.path.append("..")

from enum import Enum
# from multiprocessing import shared_memory
# from queue import Queue
from scipy.spatial.transform import Rotation as R

import numpy as np

from mocap2robot_src.Mocap2Robot_quest3Controller2Qinlong import Quest3Ctrller2Qinlong

import socket
import struct


global waist_exp
waist_exp = 0
def eulerZYXToRotationMatrix_nei(alpha, beta, gamma):

    rotation_matrix = (
        R.from_euler('z', gamma).as_matrix() @
        R.from_euler('y', beta).as_matrix() @
        R.from_euler('x', alpha).as_matrix()
    )
    return rotation_matrix

def eulerZYXToRotationMatrix_wai(alpha, beta, gamma):

    rotation_matrix = (
        R.from_euler('x', alpha).as_matrix() @
        R.from_euler('y', beta).as_matrix() @
        R.from_euler('z', gamma).as_matrix()
    )
    return rotation_matrix

def eulerZYXToHomogeneousMatrix_nei(alpha, beta, gamma):
    """
    创建齐次矩阵，其中旋转部分按照内部顺序生成。
    """
    rotation_matrix = eulerZYXToRotationMatrix_nei(alpha, beta, gamma)
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    return homogeneous_matrix

def RotationToPose(transform):

    # 提取 ZYX 欧拉角 (yaw, pitch, roll)
    rotation = R.from_matrix(transform)
    roll, pitch, yaw = rotation.as_euler('zyx', degrees=False)

    return [yaw, pitch, roll]

def homogeneousToPose(transform):
    """
    将 4x4 齐次变换矩阵转换为位置和 ZYX 欧拉角。
    返回格式：[x, y, z, roll, pitch, yaw]
    """
    # 提取平移
    x, y, z = transform[:3, 3]

    # 提取旋转矩阵
    rotation_matrix = transform[:3, :3]

    # 提取 ZYX 欧拉角 (yaw, pitch, roll)
    rotation = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = rotation.as_euler('zyx', degrees=False)

    return [x, y, z, roll, pitch, yaw]

def inverseMatrix(matrix):
    """
    求给定矩阵的逆。支持 3x3 旋转矩阵和 4x4 齐次矩阵。
    """
    return np.linalg.inv(matrix)


def update_waist_angle(stik_rigt, current_waist_angle, step=1.0, min_angle=-60, max_angle=60):
    """
    根据摇杆输入调整腰部角度，累加方式：
    - stik_rigt == 1 时减少腰部角度
    - stik_rigt == 2 时增加腰部角度
    W
    :param stik_rigt: 右摇杆的输入，1（左转）或 2（右转）
    :param current_waist_angle: 当前腰部角度
    :param step: 每次调整的角度步长（默认 1°）
    :param min_angle: 允许的最小角度（默认 -30°）
    :param max_angle: 允许的最大角度（默认 30°）
    :return: 更新后的腰部角度
    """
    if stik_rigt == 1:
        current_waist_angle = max(min_angle, current_waist_angle - step)  # 角度减少
    elif stik_rigt == 2:
        current_waist_angle = min(max_angle, current_waist_angle + step)  # 角度增加
    
    return current_waist_angle


# 定义 remote_msg_end 结构体的打包格式并使用 UDP 发送
def send_udp_message(zyx_left_robot, left_wrist_t, zyx_right_robot, right_wrist_t, joint_hand_left, joint_hand_right, head_pose,waist_stik_rigt, udp_ip, udp_port):
    # 生成 remote_msg_end 结构体的数据
    global waist_exp
    base_vel = [0.0, 0.0, 0.0]  # 根据需要设置基础速度
    arm_pos_exp_l = [left_wrist_t[0], left_wrist_t[1], left_wrist_t[2]]
    arm_att_exp_l = [zyx_left_robot[2], zyx_left_robot[1], zyx_left_robot[0]]
    cap_l = joint_hand_left[2]
    arm_pos_exp_r = [right_wrist_t[0], right_wrist_t[1], right_wrist_t[2]]
    arm_att_exp_r = [zyx_right_robot[2], zyx_right_robot[1], zyx_right_robot[0]]
    cap_r = joint_hand_right[2]
    #arm_pos_exp_l = [0.1,0.1,0.1]
    #arm_att_exp_l = [0.1,0.1,0.1]
    #cap_l = 0.1
    #arm_pos_exp_r = [0.1,0.1,0.1]
    #arm_att_exp_r = [0.1,0.1,0.1]
    #cap_r = 0.1
    # print(arm_pos_exp_l,arm_att_exp_l)

    right_R_S_old_to_new = eulerZYXToRotationMatrix_nei(-1.5708, 0, -1.5708)
    right_R_H_old_to_new = eulerZYXToRotationMatrix_nei(0, 0, 3.1415)
    left_R_S_old_to_new = eulerZYXToRotationMatrix_nei(1.5708, 0, 1.5708)
    left_R_H_old_to_new = eulerZYXToRotationMatrix_nei(0, 0, 3.1415)

    right_R_S_to_H = eulerZYXToRotationMatrix_wai(*arm_att_exp_r)
    left_R_S_to_H = eulerZYXToRotationMatrix_wai(*arm_att_exp_l)

    right_R_S_to_H_new = inverseMatrix(right_R_S_old_to_new) @ right_R_S_to_H @ right_R_H_old_to_new
    left_R_S_to_H_new = inverseMatrix(left_R_S_old_to_new) @ left_R_S_to_H @ left_R_H_old_to_new

    euler_left = RotationToPose(left_R_S_to_H_new)
    euler_right = RotationToPose(right_R_S_to_H_new)

    euler_left_deg = np.rad2deg([*euler_left])
    euler_right_deg = np.rad2deg([*euler_right])

    euler_left_deg[2] = -euler_left_deg[2]

    # 生成 `remote_msg_end` 结构体的数据
    head = 0xBB  # 设定 head 的默认值
    mode = 1  # 假设使用 epos 模式，你可以根据需要修改

    base_vel = [0.0, 0.0, 0.0]  # 基础速度
    arm_q_exp_l = [0]*7
    arm_q_exp_r = [0]*7
    waist_exp = update_waist_angle(waist_stik_rigt,waist_exp)
    head_exp = [0]*3
    head_exp[0] = -head_pose[0]
    head_exp[1] = head_pose[1]
    flt_rate = 1.0
    # print(len(head_exp))
    print("cap_l	",cap_l)
    print("cap_r	",cap_r)
    arm_pos_exp_left = [arm_pos_exp_r[0],arm_pos_exp_l[1],arm_pos_exp_l[2]]
    arm_pos_exp_right = [arm_pos_exp_l[0],arm_pos_exp_r[1],arm_pos_exp_r[2]]
    # 打包数据为二进制格式，按结构体的顺序: 3 (base_vel) + 3 (arm_pos_exp_l) + 3 (arm_att_exp_l) + 1 (cap_l) + 3 (arm_pos_exp_r) + 3 (arm_att_exp_r) + 1 (cap_r)
    packed_data = struct.pack('<cc3f3f3f7f1f3f3f7f1f1f3f1f',  # 结构体打包格式
                            bytes([head]),  # char 类型的 head
                            bytes([mode]),  # char 类型的 mode
                            *base_vel,  
                            *arm_pos_exp_left,  
                            *euler_left_deg,  
                            *arm_q_exp_l,  # 左臂的 7 个关节角度
                            cap_l,  
                            *arm_pos_exp_right,  
                            *euler_right_deg,  
                            *arm_q_exp_r,  # 右臂的 7 个关节角度
                            cap_r,  
                            waist_exp,  
                            *head_exp,  
                            flt_rate  
                            )

    # 创建 UDP 套接字并发送数据
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(packed_data, (udp_ip, udp_port))
    # print("head_exp	",head_exp)
    # print("waist_exp	",waist_exp)
    # print("left_xyz	",arm_pos_exp_l)
    # print("left_rpy	",euler_left_deg)
    # print("right_xyz",arm_pos_exp_r)
    # print("right_rpy",euler_right_deg)

class VRMocapManager:
    def __init__(self):


        # start vuer website
        self.data_provider = Quest3Ctrller2Qinlong()
        self.mocap2robot = self.data_provider


        # 设置 UDP 发送的目标 IP 和端口
        self.udp_ip = "192.168.1.188"  # 本地测试 IP
        self.udp_port = 3334        # 目标端口
        self.running = True
        self.switch_camera=1


    def run(self):

        last_b=False


        while self.running:

            data = self.data_provider.get_data()
            if data is not None:
                processed_data = self.mocap2robot.process(data)  # 现在返回的是 `new_data_format` 字典

                # 提取 `new_data_format` 中的值
                zyx_left_robot = processed_data["hand_left_rota"]
                left_wrist_t = processed_data["hand_left_tran"]
                zyx_right_robot = processed_data["hand_rigt_rota"]
                right_wrist_t = processed_data["hand_rigt_tran"]
                head_yaw = processed_data["head_yaw"]
                head_pitch = processed_data["head_pith"]
                # 解析手指抓取程度
                joint_hand_left = [0, 0, processed_data["frnt_trig_left"], 0, 0, 0]
                joint_hand_right = [0, 0, processed_data["frnt_trig_rigt"], 0, 0, 0]
                head_yaw = max(-0.6, head_yaw*1.5)
                head_yaw = min(0.6, head_yaw*1.5)
                head_pitch = max(-0.6, head_pitch*1.5)
                head_pitch = min(0.6, head_pitch*1.5)
                # 解析头部和手的位姿
                head_pose = [head_yaw*180.0/3.1415, head_pitch*180.0/3.1415, 0, 0, 0, 0]
                left_hand_pose = [*zyx_left_robot, *left_wrist_t]
                right_hand_pose = [*zyx_right_robot, *right_wrist_t]
                waist_stik_rigt = processed_data["stik_rigt"]

                if last_b==False and  processed_data["B"]==True:
                    if self.switch_camera==1:
                        self.switch_camera=2
                    elif self.switch_camera==2:
                        self.switch_camera=1
                
                last_b=processed_data["B"]
                print(processed_data["B"], self.switch_camera)


                # zyx_left_robot, left_wrist_t, zyx_right_robot, right_wrist_t, joint_hand_left, joint_hand_right,head_pose,left_hand_pose,right_hand_pose  = self.mocap2robot.process(data)

                # send_udp_message(zyx_left_robot, left_wrist_t, zyx_right_robot, right_wrist_t, joint_hand_left, joint_hand_right, self.udp_ip, self.udp_port)
                
                # 发送数据
                send_udp_message(
                    zyx_left_robot, left_wrist_t,
                    zyx_right_robot, right_wrist_t,
                    joint_hand_left, joint_hand_right,
                    head_pose,waist_stik_rigt,
                    self.udp_ip, self.udp_port
                )

            time.sleep(1/60)

    def stop(self):
        self.running = False


if __name__ == '__main__':
    print("VRMocapManager start!")
    vrMocapManager = VRMocapManager()
    try:
        vrMocapManager.run()
    except KeyboardInterrupt:
        vrMocapManager.stop()
        print("VRMocapManager stopped!")
