#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from pynput import keyboard
import threading
# sys.path.append("..")
import socket
import struct
import asyncio
from enum import Enum
# from multiprocessing import shared_memory
# from queue import Queue
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import rospy
import tf2_ros
from tf.transformations import quaternion_from_euler, euler_from_matrix, quaternion_from_matrix

import signal
from cv_bridge import CvBridge, CvBridgeError

from mocap2robot_src.Mocap2Robot_quest3Controller2Qinlong import Quest3Ctrller2Qinlong


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

def update_waist_angle(stik_rigt, current_waist_angle, step=2.5, min_angle=-90, max_angle=90):
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


gripper_state_left = 0
gripper_state_right = 0
prev_joint_hand_left = 0
prev_joint_hand_right = 0


def update_gripper_state(joint_hand_left, joint_hand_right):
    global gripper_state_left, gripper_state_right, prev_joint_hand_left, prev_joint_hand_right

    # 左夹爪状态更新
    if prev_joint_hand_left > 45 and joint_hand_left <= 45:
        # 切换状态：0 -> 50 或 50 -> 0
        gripper_state_left = 40.0 if gripper_state_left == 0 else 0
    # 更新上一次的值
    prev_joint_hand_left = joint_hand_left

    # 右夹爪状态更新
    if prev_joint_hand_right > 45 and joint_hand_right <= 45:
        # 切换状态：0 -> 50 或 50 -> 0
        gripper_state_right = 40.0 if gripper_state_right == 0 else 0
    # 更新上一次的值
    prev_joint_hand_right = joint_hand_right




class VRMocapManager:
    def __init__(self):
        # start vuer website
        self.data_provider = Quest3Ctrller2Qinlong()
        self.mocap2robot = self.data_provider

        self.log_file_path_base = "vrmocap_data_log"  # 文件基础名称
        self.log_file_counter = 1  # 文件后缀计数器
        self.is_logging = False
        self.log_file_path = None  # 当前日志文件路径

        # 设置 UDP 发送的目标 IP 和端口
        self.udp_ip = "192.168.1.188"  # 本地测试 IP
        self.udp_port = 3334        # 目标端口
        self.running = True
        self.switch_camera=1

        # 当前文件路径：.../my_python_package/src/
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 到 config 目录的路径：../config/
        config_dir = os.path.abspath(os.path.join(current_dir, "..", "config"))

        # 创建目录（可选：如果不存在就自动创建）
        os.makedirs(config_dir, exist_ok=True)

        self.config_dir = config_dir


    # 定义 remote_msg_end 结构体的打包格式并使用 UDP 发送
    def send_udp_message(self,zyx_left_robot, left_wrist_t, zyx_right_robot, right_wrist_t, joint_hand_left, joint_hand_right, head_pose,waist_stik_rigt, udp_ip, udp_port):
        # 生成 remote_msg_end 结构体的数据
        # 生成 remote_msg_end 结构体的数据
        global waist_exp
        base_vel = [0.0, 0.0, 0.0]  # 根据需要设置基础速度
        arm_pos_exp_l = [left_wrist_t[0], left_wrist_t[1], left_wrist_t[2]]
        arm_att_exp_l = [zyx_left_robot[2], zyx_left_robot[1], zyx_left_robot[0]]
        cap_l = joint_hand_left[2]
        arm_pos_exp_r = [right_wrist_t[0], right_wrist_t[1], right_wrist_t[2]]
        arm_att_exp_r = [zyx_right_robot[2], zyx_right_robot[1], zyx_right_robot[0]]
        cap_r = joint_hand_right[2]

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
        arm_pos_exp_left = [arm_pos_exp_r[0],arm_pos_exp_l[1],arm_pos_exp_l[2]]
        arm_pos_exp_right = [arm_pos_exp_l[0],arm_pos_exp_r[1],arm_pos_exp_r[2]]
        arm_att_exp_left = [euler_left_deg[0]*3.1415/180.0,euler_left_deg[1]*3.1415/180.0,euler_left_deg[2]*3.1415/180.0]
        arm_att_exp_right = [euler_right_deg[0]*3.1415/180.0,euler_right_deg[1]*3.1415/180.0,euler_right_deg[2]*3.1415/180.0]
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
        print("left_rpy	",euler_left_deg)
        print("right_rpy",euler_right_deg)

        self.log_to_file(arm_att_exp_left, arm_pos_exp_l, arm_att_exp_right, arm_pos_exp_r, cap_l, cap_r, waist_exp, head_exp)



    def start_new_log_file(self):
        """Start a new log file with an incremented suffix."""
        filename = f"{self.log_file_path_base}_{self.log_file_counter}.txt"
        self.log_file_path = os.path.join(self.config_dir, filename)
        self.log_file_counter += 1
        with open(self.log_file_path, 'w') as f:
            pass  # 创建文件并清空内容
        rospy.loginfo(f"Started new log file: {self.log_file_path}")
        self.is_logging = True

    def stop_logging(self):
        """Stop logging."""
        if self.is_logging:
            rospy.loginfo(f"Stopped logging to file: {self.log_file_path}")
        self.is_logging = False

    def log_to_file(self, zyx_left_robot, left_wrist_t, zyx_right_robot, right_wrist_t, gripper_left, gripper_right, waist_exp, head_exp):
        """Log the data to a text file."""
        if self.is_logging:
            with open(self.log_file_path, 'a') as f:
                log_line = (
                    f"{left_wrist_t[0] / 1000},"
                    f"{left_wrist_t[1] / 1000},"
                    f"{left_wrist_t[2] / 1000}," 
                    f"{zyx_left_robot[2]}," 
                    f"{zyx_left_robot[1]}," 
                    f"{zyx_left_robot[0]}," 
                    f"{gripper_left}," 
                    f"{right_wrist_t[0] / 1000},"
                    f"{right_wrist_t[1] / 1000},"
                    f"{right_wrist_t[2] / 1000}," 
                    f"{zyx_right_robot[2]}," 
                    f"{zyx_right_robot[1]}," 
                    f"{zyx_right_robot[0]}," 
                    f"{gripper_right}," 
                    f"{0.05},"
                    f"{waist_exp}," 
                    f"{head_exp[0]}," 
                    f"{head_exp[1]}\n"
                    # f"{0.02}\n" 
                )
                f.write(log_line)

    def toggle_logging(self, start):
        """Toggle logging state."""
        self.is_logging = start
        state = "started" if start else "stopped"
        # rospy.loginfo(f"Logging {state}.")

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
                # 发送数据
                self.send_udp_message(
                    zyx_left_robot, left_wrist_t,
                    zyx_right_robot, right_wrist_t,
                    joint_hand_left, joint_hand_right,
                    head_pose,waist_stik_rigt,
                    self.udp_ip, self.udp_port
                )

            time.sleep(1/60)
    def stop(self):
        self.running = False


def handle_keyboard(manager):
    """Listen for keyboard events to control logging."""
    print("Keyboard listener started.")  # ✅ 加这一行
    def on_press(key):
        try:
            if key.char == 'o':  # Start a new log file
                manager.start_new_log_file()
            elif key.char == 'p':  # Stop logging
                manager.stop_logging()
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == '__main__':
    print("VRMocapManager start!")
    vrMocapManager = VRMocapManager()
    threading.Thread(target=handle_keyboard, args=(vrMocapManager,), daemon=True).start()
    try:
        vrMocapManager.run()
    except KeyboardInterrupt:
        vrMocapManager.stop()
        print("VRMocapManager stopped!")
