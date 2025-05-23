import socket
import struct
import threading
from collections import deque

import numpy as np
# from tf.transformations import euler_matrix, quaternion_matrix

from scipy.spatial.transform import Rotation as R

from .Mocap2Robot import Mocap2Robot
from .log_utils import print1

import struct
from .HelpFuction import matrix3d_to_euler_angles_zyx, xyz_quaternion_to_homogeneous, rpy2rotation_matrix, \
    rotation_matrix_to_rpy, \
    find_axis_angle, calc_dist, rpy2rotation_matrix, fast_mat_inv



# 替代 euler_matrix
def euler_matrix(roll, pitch, yaw, axes='szyx'):
    """
    根据 roll, pitch, yaw 创建旋转矩阵，默认使用 'szyx' 顺序。
    """
    if axes == 'szyx':
        return R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    else:
        raise ValueError("Unsupported rotation order.")
    
# 替代 quaternion_matrix
def quaternion_matrix(quaternion):
    """
    从四元数生成旋转矩阵。
    """
    return R.from_quat(quaternion).as_matrix()



def lim_angle(angle):
    if angle < 0:
        angle = 0
    if angle > 90:
        angle = 90

    return int(angle)


class Quest3ControllerDataProcess:
    def __init__(self):
        self.ee_init_value_l = np.array([-1.5708, 1.5708, 0.0, -0.2, 0.450, 0.2, 0.5233])
        self.ee_init_value_r = np.array([1.5708, 1.5708, 0.0, -0.2, -0.450, 0.2, -0.5233])

        # self.ee_init_rt_l = np.eye(4)
        # self.ee_init_rt_l = euler_matrix(*self.ee_init_value_l[:3], 'szyx')
        # self.ee_init_rt_l[:3, -1] = self.ee_init_value_l[3:6]

        # self.ee_init_rt_r = np.eye(4)
        # self.ee_init_rt_r = euler_matrix(*self.ee_init_value_r[:3], 'szyx')
        # self.ee_init_rt_r[:3, -1] = self.ee_init_value_r[3:6]

        self.ee_init_rt_l = np.eye(4)
        self.ee_init_rt_l[:3, :3] = euler_matrix(*self.ee_init_value_l[:3], 'szyx')
        self.ee_init_rt_l[:3, -1] = self.ee_init_value_l[3:6]

        self.ee_init_rt_r = np.eye(4)
        self.ee_init_rt_r[:3, :3] = euler_matrix(*self.ee_init_value_r[:3], 'szyx')
        self.ee_init_rt_r[:3, -1] = self.ee_init_value_r[3:6]

        
        self.ee_cur_rt_l = self.ee_init_rt_l.copy()
        self.ee_cur_rt_r = self.ee_init_rt_r.copy()

        self.ee_last_rt_l = self.ee_init_rt_l.copy()
        self.ee_last_rt_r = self.ee_init_rt_r.copy()

    def handle_raw_data(self, data_bytes):
        '''
        if l == 4 * 48 * 7:
                       ...
                       fun(data_bytes)
        '''
        float_array = struct.unpack(f'{48 * 7}f', data_bytes)
        xyzqwqxqyqz = np.array(float_array).reshape((48, 7))
        return xyzqwqxqyqz

    def process(self, xyzqwqxqyqz):
        # global ee_cur_rt_l, ee_cur_rt_r, ee_last_rt_l, ee_last_rt_r

        # if save_once:
        #     np.save('/home/jyw/posedata.npy',xyzqwqxqyqz)
        #     save_once=False

        cmd = xyzqwqxqyqz[1, :].copy()
        cmd2 = xyzqwqxqyqz[2, :].copy()
        cmd3 = xyzqwqxqyqz[3, :].copy()

        # PrimaryIndexTrigger left grasp
        cmd[0]
        # SecondaryIndexTrigger right grasp
        cmd2[2]
        # B
        cmd[1]
        # Y
        cmd[6]
        #right hand stick 0,1left,2right,3up,4down,
        cmd2[0] 
        #left hand stick 0,1left,2right,3up,4down,
        cmd2[1] 

        # print(cmd)
        # 0 left grasp
        # 1 right grasp
        # 2 left move
        # 3 right move
        # 4 left reset
        # 5 right reset
        if cmd[2] == 0:
            self.ee_cur_rt_l = self.ee_last_rt_l.copy()
        if cmd[3] == 0:
            self.ee_cur_rt_r = self.ee_last_rt_r.copy()

        if cmd[4] == 1:
            self.ee_cur_rt_l = self.ee_init_rt_l.copy()
            self.ee_last_rt_l = self.ee_init_rt_l.copy()
        if cmd[5] == 1:
            self.ee_cur_rt_r = self.ee_init_rt_r.copy()
            self.ee_last_rt_r = self.ee_init_rt_r.copy()

        left_grasp = cmd[0] * 50
        right_grasp = cmd[1] * 50

        # unity left frame to right frame
        xyzqwqxqyqz[:, 3] *= -1

        xyzqwqxqyqz = xyzqwqxqyqz[:, [0, 2, 1, 3, 4, 6, 5]]

        xyz = xyzqwqxqyqz[:, :3]
        # qwqxqyqz=xyzqwqxqyqz[:,3:]
        qxqyqzqw = xyzqwqxqyqz[:, [4, 5, 6, 3]]

        rt_list = []
        cmd_list=[1,2,3]
        for i in range(48):
            if i in cmd_list:
                rt_list.append(np.eye(4))
            else:
                rt_base_quest_2_part_quest = np.eye(4)
                rt_base_quest_2_part_quest[:3, :3] = quaternion_matrix(qxqyqzqw[i, :])
                rt_base_quest_2_part_quest[:3, -1] = xyz[i, :]
                rt_list.append(rt_base_quest_2_part_quest)

        rt_l_hand_quest_2_l_hand_robot = np.eye(4)
        # rt_l_hand_quest_2_l_hand_robot[:3,:3]=rpy2rotation_matrix(np.deg2rad(180),0,np.deg2rad(0))
        rt_l_hand_quest_2_l_hand_robot[:3, :3] = rpy2rotation_matrix(0, np.deg2rad(-90), 0)

        rt_r_hand_quest_2_r_hand_robot = np.eye(4)
        # rt_r_hand_quest_2_r_hand_robot[:3,:3]=rpy2rotation_matrix(np.deg2rad(90),0,np.deg2rad(180))
        # rt_r_hand_quest_2_r_hand_robot[:3,:3]=rpy2rotation_matrix(np.deg2rad(0),np.deg2rad(180),0)
        rt_r_hand_quest_2_r_hand_robot[:3, :3] = rpy2rotation_matrix(np.deg2rad(180), np.deg2rad(-90), 0)

        if cmd[2] > 0:
            rt_l_ee1_quest_2_l_ee2_quest = rt_list[0]
            ee_new_rt_l = np.eye(4)
            delta_ee_l = fast_mat_inv(
                rt_l_hand_quest_2_l_hand_robot) @ rt_l_ee1_quest_2_l_ee2_quest @ rt_l_hand_quest_2_l_hand_robot

            # result correct but method not clear
            ee_new_rt_l[:3, -1] = self.ee_cur_rt_l[:3, -1] + delta_ee_l[:3, -1]
            ee_new_rt_l[:3, :3] = delta_ee_l[:3, :3] @ self.ee_cur_rt_l[:3, :3]

            # update last
            self.ee_last_rt_l = ee_new_rt_l.copy()

        # print(cmd[3])
        if cmd[3] > 0:
            rt_r_ee1_quest_2_r_ee2_quest = rt_list[24 + 0]
            # print(rt_r_ee1_quest_2_r_ee2_quest)
            ee_new_rt_r = np.eye(4)
            delta_ee_r = fast_mat_inv(
                rt_r_hand_quest_2_r_hand_robot) @ rt_r_ee1_quest_2_r_ee2_quest @ rt_r_hand_quest_2_r_hand_robot
            ee_new_rt_r[:3, -1] = self.ee_cur_rt_r[:3, -1] + delta_ee_r[:3, -1]
            ee_new_rt_r[:3, :3] = delta_ee_r[:3, :3] @ self.ee_cur_rt_r[:3, :3]
            # update last
            self.ee_last_rt_r = ee_new_rt_r.copy()

        # print(ee_new_rt_l[:3,-1].flatten(),delta_ee_l[:3,-1].flatten())
        # print(ee_new_rt_r[:3,-1].flatten(),delta_ee_r[:3,-1].flatten())

        # here we use last result

        zyx_left_robot = matrix3d_to_euler_angles_zyx(self.ee_last_rt_l)
        zyx_right_robot = matrix3d_to_euler_angles_zyx(self.ee_last_rt_r)

        left_wrist_t = self.ee_last_rt_l[:3, -1] * 1000
        right_wrist_t = self.ee_last_rt_r[:3, -1] * 1000
        # print("left_theta_rad",self.left_theta_rad)
        # print(left_data,left_wrist_t)
        # print([zyx_left_robot,
        #         left_wrist_t,
        #         zyx_right_robot,
        #         right_wrist_t,
        #         left_grasp,0,0,0,0,0,
        #         right_grasp,0,0,0,0,0,])

        msg1=[*zyx_left_robot,
              *left_wrist_t,
              1,
              *zyx_right_robot,
              *right_wrist_t,
              -1,1,1]
        
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        packed_data2=struct.pack('16f', 
                             *msg1)
        sock2.sendto(packed_data2, ("127.0.0.1", 5075))


        idxRightHandDelta = 24
        idxLeftHand=25
        idxRightHand=26
        idxHead=27

        head_pose=np.zeros(6)
        left_hand_pose=np.zeros(6)
        right_hand_pose=np.zeros(6)

        head_pose[3:]=rt_list[idxHead][:3,-1]
        left_hand_pose[3:]=rt_list[idxLeftHand][:3,-1]
        right_hand_pose[3:]=rt_list[idxRightHand][:3,-1]

        head_pose[:3]=matrix3d_to_euler_angles_zyx(rt_list[idxHead])
        left_hand_pose[:3]=matrix3d_to_euler_angles_zyx(rt_list[idxLeftHand])
        right_hand_pose[:3]=matrix3d_to_euler_angles_zyx(rt_list[idxRightHand])

        # print(xyzqwqxqyqz[idxHead,:])

        zyx_rad=(head_pose[:3])

        pan=  zyx_rad[0]
        tilt=zyx_rad[2]

  

        new_data_format={
            "hand_left_tran":left_wrist_t,
            "hand_left_rota":zyx_left_robot,
            "hand_rigt_tran":right_wrist_t,
            "hand_rigt_rota":zyx_right_robot,
            "head_yaw":pan, #rad
            "head_pith":tilt, #rad
            "frnt_trig_left": cmd[0], #0.0-1.0
            "frnt_trig_rigt":cmd2[2], #0.0-1.0
            "A":0, # reset right
            "B": cmd[1] ==1,
            "X":0,  # reset left
            "Y":cmd[6]==1,
            "stik_left":cmd2[1],
            "stik_rigt":cmd2[0] 
        }

        # return [zyx_left_robot,
        #         left_wrist_t,
        #         zyx_right_robot,
        #         right_wrist_t,
        #         (80, left_grasp, left_grasp, left_grasp, left_grasp, left_grasp),
        #         (80, right_grasp, right_grasp, right_grasp, right_grasp, right_grasp),
        #         head_pose,
        #         left_hand_pose,
        #         right_hand_pose ]
    
        return new_data_format


class Quest3Ctrller2Qinlong(Mocap2Robot):
    def __init__(self):
        super().__init__()
        # start one thread to receive data

        self.data_process = Quest3ControllerDataProcess()
        self.shared_resource_lock = threading.Lock()

        # Define a deque with a maximum size of 2
        self.data_queue = deque(maxlen=2)
        self.running = True  # 使用此标志来控制线程

        t1 = threading.Thread(target=self.receive_data)
        t1.start()

    def receive_data(self):
        # 配置服务器信息
        server_port = 5015
        BUFFER_SIZE = 8888
        # broadcast_ip = '255.255.255.255'
        broadcast_ip = '192.168.1.130'
        port = 4032# 5005
        message = b"Hello, devices!"
        max_retries = 5  # 设定一个最大重试次数
        retry_count = 0

        # 初始化并启动接收循环
        while self.running:
            # 创建新的套接字
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind(('', server_port))
            sock.settimeout(1.0)

            try:
                # 如果达到重试次数上限，重新发送广播消息
                if retry_count >= max_retries:
                    print("重新广播消息以重新连接...")
                    sock.sendto(message, (broadcast_ip, port))
                    retry_count = 0

                data_bytes, addr = sock.recvfrom(BUFFER_SIZE)
                if not data_bytes:
                    break

                l = len(data_bytes)
                if l == 4 * 48 * 7:
                    with self.shared_resource_lock:
                        self.data_queue.append(self.data_process.handle_raw_data(data_bytes))
                
                retry_count = 0  # 成功接收数据后重置重试计数
            except socket.timeout:
                print("no data")
                retry_count += 1
            except socket.error:
                print("socket.error")
                break
            finally:
                sock.close()


    def get_data(self):
        """
        retrive data
        :return:
        """
        data_out = None
        with self.shared_resource_lock:
            if self.data_queue:
                data_out = self.data_queue.popleft()  # Get the oldest item
                # print(f"Consumed: {item}, Queue: {list(data_queue)}")
            # else:
            #     print("Queue is empty, waiting for data...")
        # self.shared_resource_lock.acquire()
        # data_out= self.xyzqwqxqyqz.copy()
        # self.shared_resource_lock.release()
        return data_out

    def process(self, xyzqwqxqyqz):
        return self.data_process.process(xyzqwqxqyqz)

