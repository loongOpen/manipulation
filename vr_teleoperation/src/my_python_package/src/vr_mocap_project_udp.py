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

from mocap2robot_src.log_utils import print1
from vr_mocap_data_wrapper_udp_server import VRMocapDataWrapperUdpServer

# sys.path.append("..")

import asyncio
from enum import Enum
# from multiprocessing import shared_memory
# from queue import Queue

import cv2
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler, euler_from_matrix, quaternion_from_matrix

import signal
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from mocap2robot_src.Mocap2Robot_Vuer2Qinlong import Vuer2Qinlong

from mocap2robot.srv import MocapCalibrate, MocapCalibrateResponse
from mocap2robot.msg import VpControlData
from mocap2robot.msg import vp_control


def create_tf_mat(tf, parent, child):
    tf_msg = TransformStamped()

    # Set the frame IDs
    tf_msg.header.frame_id = parent  # Parent frame
    tf_msg.child_frame_id = child  # Child frame

    q = quaternion_from_matrix(tf)

    # Set the initial position (x, y, z)
    tf_msg.transform.translation.x = tf[0, -1]
    tf_msg.transform.translation.y = tf[1, -1]
    tf_msg.transform.translation.z = tf[2, -1]

    # Set the initial orientation (roll, pitch, yaw)

    tf_msg.transform.rotation.x = q[0]
    tf_msg.transform.rotation.y = q[1]
    tf_msg.transform.rotation.z = q[2]
    tf_msg.transform.rotation.w = q[3]

    tf_msg.header.stamp = rospy.Time.now()

    return tf_msg


class VRMocapManager:
    def __init__(self):
        rospy.init_node('VRMocapManager')

        # Create a TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Publish the TF message at a rate of 10 Hz
        self.tf_publish_rate = rospy.Rate(60)

        # start vuer website
        self.data_provider = VRMocapDataWrapperUdpServer()
        self.mocap2robot = Vuer2Qinlong()

        # set service
        self.set_service()

    def set_service(self):
        class CaliCmd(Enum):

            calibrate_shoulder = 1
            # 微调左右肩的偏移
            left_shoulder_x_up = 2
            left_shoulder_x_down = 3
            left_shoulder_y_up = 4
            left_shoulder_y_down = 5
            left_shoulder_z_up = 6
            left_shoulder_z_down = 7
            right_shoulder_x_up = 8
            right_shoulder_x_down = 9
            right_shoulder_y_up = 10
            right_shoulder_y_down = 11
            right_shoulder_z_up = 12
            right_shoulder_z_down = 13

        def handle_button_request(req):
            dx = 0.01
            rospy.loginfo(str(req.cmd) )
            if req.cmd == CaliCmd.calibrate_shoulder.value:
                self.mocap2robot.calibrate_shoulder()
                rospy.loginfo("Button pressed!")
                # 在这里添加你的按钮按下的处理逻辑
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.left_shoulder_x_up.value:
                self.mocap2robot.tf_he_robot_to_l_s_robot[0, -1] += dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")

            if req.cmd == CaliCmd.left_shoulder_x_down.value:
                self.mocap2robot.tf_he_robot_to_l_s_robot[0, -1] -= dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.left_shoulder_y_up.value:
                self.mocap2robot.tf_he_robot_to_l_s_robot[1, -1] += dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.left_shoulder_y_down.value:
                self.mocap2robot.tf_he_robot_to_l_s_robot[1, -1] -= dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.left_shoulder_z_up.value:
                self.mocap2robot.tf_he_robot_to_l_s_robot[2, -1] += dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.left_shoulder_z_down.value:
                self.mocap2robot.tf_he_robot_to_l_s_robot[2, -1] -= dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.right_shoulder_x_up.value:
                self.mocap2robot.tf_he_robot_to_r_s_robot[0, -1] += dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.right_shoulder_x_down.value:
                self.mocap2robot.tf_he_robot_to_r_s_robot[0, -1] -= dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.right_shoulder_y_up.value:
                self.mocap2robot.tf_he_robot_to_r_s_robot[1, -1] += dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.right_shoulder_y_down.value:
                self.mocap2robot.tf_he_robot_to_r_s_robot[1, -1] -= dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.right_shoulder_z_up.value:
                self.mocap2robot.tf_he_robot_to_r_s_robot[2, -1] += dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            if req.cmd == CaliCmd.right_shoulder_z_down.value:
                self.mocap2robot.tf_he_robot_to_r_s_robot[2, -1] -= dx
                return MocapCalibrateResponse(success=True, message="Button press handled successfully.")
            else:
                rospy.loginfo(str(req.cmd) + "Button not pressed.")
                return MocapCalibrateResponse(success=False, message="Button not pressed.")

        s = rospy.Service('/mocap/calibrate_shoulder_service', MocapCalibrate, handle_button_request)
        rospy.loginfo("Ready to handle /mocap/calibrate_shoulder_service requests.")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            self.data_provider.set_image(cv_image)

            # Display the image using OpenCV
            # cv2.imshow("Image window", cv_image)
            # cv2.waitKey(1)  # Refresh display at 1ms interval
        except CvBridgeError as e:
            print(e)
            rospy.logerr(e)

    def run(self):
        # cap = cv2.VideoCapture(0)

        right_hand_joint_state_pub = rospy.Publisher('/mocap/right_hand_joints', JointState, queue_size=10)
        left_hand_joint_state_pub = rospy.Publisher('/mocap/left_hand_joints', JointState, queue_size=10)

        # vp_control_pub = rospy.Publisher('/vp_control', VpControlData, queue_size=10)

        vp_control_pub = rospy.Publisher('/vp_control', vp_control, queue_size=10)

        # subscribe image
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        self.data_provider.run()

        while not rospy.is_shutdown():
            # set image
            # ret, frame = cap.read()
            # image = cv2.resize(frame, (img_shape[1], img_shape[0]))
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # cv2.imshow("canvas",frame)
            # cv2.waitKey(30)

            # np.copyto(self.img_array, image)

            # prepare message

            data = self.data_provider.get_data()

            (tf_hand_left, tf_hand_right, joint_hand_left, joint_hand_right,
             head_data, left_data, right_data,
             zyx_left_robot, zyx_right_robot) = self.mocap2robot.process(data)

            # publish message
            tf_msg = create_tf_mat(tf_hand_left, 'mocap2robot/l_shoulder', 'mocap2robot/l_wrist')
            self.tf_broadcaster.sendTransform(tf_msg)
            tf_msg = create_tf_mat(tf_hand_right, 'mocap2robot/r_shoulder', 'mocap2robot/r_wrist')
            self.tf_broadcaster.sendTransform(tf_msg)

            tf_msg = create_tf_mat(self.mocap2robot.tf_he_robot_to_l_s_robot, 'mocap/head', 'mocap/l_shoulder')
            self.tf_broadcaster.sendTransform(tf_msg)
            tf_msg = create_tf_mat(self.mocap2robot.tf_he_robot_to_r_s_robot, 'mocap/head', 'mocap/r_shoulder')
            self.tf_broadcaster.sendTransform(tf_msg)

            tf_msg = create_tf_mat(head_data, 'map',
                                   'head')

            # Publish the TF message
            self.tf_broadcaster.sendTransform(tf_msg)

            # tf_msg = create_tf_xyz_quat(float_array[1]/1000,float_array[2]/1000,float_array[3]/1000,float_array[4],float_array[5],float_array[6],float_array[7],'map',link)
            tf_msg = create_tf_mat(left_data, 'map',
                                   'left_wrist')

            # Publish the TF message
            self.tf_broadcaster.sendTransform(tf_msg)

            # tf_msg = create_tf_xyz_quat(float_array[1]/1000,float_array[2]/1000,float_array[3]/1000,float_array[4],float_array[5],float_array[6],float_array[7],'map',link)
            tf_msg = create_tf_mat(right_data, 'map',
                                   'right_wrist')

            # Publish the TF message
            self.tf_broadcaster.sendTransform(tf_msg)

            right_hand_joint_state = JointState()
            right_hand_joint_state.header = Header()
            right_hand_joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
            right_hand_joint_state.position = np.zeros(6)
            right_hand_joint_state.velocity = np.zeros(6)
            right_hand_joint_state.effort = np.zeros(6)
            right_hand_joint_state.header.stamp = rospy.Time.now()

            left_hand_joint_state = JointState()
            left_hand_joint_state.header = Header()
            left_hand_joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
            left_hand_joint_state.position = np.zeros(7)
            left_hand_joint_state.velocity = np.zeros(7)
            left_hand_joint_state.effort = np.zeros(7)
            left_hand_joint_state.header.stamp = rospy.Time.now()

            # Update joint positions
            for i in range(6):
                left_hand_joint_state.position[i] = joint_hand_left[i]
                right_hand_joint_state.position[i] = joint_hand_right[i]

            # rospy.loginfo(joint_state)
            left_hand_joint_state_pub.publish(left_hand_joint_state)
            right_hand_joint_state_pub.publish(right_hand_joint_state)

            left_wrist_t = tf_hand_left[:3, -1] * 1000
            right_wrist_t = tf_hand_right[:3, -1] * 1000
            msg = vp_control()
            msg.arx_pos_left = [*left_wrist_t, zyx_left_robot[2], zyx_left_robot[1], zyx_left_robot[0]]
            msg.cmd_left = 1
            msg.arx_pos_right = [*right_wrist_t, zyx_right_robot[2], zyx_right_robot[1], zyx_right_robot[0]]
            msg.cmd_right = 1
            msg.vcap = [np.clip(joint_hand_left[2], 0, np.pi / 2) / 90 * 85,
                        np.clip(joint_hand_right[2], 0, np.pi / 2) / 90 * 85]
            msg.hand_q_left=[*joint_hand_left]
            msg.hand_q_right=[*joint_hand_right]

            vp_control_pub.publish(msg)

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


    def on_release(key):
        '松开按键时执行。'
        print1('{0} released'.format(
            key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False
        if key.char == 'c':
            # Stop listener
            vrMocapManager.mocap2robot.calibrate_shoulder()
            print1('c', tag=1)


    # listener = keyboard.Listener(
    #     on_release=on_release)
    # listener.start()

    vrMocapManager.run()
