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

from mocap2robot_src.Mocap2Robot_quest3Controller2Qinlong import Quest3Ctrller2Qinlong

from mocap2robot.srv import MocapCalibrate, MocapCalibrateResponse
from mocap2robot.msg import VpControlData
from mocap2robot.msg import vp_control
from mocap2robot.msg import VrPose

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
        rospy.init_node('mocap_manager_node')

        # Create a TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Publish the TF message at a rate of 10 Hz
        self.tf_publish_rate = rospy.Rate(60)

        # start vuer website
        self.data_provider = Quest3Ctrller2Qinlong()
        self.mocap2robot = self.data_provider

    def run(self):


        # vp_control_pub = rospy.Publisher('/vp_control', VpControlData, queue_size=10)

        vp_control_pub = rospy.Publisher('/vp_control', vp_control, queue_size=10)
        vr_pose_pub = rospy.Publisher('/VrPose', VrPose, queue_size=10)

        print("start main loop")

        while not rospy.is_shutdown():

            data = self.data_provider.get_data()
            if data is not None:
                zyx_left_robot, left_wrist_t, zyx_right_robot, right_wrist_t, joint_hand_left, joint_hand_right,head_pose,left_hand_pose,right_hand_pose  = self.mocap2robot.process(data)


                msg = vp_control()
                msg.arx_pos_left = [*left_wrist_t, zyx_left_robot[2], zyx_left_robot[1], zyx_left_robot[0]]
                msg.cmd_left = 1
                msg.arx_pos_right = [*right_wrist_t, zyx_right_robot[2], zyx_right_robot[1], zyx_right_robot[0]]
                msg.cmd_right = 1
                msg.vcap = [joint_hand_left[2], joint_hand_right[2]]
                msg.hand_q_left = [*joint_hand_left]
                msg.hand_q_right = [*joint_hand_right]

                vp_control_pub.publish(msg)

                vr_pose_msg=VrPose()
                vr_pose_msg.pos_head=[*head_pose]
                vr_pose_msg.pos_left_hand=[*left_hand_pose]
                vr_pose_msg.pos_right_hand=[*right_hand_pose]
                
                vr_pose_pub.publish(vr_pose_msg)

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

    # listener = keyboard.Listener(
    #     on_release=on_release)
    # listener.start()

    vrMocapManager.run()
