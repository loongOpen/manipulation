#!/usr/bin/env python
import threading
import rospy
from cv_bridge import CvBridgeError, CvBridge
from sensor_msgs.msg import Image
import mocap2robot_src
from mocap2robot_src.webcam.webcam import main_simple
from mocap2robot_src.common.config import get_config
from vr_udp_mini import VRMocapManager

bridge = CvBridge()


vrMocapManager = VRMocapManager()



def image_callback(msg):
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        if vrMocapManager.switch_camera ==1:  
            mocap2robot_src.webcam.webcam.frame_buffer = cv_image
        # print("ros cb",frame_buffer.shape)

        # self.data_provider.set_image(cv_image)

        # Display the image using OpenCV
        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(1)  # Refresh display at 1ms interval
    except CvBridgeError as e:
        print(e)
        rospy.logerr(e)

def image_callback2(msg):
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        if vrMocapManager.switch_camera ==2:
            mocap2robot_src.webcam.webcam.frame_buffer = cv_image
        # print("ros cb",frame_buffer.shape)

        # self.data_provider.set_image(cv_image)

        # Display the image using OpenCV
        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(1)  # Refresh display at 1ms interval
    except CvBridgeError as e:
        print(e)
        rospy.logerr(e)



def mocap_thread_f():
    print("VRMocapManager start!")
    
    try:
        vrMocapManager.run()
    except KeyboardInterrupt:
        vrMocapManager.stop()
        print("VRMocapManager stopped!")


if __name__ == "__main__":
    rospy.init_node('mocap_manager_node')
    image_sub = rospy.Subscriber(get_config("ros")["cam_topic"], Image, image_callback)
    image_sub2 = rospy.Subscriber("/cam_high/color/image_raw", Image, image_callback2)

    mocap_thead=threading.Thread(target=mocap_thread_f)
    mocap_thead.start()

    

    main_simple()
