#!/usr/bin/env python
import rospy
import tf

import rospy
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32MultiArray


def callback(transform_stamped):
    rate = rospy.Rate(10)  # Process at 10 Hz
    # Log the received message
    # rospy.loginfo("Received TransformStamped message")
    #
    # # Print the header (time and frame information)
    # rospy.loginfo("Timestamp: {}".format(transform_stamped.header.stamp))
    # rospy.loginfo("Parent frame: {}".format(transform_stamped.header.frame_id))
    # rospy.loginfo("Child frame: {}".format(transform_stamped.child_frame_id))
    #
    # # Print the translation
    # rospy.loginfo("Translation: x = {}, y = {}, z = {}".format(
    #     transform_stamped.transform.translation.x,
    #     transform_stamped.transform.translation.y,
    #     transform_stamped.transform.translation.z))
    #
    # # Print the rotation (quaternion)
    # rospy.loginfo("Rotation: x = {}, y = {}, z = {}, w = {}".format(
    #     transform_stamped.transform.rotation.x,
    #     transform_stamped.transform.rotation.y,
    #     transform_stamped.transform.rotation.z,
    #     transform_stamped.transform.rotation.w))
    rate.sleep()

def listener():
    # Initialize the ROS node
    rospy.init_node('tf_subscriber_node', anonymous=True)

    # Subscribe to the 'custom_transform_topic' topic
    rospy.Subscriber('custom_transform_topic', TransformStamped, callback)

    # Keep the node running until it's stopped
    rospy.spin()

def tf_callback():
    # Create a listener object
    listener = tf.TransformListener()
    rate = rospy.Rate(30.0)

    while not rospy.is_shutdown():





        try:
            # Wait for the transformation from 'map' to '1' to become available
            # listener.waitForTransform('map', '1', rospy.Time())

            # Get the transformation
            (trans, rot) = listener.lookupTransform('map', '1', rospy.Time(0))

            # Print the translation and rotation
            rospy.loginfo("Translation: {}".format(trans))
            rospy.loginfo("Rotation: {}".format(rot))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("Transform error: {}".format(e))

        rate.sleep()

if __name__ == '__main__':
    # rospy.init_node('tf_subscriber_node')
    #
    # # Call the tf_callback function
    # tf_callback()
    #
    # # Keep the node alive
    # rospy.spin()

    listener()
