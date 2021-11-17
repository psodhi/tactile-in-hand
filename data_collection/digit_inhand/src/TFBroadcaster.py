# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python

import rospy
import tf

from geometry_msgs.msg import PoseStamped

class TFBroadcaster:

    def __init__(self):

        self.obj_shape = rospy.get_param("obj_shape")

        # broadcast tf frames for digit mocap pose messages
        self.digit_sub = rospy.Subscriber(
            "/vrpn_client_node/Digit/pose", PoseStamped, self.handler_digit_tf)

        # publish new messages for digit centers
        self.digit_center_pub = rospy.Publisher(
            "/vrpn_client_node/Digit/center/pose", PoseStamped, queue_size=1)
        self.digit_pose_sub = rospy.Subscriber(
            "/vrpn_client_node/Digit/pose", PoseStamped, self.callback_digit_pose)

        # self.object_center_pub = rospy.Publisher(
        #     "/vrpn_client_node/{0}/center/pose".format(self.obj_shape), PoseStamped, queue_size=1)

        # broadcast tf frames for the new digit center messages
        self.digit_center_sub = rospy.Subscriber(
            "/vrpn_client_node/Digit/center/pose", PoseStamped, self.handler_digit_center_tf)

    def handler_digit_tf(self, msg):
        br = tf.TransformBroadcaster()
        br.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                         (msg.pose.orientation.x, msg.pose.orientation.y,
                          msg.pose.orientation.z, msg.pose.orientation.w),
                         msg.header.stamp, "/digit/", msg.header.frame_id)

    def callback_digit_pose(self, msg):
        digit_center_pose = PoseStamped()

        digit_center_pose.header = msg.header
        digit_center_pose.header.frame_id = "/digit/"

        (offset_x, offset_y, offset_z) = (0.03937, 0.014, -0.007)

        digit_center_pose.pose.position.x = offset_x
        digit_center_pose.pose.position.y = offset_y
        digit_center_pose.pose.position.z = offset_z

        digit_center_pose.pose.orientation.x = 0
        digit_center_pose.pose.orientation.y = 0
        digit_center_pose.pose.orientation.z = 0
        digit_center_pose.pose.orientation.w = 1

        self.digit_center_pub.publish(digit_center_pose)

    def handler_digit_center_tf(self, msg):
        br = tf.TransformBroadcaster()
        br.sendTransform((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                         (msg.pose.orientation.x, msg.pose.orientation.y,
                          msg.pose.orientation.z, msg.pose.orientation.w),
                         msg.header.stamp, "/digit/center/", msg.header.frame_id)

def main():
    rospy.init_node("tf_broadcaster")
    rospy.loginfo("Initialized tf_broadcaster node.")

    tf_broadcaster = TFBroadcaster()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
