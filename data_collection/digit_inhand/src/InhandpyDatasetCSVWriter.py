# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python

import rospy
import rospkg

import numpy as np
import pandas as pd

import io
import os
import tf
import json


import cv2
import imageio

from cv_bridge import CvBridge

from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge, CvBridgeError
from jsk_rviz_plugins.msg import OverlayText

BASE_PATH = ''

class InhandpyDatasetCSVWriter:
    def __init__(self):

        self.obj_shape = rospy.get_param("obj_shape")
        
        # ros vars
        self.tf_listener = tf.TransformListener()
        self.bridge = CvBridge()
        self.digit_image_sub = rospy.Subscriber(
            "/digit/digit_alpha/image_raw/", Image, self.callback_digit_image, queue_size=1)

        # overhead image editor
        self.bridge = CvBridge()
        self.overhead_image_sub = rospy.Subscriber(
            "/rgb/image_raw", Image, self.callback_overhead_image, queue_size=1)
        self.overhead_image_crop_pub = rospy.Publisher("/rgb/image_crop", Image, queue_size=1)

        # dataset vars
        self.dstdir_dataset = rospy.get_param("dstdir_dataset")
        self.bag_name = rospy.get_param("bag_name")
        rospy.loginfo("[InhandpyDatasetCSVWriter] Extracting bag {0}.bag".format(self.bag_name))

        self.data_list = []
        self.data_csvname = "poses_imgs"
        os.popen("mkdir -p {0}/{1}/{2:04d}/color".format(self.dstdir_dataset, self.bag_name, 0), 'r')

        # contact thresholds
        self.contact_thresh = 0.1
        if (self.obj_shape == 'sphere'): self.contact_thresh = 0.1
        if (self.obj_shape == 'cube'): self.contact_thresh = 0.075

        # contact episode vars
        self.contact_episode_idx = 0
        self.counter = 0
        self.num_incontact = 0
        self.min_num_incontact = 5

        self.world_T_obj = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        if (self.obj_shape == 'sphere'): self.world_T_obj = np.array([[1., 0., 0., 0.07], [0., 1., 0., -0.0675], [0., 0., 1., 0.09], [0., 0., 0., 1.]])
        if (self.obj_shape == 'cube'): self.world_T_obj = np.array([[1., 0., 0., -0.15], [0., 1., 0., -0.1475], [0., 0., 1., 0.02], [0., 0., 0., 1.]])

        # contact flag publisher
        self.contact_flag_pub = rospy.Publisher("/digit/contact/flag", OverlayText, queue_size=1)

        # cloud publisher: accumulate contact points over timesteps
        self.cloud_pub = rospy.Publisher("/digit/center/cloud", PointCloud, queue_size=1)
        self.cloud_msg = PointCloud()

        # read mean, std images from file for the particular bag dataset
        rospack = rospkg.RosPack()
        self.path_pkg = rospack.get_path('digit_inhand')
        filename = "{0}/local/resources/digit/{1}/mean_std_img.json".format(self.path_pkg, self.bag_name)
        with open(filename) as f:
            data = json.load(f)
            self.mean_img = np.asarray(data['mean_img'], dtype=np.float32)
            self.std_img = np.asarray(data['std_img'], dtype=np.float32) 

    def in_contact(self, img):

        # Compute per-image sum of stddev squared
        diff = np.linalg.norm((img - self.mean_img)/self.std_img)**2
        diff = diff / self.mean_img.size

        # Count the percent of pixels that are significantly different from their mean values
        diff_cnt = np.sum(((img - self.mean_img)/self.std_img)**2 > 4**2)
        diff_cnt = float(diff_cnt) / float(self.mean_img.size)

        # rospy.loginfo("[InhandpyDatasetCSVWriter::callback_digit_image] diff_cnt: {}".format(diff_cnt))
        contact_flag = diff_cnt > self.contact_thresh

        # rospy.loginfo("diff_cnt: {0}, contact_flag: {1}\n".format(diff_cnt, contact_flag))
        contact_flag_msg = OverlayText()
        contact_flag_msg.text = "IN CONTACT" if (contact_flag is True) else ""
        self.contact_flag_pub.publish(contact_flag_msg)

        rospy.loginfo("[InhandpyDatasetCSVWriter::in_contact] diff_cnt: {0}, contact_flag {1}".format(diff_cnt, contact_flag))

        return contact_flag

    def rosimg_to_numpy(self, imgmsg):
        if hasattr(imgmsg, 'format') and 'compressed' in imgmsg.format:
            img = np.asarray(Image.open(io.BytesIO(imgmsg.data)))
            return img

        return np.frombuffer(imgmsg.data, dtype=np.uint8).reshape(imgmsg.height, imgmsg.width, 3)[:, :, ::-1]
    
    def interpolate_img(self, img, rows, cols):
        img = cv2.resize(img, dsize=(cols, rows),interpolation=cv2.INTER_AREA)
        return img

    def save_episode_step(self, eps_idx, step_idx, img_color, obj_pos, obj_ori, digit_pos, digit_ori):

        img_color_loc = "{0:04d}/color/{1:04d}.png".format(eps_idx, step_idx)

        # reshape img to match tacto: (240,320,3) -> (120,160,3) -> (160,120,3)
        img_color = self.interpolate_img(img=img_color, rows=120, cols=160)
        img_color = np.transpose(img_color, (1,0,2))
        
        rospy.loginfo("[save_episode_step::in_contact] img_color_loc: {0}/{1}/{2}".format(
            self.dstdir_dataset, self.bag_name, img_color_loc))

        imageio.imwrite("{0}/{1}/{2}".format(self.dstdir_dataset,
                                             self.bag_name, img_color_loc), img_color)

        img_normal_loc = "{0:04d}/normal/{1:04d}.png".format(eps_idx, step_idx)

        data_row = {'obj_pos': obj_pos.tolist(),
                    'obj_ori': obj_ori.tolist(),
                    'digit_pos': digit_pos.tolist(),
                    'digit_ori': digit_ori.tolist(),
                    'img_color_loc': img_color_loc,
                    'img_normal_loc': img_normal_loc
                    }

        self.data_list.append(data_row)

    def save_episode_dataset(self, eps_idx):
        
        csvfile = "{0}/{1}/{2:04d}/{3}.csv".format(self.dstdir_dataset, self.bag_name, eps_idx, self.data_csvname)
        self.data_frame = pd.DataFrame(self.data_list)
        self.data_frame.to_csv(csvfile)

        rospy.loginfo("Saving episode {0} to {1}".format(eps_idx, csvfile))

        # reset vars for a new episode
        self.data_list = []

        os.popen("mkdir -p {0}/{1}/{2:04d}/color".format(self.dstdir_dataset, self.bag_name, eps_idx+1), 'r')

    def callback_overhead_image(self, msg):

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # img = self.rosimg_to_numpy(msg)
        except CvBridgeError as e:
            rospy.logwarn(
                "[InhandpyDatasetCSVWriter::callback_digit_image] {0}".format(e))
            return

        img_crop = img[100:500, 450:1000, :] # 720 x 1280 x 4
        img_msg = self.bridge.cv2_to_imgmsg(img_crop, encoding='passthrough')
        self.overhead_image_crop_pub.publish(img_msg)

    def callback_digit_image(self, msg):

        try:
            img = self.rosimg_to_numpy(msg)
        except CvBridgeError as e:
            rospy.logwarn(
                "[InhandpyDatasetCSVWriter::callback_digit_image] {0}".format(e))
            return

        try:
            # looks up arg2 frame transform in arg1 frame
            # (obj_pos, obj_ori) = self.tf_listener.lookupTransform(
            #     "world", "/object/center/", rospy.Time(0))
            (tf_digit_pos, tf_digit_ori) = self.tf_listener.lookupTransform(
                "world", "/digit/center/", rospy.Time(0)) # returns list

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(
                "[InhandpyDatasetCSVWriter::callback_digit_image] TF lookup failed")
            return

        if (self.in_contact(img)):

            header = msg.header
            header.frame_id = "world"
            self.cloud_msg.header = header
            self.cloud_msg.points.append(Point32(tf_digit_pos[0], tf_digit_pos[1], tf_digit_pos[2]))
            self.cloud_pub.publish(self.cloud_msg)

            # tf listener
            tf_digit_ori = tf.transformations.quaternion_matrix(tf_digit_ori)[0:3, 0:3] # 4x4
            tf_digit_pos, tf_digit_ori = np.array(tf_digit_pos), np.array(tf_digit_ori)
            world_T_digit = np.eye(4)
            world_T_digit[0:3, 0:3], world_T_digit[0:3,-1] = tf_digit_ori, tf_digit_pos
            
            # digit_pos = np.array([0.,0.,0.011])
            # digit_ori = np.array([[2.220446049250313e-16, -0.0, -1.0],
            #                       [0.0, 1.0, -0.0], [1.0, 0.0, 2.220446049250313e-16]])
            # T_digit = np.eye(4)            
            # T_digit[0:3, 0:3], T_digit[0:3,-1] = digit_ori, digit_pos

            # placing digit in a new world frame
            world_prime_T_digit = np.eye(4)

            # object pose in new world frame
            world_prime_T_obj = np.matmul(world_prime_T_digit, np.matmul(np.linalg.inv(world_T_digit), self.world_T_obj))

            # write to file
            T_digit = world_prime_T_digit
            T_obj = world_prime_T_obj
            obj_pos, obj_ori = T_obj[0:3,-1], T_obj[0:3,0:3]
            digit_pos, digit_ori = T_digit[0:3,-1], T_digit[0:3,0:3]

            self.save_episode_step(eps_idx=self.contact_episode_idx, step_idx=self.num_incontact, img_color=img,
                                   obj_pos=obj_pos, obj_ori=obj_ori, digit_pos=digit_pos, digit_ori=digit_ori)

            self.num_incontact = self.num_incontact + 1

        else:
            self.counter = self.counter + 1

        # start new contact episode
        if ((self.counter > 10) & (self.num_incontact > 1)):

            if (self.num_incontact > self.min_num_incontact):
                self.save_episode_dataset(eps_idx=self.contact_episode_idx)
                self.contact_episode_idx = self.contact_episode_idx + 1
            else:
                self.data_list = []

            self.counter = 0
            self.num_incontact = 0

def main():

    rospy.init_node('inhandpy_dataset_csv_writer', anonymous=True)
    rospy.loginfo("Initialized inhandpy_dataset_csv_writer node.")

    inhandpy_dataset_writer = InhandpyDatasetCSVWriter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
