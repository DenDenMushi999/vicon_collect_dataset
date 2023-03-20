import os
from pathlib import Path

import cv2 as cv
import time

import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped

OUT_DIR = Path("")

class PosedImagesCollector:
    def __init__(self):
        self.dataset_size = 250
        self.count = 0
        self.last_img = None
        self.last_pose = None
        self.last_frame_time = None

        self.cv_bridge = CvBridge()
        self.poses_arr = []
        self.times_arr = []

    def vicon_callback(self, msg):
        self.last_pose = []
        # (append, not summ)
        self.last_pose.append(msg.pose.translation.x)
        self.last_pose.append(msg.pose.translation.y)
        self.last_pose.append(msg.pose.translation.z)
        self.last_pose.append(msg.pose.orientation.x)
        self.last_pose.append(msg.pose.orientation.y)
        self.last_pose.append(msg.pose.orientation.z)
        self.last_pose.append(msg.pose.orientation.w)

    def image_callback(self, msg):
        try:
            img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            print("Error: cannot convert from img_msg to cv2_image")
        else:
            if (self.count < self.dataset_size):
                self.count += 1
                self.last_img = img
                self.last_frame_time = msg.header.stamp
                # save new posed frame
                self.poses_arr.append(self.last_pose)
                self.times_arr.append(self.last_frame_time)
                cv.imwrite( str(OUT_DIR/'images'/str(self.count).zfill(6)), img)
            else:
                rospy.loginfo('Collecting dataset is finished. Saving all results...')
                with open( str(OUT_DIR/'times.txt'), 'r') as f:
                    f.write('\n'.join(str(pose) for pose in self.poses_arr))


    def listener(self, obj_name: str):
        rospy.init_node('posed_images_collector', anonymous=True)

        image_sub = rospy.Subscriber("camera/color/image_raw", Image, self.image_callback, queue_size=10)
        vicon_sub = rospy.Subscriber(f"vicon/{obj_name}/{obj_name}", TransformStamped, self.vicon_callback, queue_size=10)

        rospy.spin()
