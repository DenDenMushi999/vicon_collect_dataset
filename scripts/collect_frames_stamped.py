import os
import time
from pathlib import Path
# from typing import Union

import cv2 as cv
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped

OUT_DIR = Path("")


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

class StampedImagesCollector:
    def __init__(self, img_topic: str, out_dir, dataset_size=100):
        self.img_topic = img_topic
        self.out_dir = out_dir
        self.dataset_size = dataset_size

        self.count = 0
        self.is_finished = False
        self.t_start = time.time()

        self.cv_bridge = CvBridge()
        self.last_frame = None
        self.last_time = None
        self.times_arr = []

    def _image_callback(self, msg):
        try:
            img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            rospy.logerr("Error: cannot convert from img_msg to cv2_image")
            self.last_frame = None
            self.last_time = None
        else:
            self.last_time = time.time() - self.t_start
            self.last_frame = img

    def timer_callback(self, event=None):
        if (np.any(self.last_frame)):
            self.count += 1
        if ( self.count <= self.dataset_size or self.dataset_size < 0):
            if (np.any(self.last_frame)):
                self.times_arr.append(self.last_time)
                cv.imwrite( str(self.out_dir/'image_0'/(str(self.count-1).zfill(6) + '.png')), self.last_frame)
        else:
            rospy.loginfo('Collecting dataset is finished. Saving all results...')
            if not self.is_finished:
                self.is_finished = True
                with open( str(self.out_dir/'times.txt'), 'w') as f:
                    f.write('\n'.join(str(t) for t in self.times_arr))

    def start_listen(self):
        print('listener started')
        rospy.init_node('posed_images_collector', anonymous=True)

        self.out_dir = Path(self.out_dir)
        create_dir(self.out_dir/'image_0')

        timer = rospy.Timer(rospy.Duration(1/10), self.timer_callback)
        image_sub = rospy.Subscriber( self.img_topic, Image, self._image_callback, queue_size=10)

        rospy.spin()


def main():
    img_topic = '/camera0/color/image_raw'
    out_dir = ''
    dataset_size = 600
    images_collector = StampedImagesCollector(img_topic, out_dir, dataset_size)

    images_collector.start_listen()

if __name__ == "__main__":
    main()