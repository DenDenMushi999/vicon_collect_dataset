import os
import time
from pathlib import Path
import argparse

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
    def __init__(self, img_topic: str, out_dir, dataset_size=100, fps=10):
        self.img_topic = img_topic
        self.out_dir = out_dir
        self.dataset_size = dataset_size
        self.fps = fps

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
            self.count += 1
            self.last_time = time.time() - self.t_start
            self.last_frame = img
            if ( self.count <= self.dataset_size or self.dataset_size < 0):
                self.times_arr.append(self.last_time)
                cv.imwrite( str(self.out_dir/'image_0'/(str(self.count-1).zfill(6) + '.png')), self.last_frame)
            else:
                rospy.loginfo('Collecting dataset is finished. Saving all results...')
                if not self.is_finished:
                    self.is_finished = True
                    with open( str(self.out_dir/'times.txt'), 'w') as f:
                        f.write('\n'.join(str(t) for t in self.times_arr))


    # def timer_callback(self, event=None):
    #     if (np.any(self.last_frame)):
    #         self.count += 1
    #     if ( self.count <= self.dataset_size or self.dataset_size < 0):
    #         if (np.any(self.last_frame)):
    #             self.times_arr.append(self.last_time)
    #             cv.imwrite( str(self.out_dir/'image_0'/(str(self.count-1).zfill(6) + '.png')), self.last_frame)
    #     else:
    #         rospy.loginfo('Collecting dataset is finished. Saving all results...')
    #         if not self.is_finished:
    #             self.is_finished = True
    #             with open( str(self.out_dir/'times.txt'), 'w') as f:
    #                 f.write('\n'.join(str(t) for t in self.times_arr))


    def start_listen(self):
        print('listener started')
        rospy.init_node('posed_images_collector', anonymous=True)

        self.out_dir = Path(self.out_dir)
        create_dir(self.out_dir/'image_0')

        # timer = rospy.Timer(rospy.Duration(1/self.fps), self.timer_callback)
        image_sub = rospy.Subscriber( self.img_topic, Image, self._image_callback, queue_size=2)

        rospy.spin()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_topic', type=str, default='/camera0/color/image_raw', help='topic where images are published')
    parser.add_argument('--out_dir', type=str, help='where to save dataset')
    parser.add_argument('--dataset_size', type=int, default=250, help='number of images in dataset')
    parser.add_argument('--fps', type=int, default=15, help='fps in output dataset')

    args = parser.parse_args()
    img_topic = args.img_topic
    out_dir = args.out_dir
    dataset_size = args.dataset_size
    # fps = args.fps

    images_collector = StampedImagesCollector(img_topic, out_dir, dataset_size)

    images_collector.start_listen()

if __name__ == "__main__":
    main()