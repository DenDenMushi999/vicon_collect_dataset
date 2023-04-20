import yaml
import time 
from pathlib import Path

import cv2 as cv
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped


class StampedImagesCollector:
    def __init__(self, img_topic: str, mode='video', img_path=None, rate=1):
        self.img_topic = img_topic
        self.t_start = time.time()
        self.mode = mode

        self.cv_bridge = CvBridge()

        self.rate = rate
        self.t_start = time.time()
        self.last_time = None

        self.last_frame = None
        self.last_det_img = None
        self.times_arr = []

        camera_params_file = 'configs/realsense_d435i_640.yaml'
        with open(camera_params_file) as f:
            camera_params = yaml.load(f, Loader=yaml.FullLoader)

        self.intrinsics = np.asarray(camera_params['intrinsics'], dtype=np.float32)
        self.dist_coeffs = np.asarray(camera_params['distortion'], dtype=np.float32)

        self.markerLength = 0.036
        self.aruco_params = cv.aruco.DetectorParameters_create()
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
            
    
    def start_listen(self):
        if self.mode == 'video':
            cv.waitKey(2000)
            print('listener started')
            rospy.init_node('aruco_detector', anonymous=True)

            self.image_sub = rospy.Subscriber( self.img_topic, Image, self._image_callback)
            self.det_pub = rospy.Publisher(str(Path(img_topic).parent/"aruco_det_img"), Image, queue_size=1)
            time.sleep(2)
            self.timer = rospy.Timer(rospy.Duration(1/self.rate), self.timer_callback)
            rospy.spin()
        else:
            self.last_frame = cv.imread(img_path)


    def _image_callback(self, msg):
        try:
            img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            rospy.logerr("Error: cannot convert from img_msg to cv2_image")
            self.last_frame = None
            self.last_time = None

        self.last_time = time.time() - self.t_start
        self.last_frame = img
        self.times_arr.append(self.last_time)

    def timer_callback(self, event=None):
        self.aruco_detect()


    def aruco_detect(self):
        print('detecting arucos')
        (corners, ids, rejected) = cv.aruco.detectMarkers( self.last_frame, self.aruco_dict,
                parameters=self.aruco_params)

        aruco_pts = np.zeros((4,3), dtype=np.float32)
        markerLength = self.markerLength
        aruco_pts[0] = [-markerLength/2, markerLength/2, 0]
        aruco_pts[1] = [markerLength/2, markerLength/2, 0]
        aruco_pts[2] = [markerLength/2, -markerLength/2, 0]
        aruco_pts[3] = [-markerLength/2, -markerLength/2, 0]

        self.last_det_img = self.last_frame.copy()
        if (ids is not None):
            nMarkers = ids.size
            print(f'detected {nMarkers} markers')
            print(f'ids: {ids}')
            self.last_det_img = cv.aruco.drawDetectedMarkers(self.last_frame, corners, ids)

            aruco_pos_arr = []
            aruco_rot_arr = []
            aruco_T_arr = []

            for i, c in enumerate(corners):
                retval, rvec, tvec = cv.solvePnP( aruco_pts, c, self.intrinsics, self.dist_coeffs)	
                if retval:
                    rot = np.zeros((3,3))
                    cv.Rodrigues(rvec, rot)
                    aruco_pos_arr.append(tvec)
                    aruco_rot_arr.append(rvec)
                    T = np.zeros((4,4))
                    T[:3,:3] = rot
                    T[:3,3] = tvec.reshape((3,))
                    T[3,3] = 1
                    aruco_T_arr.append(T)
                    # rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers( corners, 0.18, intrinsics, dist_coeffs )
                    print(f'tvec: {tvec}, rvec: {rvec}')
                    self.last_det_img = cv.drawFrameAxes(self.last_det_img, self.intrinsics, self.dist_coeffs, rvec, tvec, 0.1)
                # corners_galery_rel_galery = np.array([ [[galery_map_size[0] + padding_size], [galery_map_size[1] + padding_size], [0.]],
                #                                        [[-padding_size], [galery_map_size[1] + padding_size], [0.]],
                #                                        [[-padding_size], [-padding_size], [0.]],
                #                                        [[galery_map_size[0] + padding_size], [-padding_size], [0.]] ])

            # X rot
            Rot_ArW = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
            
            Trans_ArW = np.array([[-0.1,-0.1, 0.2]])
            # Trans_ArW = np.array([[0,0,0]])
            T_ArW = np.zeros((4,4))
            T_ArW[:3,:3] = Rot_ArW
            T_ArW[:3,3] = Trans_ArW
            T_ArW[3,3] = 1
            
            # aruco_rot_m = np.zeros((3,3))
            # cv.Rodrigues(aruco_rot_arr[2], aruco_rot_m)
            # rvec_world =  aruco_rot_m @ aruco_to_world_rot_m
            
            ind = np.where(ids.flatten() == 0)
            print('ind: ', ind)
            if (ind[0].size != 0):
                T_wc = aruco_T_arr[ind[0][0]] @ T_ArW

                # print(rvec_world)
                # self.last_det_img = cv.drawFrameAxes(self.last_det_img, intrinsics, dist_coeffs, rvec_world, aruco_pos_arr[2], 0.1)
                self.last_det_img = cv.drawFrameAxes(self.last_det_img, self.intrinsics, self.dist_coeffs, T_wc[:3,:3], T_wc[:3,3], 0.1)

            # if rvecs is not None :
            #     for rvec, tvec, id in zip(rvecs, tvecs, ids) :
            #         rotation_matrix = np.zeros(shape=(3,3))
            #         cv.Rodrigues(rvec, rotation_matrix)
        else:
            print('didn\'t find arucos')
        img_msg = self.cv_bridge.cv2_to_imgmsg(self.last_det_img, encoding="rgb8")
        self.det_pub.publish(img_msg)
        print('published_image')


# corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(img)

mode = 'video'
img_path = 'aruco.png'
img_topic = '/camera/color/image_raw'
rate = 5

image_collector = StampedImagesCollector(img_topic, mode='video', rate=rate)
image_collector.start_listen()



# while True:
#     print('video')
#     if mode == 'video': 
#         retval, img = cap.retrieve()
#         print(img.shape)
#         cv.imshow('cam', img)

#     (corners, ids, rejected) = cv.aruco.detectMarkers( img, aruco_dict,
#                 parameters=aruco_params)

#     aruco_pts = np.zeros((4,3), dtype=np.float32)
#     aruco_pts[0] = [-markerLength/2, markerLength/2, 0]
#     aruco_pts[1] = [markerLength/2, markerLength/2, 0]
#     aruco_pts[2] = [markerLength/2, -markerLength/2, 0]
#     aruco_pts[3] = [-markerLength/2, -markerLength/2, 0]

#     img_det = img.copy()
#     if (ids is not None):
#         img_det = cv.aruco.drawDetectedMarkers(img, corners, ids)
#         cv.imshow('det', img_det)

#         nMarkers = ids.size
#         aruco_pos_arr = []
#         aruco_rot_arr = []
#         aruco_T_arr = []
#         for i, c in enumerate(corners):
#             retval, rvec, tvec = cv.solvePnP( aruco_pts, c, intrinsics, dist_coeffs)	
#             if retval:
#                 rot = np.zeros((3,3))
#                 cv.Rodrigues(rvec, rot)
#                 aruco_pos_arr.append(tvec)
#                 aruco_rot_arr.append(rvec)
#                 T = np.zeros((4,4))
#                 T[:3,:3] = rot
#                 T[:3,3] = tvec.reshape((3,))
#                 T[3,3] = 1
#                 aruco_T_arr.append(T)
#                 # rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers( corners, 0.18, intrinsics, dist_coeffs )
#                 print(f'tvec: {tvec}, rvec: {rvec}')
#                 img_det = cv.drawFrameAxes(img_det, intrinsics, dist_coeffs, rvec, tvec, 0.1)
#             # corners_galery_rel_galery = np.array([ [[galery_map_size[0] + padding_size], [galery_map_size[1] + padding_size], [0.]],
#             #                                        [[-padding_size], [galery_map_size[1] + padding_size], [0.]],
#             #                                        [[-padding_size], [-padding_size], [0.]],
#             #                                        [[galery_map_size[0] + padding_size], [-padding_size], [0.]] ])

#         # X rot
#         Rot_ArW = np.array([[1, 0, 0],
#                             [0, 0, 1],
#                             [0, -1, 0]])
        
#         Trans_ArW = np.array([[-0.1,-0.1, 0.2]])
#         # Trans_ArW = np.array([[0,0,0]])
#         T_ArW = np.zeros((4,4))
#         T_ArW[:3,:3] = Rot_ArW
#         T_ArW[:3,3] = Trans_ArW
#         T_ArW[3,3] = 1
#         aruco_rot_m = np.zeros((3,3))
#         # cv.Rodrigues(aruco_rot_arr[2], aruco_rot_m)
#         # rvec_world =  aruco_rot_m @ aruco_to_world_rot_m
#         T_wc = aruco_T_arr[3] @ T_ArW

#         # print(rvec_world)
#         # img_det = cv.drawFrameAxes(img_det, intrinsics, dist_coeffs, rvec_world, aruco_pos_arr[2], 0.1)
#         img_det = cv.drawFrameAxes(img_det, intrinsics, dist_coeffs, T_wc[:3,:3], T_wc[:3,3], 0.1)

#         # if rvecs is not None :
#         #     for rvec, tvec, id in zip(rvecs, tvecs, ids) :
#         #         rotation_matrix = np.zeros(shape=(3,3))
#         #         cv.Rodrigues(rvec, rotation_matrix)

#     cv.imshow('det_3d', img_det)

#     key = cv.waitKey(100)
#     if key == ord('q'):
#         break
