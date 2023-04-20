import yaml

import cv2 as cv
import numpy as np

camera_params_file = 'configs/realsense_d435i_640.yaml'
with open(camera_params_file) as f:
    camera_params = yaml.load(f, Loader=yaml.FullLoader)

intrinsics = np.asarray(camera_params['intrinsics'], dtype=np.float32)
dist_coeffs = np.asarray(camera_params['distortion'], dtype=np.float32)
# cv.VideoCapture inputVideo;
# inputVideo.open(0);

# cv.Mat cameraMatrix, distCoeffs;

markerLength = 0.036
aruco_params = cv.aruco.DetectorParameters_create()
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
print(dir(aruco_dict))
# cap = cv.VideoCapture()
# cap.open(4)
# corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(img)

print('hello')

img = cv.imread('arucos.png')
print(img.shape)
(corners, ids, rejected) = cv.aruco.detectMarkers( img, aruco_dict,
            parameters=aruco_params)



aruco_pts = np.zeros((4,3), dtype=np.float32)
aruco_pts[0] = [-markerLength/2, markerLength/2, 0]
aruco_pts[1] = [markerLength/2, markerLength/2, 0]
aruco_pts[2] = [markerLength/2, -markerLength/2, 0]
aruco_pts[3] = [-markerLength/2, -markerLength/2, 0]

print(aruco_pts.dtype)
print(corners[0].dtype)
if (ids.size > 0):
    img_det = img.copy()
    img_det = cv.aruco.drawDetectedMarkers(img, corners, ids)
    cv.imshow('det', img_det)
    cv.waitKey(2000)

    nMarkers = ids.size
    aruco_pos_arr = []
    aruco_rot_arr = []
    for i, c in enumerate(corners):
        retval, rvec, tvec = cv.solvePnP( aruco_pts, c, intrinsics, dist_coeffs)	
        aruco_pos_arr.append(tvec)
        aruco_rot_arr.append(rvec)
        # rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers( corners, 0.18, intrinsics, dist_coeffs )
        print(f'tvec: {tvec}, rvec: {rvec}')
        img_det = cv.drawFrameAxes(img_det, intrinsics, dist_coeffs, rvec, tvec, 0.1)
        # corners_galery_rel_galery = np.array([ [[galery_map_size[0] + padding_size], [galery_map_size[1] + padding_size], [0.]],
        #                                        [[-padding_size], [galery_map_size[1] + padding_size], [0.]],
        #                                        [[-padding_size], [-padding_size], [0.]],
        #                                        [[galery_map_size[0] + padding_size], [-padding_size], [0.]] ])

    aruco_to_world_rot_m = np.array([[1, 0, 0],
                                     [0, 0, -1],
                                     [0, 1, 0]])
    
    aruco_rot_m = np.zeros((3,3))
    cv.Rodrigues(aruco_rot_arr[2], aruco_rot_m)
    rvec_world =  aruco_rot_m @ aruco_to_world_rot_m
    print(rvec_world)
    img_det = cv.drawFrameAxes(img_det, intrinsics, dist_coeffs, rvec_world, aruco_pos_arr[2], 0.1)

    # if rvecs is not None :
    #     for rvec, tvec, id in zip(rvecs, tvecs, ids) :
    #         rotation_matrix = np.zeros(shape=(3,3))
    #         cv.Rodrigues(rvec, rotation_matrix)

    cv.imshow('det_3d', img_det)
    while True:
        key = cv.waitKey(100)
        if key == ord('q'):
            break


# while cap.grab():
#     retval, img = cap.retrieve()
#     if retval:
#         (corners, ids, rejected) = cv.aruco.detectMarkers( img, aruco_dict,
#             parameters=aruco_params)

#         cv.imshow('img from cam', img)
#     else:
#         print('can\'t get image, skip')

#     key = cv.waitKey(100)
#     if key == ord('q'):
#         break



# // Set coordinate system
# cv.Mat aruco_pts (4, 1, CV_32FC3);
# aruco_pts .ptr<cv.Vec3f>(0)[0] = cv.Vec3f(-markerLength/2.f, markerLength/2.f, 0);
# aruco_pts .ptr<cv.Vec3f>(0)[1] = cv.Vec3f(markerLength/2.f, markerLength/2.f, 0);
# aruco_pts .ptr<cv.Vec3f>(0)[2] = cv.Vec3f(markerLength/2.f, -markerLength/2.f, 0);
# aruco_pts .ptr<cv.Vec3f>(0)[3] = cv.Vec3f(-markerLength/2.f, -markerLength/2.f, 0);
# cv.aruco.DetectorParameters detectorParams = cv.aruco.DetectorParameters();
# cv.aruco.Dictionary dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250);
# aruco.ArucoDetector detector(dictionary, detectorParams);
# while (inputVideo.grab()) {
#     cv.Mat image, imageCopy;
#     inputVideo.retrieve(image);
#     image.copyTo(imageCopy);
#     std.vector<int> ids;
#     std.vector<std.vector<cv.Point2f>> corners;
#     detector.detectMarkers(image, corners, ids);
#     // If at least one marker detected
#     if (ids.size() > 0) {
#         cv.aruco.drawDetectedMarkers(imageCopy, corners, ids);
#         int nMarkers = corners.size();
#         std.vector<cv.Vec3d> rvecs(nMarkers), tvecs(nMarkers);
#         // Calculate pose for each marker
#         for (int i = 0; i < nMarkers; i++) {
#             solvePnP(aruco_pts , corners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
#         }
#         // Draw axis for each marker
#         for(unsigned int i = 0; i < ids.size(); i++) {
#             cv.drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
#         }
#     }
#     // Show resulting image and close window
#     cv.imshow("out", imageCopy);
#     char key = (char) cv.waitKey(waitTime);
#     if (key == 27)
#         break;
# }