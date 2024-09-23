import time
import cv2
import numpy as np
import argparse
tvec2=None
rvec2=None
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

aruco=cv2.aruco
ARUCO_DICT = {
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250
}
matrix_coefficients=np.array([[1348.25159,0.,775.666653],
 [0.,1310.33643,367.746816],
 [0.,0.,1.]])
distortion_coefficients=np.array([0.18415926,-0.8841227,-0.00404495,0.05981116,2.72023328])
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_5X5_250", ## DICT_4X4_50
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())
vid = cv2.VideoCapture(0)
if not (vid.isOpened()):
    print("Could not open video device")
i = 0
while(True): 
    ret, frame = vid.read() 
    cv2.rectangle(frame, (10, 10), (50, 50), (0, 255, 0), 3)  
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]]) 
    parameters = cv2.aruco.DetectorParameters()  
    detector = cv2.aruco.ArucoDetector(aruco_dict,parameters)
    corners,ids,rejected =detector.detectMarkers(frame)
    if (len(corners)>0):
        id=ids.flatten()
        cv2.aruco.drawDetectedMarkers(frame, corners)
        # corners_2=np.array(corners)
        # for co
        # corners_3=corners_2.flatten()
        # print(corners_3)

        rvec, tvec, markerPoints = my_estimatePoseSingleMarkers(corners, 10, matrix_coefficients,distortion_coefficients)
        if tvec2 is not None:
            print("enters?")
            stvec = np.array(tvec)
            stvec2 = np.array(tvec2)
            stvec3 = np.subtract(stvec2, stvec)
            print(np.multiply(1,stvec3))
        tvec2=tvec.copy()    
        print("run time number: ", i)    
        
        if rvec2 is not None:
            print("enters?")
            srvec = np.array(rvec)
            srvec2 = np.array(rvec2)
            srvec3 = np.subtract(srvec2, srvec)
            try:
                cv2.drawFrameAxes(frame,matrix_coefficients,distortion_coefficients,srvec,stvec,length=5)
            except:
                print("oops")    
            #   print(srvec3)
        rvec2=rvec.copy() 
        for markerid,corner in zip(id,corners):
            corners = corner.reshape((4, 2))
            topLeft, topRight, bottomRight, bottomLeft = corners
            
            cv2.putText(frame,str(markerid),(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0), 2)
               
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)  
    if cv2.waitKey(300) & 0xFF == ord('q'): 
        break 

    i=i+1

vid.release() 
cv2.destroyAllWindows() 
