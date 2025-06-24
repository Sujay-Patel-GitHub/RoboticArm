# get live video using cv fast
import time
from threading import Thread

import cv2
import numpy as np

ARUCO_DICT = {
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
}




class PositionEstimation:

    def detect_marker(self, img):
        arucoDict = cv2.aruco.Dictionary(ARUCO_DICT["DICT_APRILTAG_16h5"])
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
        if len(corners) > 0:
            return (corners, ids)
        else:
            return (None, None)

    def calculate_center(self, cor1):
        if cor1 is not None:
            for i in range(len(cor1)):
                x1 = cor1[i][0][0][0]
                y1 = cor1[i][0][0][1]
                x2 = cor1[i][0][1][0]
                y2 = cor1[i][0][1][1]
                x3 = cor1[i][0][2][0]
                y3 = cor1[i][0][2][1]
                x4 = cor1[i][0][3][0]
                y4 = cor1[i][0][3][1]
                x = (x1 + x2 + x3 + x4) / 4
                y = (y1 + y2 + y3 + y4) / 4
                # print("x1: {}, y1: {}".format(x, y))
                return int(x), int(y)
            else:
                return None

    def init_cam(self,Horizontal_Resolution, Vertical_Resolution, focal_length):
        self.Horizontal_Resolution = Horizontal_Resolution
        self.Vertical_Resolution = Vertical_Resolution
        self.focal_length = focal_length

        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.Horizontal_Resolution)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.Vertical_Resolution)
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def get_img_and_estimate_position(self):

        while not(cv2.waitKey(1) & 0xFF == ord('q')):
            stat_time = time.time()
            ret, img = self.cap.read()
            print("time: {}".format(int((time.time() - stat_time) * 1000)))

            img1 = np.hsplit(img, 2)[0]
            img2 = np.hsplit(img, 2)[1]
            img = img1

            grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # sharpen the image using kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

            grey1 = cv2.filter2D(grey1, -1, kernel)
            grey2 = cv2.filter2D(grey2, -1, kernel)


            cor1, ids1 = self.detect_marker(grey1)
            cor2, ids2 = self.detect_marker(grey2)
            # print("time: {}".format(int((time.time() - stat_time) * 1000)))

            center1 = self.calculate_center(cor1)
            center2 = self.calculate_center(cor2)

            if center1 is not None and center2 is not None:
                x1, y1 = center1
                x2, y2 = center2
                if (x1 - x2) != 0:
                    d = (self.focal_length * 12) / abs((x1 - x2))
                    # fps put

                    from math import sin, pi

                    # calculate the angle
                    hd1 = sin(pi / 4) * d * (x1 / self.Horizontal_Resolution)
                    vd1 = sin(pi / 6) * d * (y1 / self.Vertical_Resolution)
                    hd2 = sin(pi / 4) * d * (x1 / self.Horizontal_Resolution)
                    vd2 = sin(pi / 6) * d * (y1 / self.Vertical_Resolution)
                    # show the image with the distance

                    txt = "D: {}".format(round(d, 2)) + " HD: {}".format(round(hd1, 2)) + " VD : {}".format(
                        round(vd1, 2)) + " cm"
                    cv2.putText(img, txt, center1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

            fps = int(1 / (time.time() - stat_time))
            cv2.putText(img, "FPS: {}".format(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

            # resize the image
            img = cv2.resize(img, (1280, 720))
            #convert to gray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #sharp the image
            img = cv2.filter2D(img, -1, kernel)
            #show the image

            cv2.imshow("img", img)

            cv2.waitKey(1)
            # print("time: {}".format(int((time.time() - stat_time) * 1000)))
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    pos = PositionEstimation()
    pos.init_cam(4320*2, 1440, 1400)
    print("Press 'q' to quit")
    pos.get_img_and_estimate_position()

