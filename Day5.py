import cv2
import numpy as np
import time

ARUCO_DICT = {
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
}

class PositionEstimation:
    def __init__(self):
        # Initialize detector for performance
        self.arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_APRILTAG_16h5"])
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)
        self.cap = None

    def detect_marker(self, img):
        try:
            corners, ids, rejected = self.detector.detectMarkers(img)
            return (corners, ids) if len(corners) > 0 else (None, None)
        except Exception as e:
            print(f"Detection failed: {str(e)}")
            return (None, None)

    def calculate_center(self, cor1):
        if cor1 is not None and len(cor1) > 0:
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
                return int(x), int(y)
        return None

    def init_cam(self, Horizontal_Resolution, Vertical_Resolution, focal_length):
        self.Horizontal_Resolution = Horizontal_Resolution
        self.Vertical_Resolution = Vertical_Resolution
        self.focal_length = focal_length
        
        # Try different camera indices if camera 3 is not available
        for camera_index in [3, 0, 1, 2]:
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                print(f"Camera opened successfully at index {camera_index}")
                break
        else:
            raise IOError("Cannot open any camera")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.Horizontal_Resolution)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.Vertical_Resolution)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # Verify actual resolution
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Requested resolution: {self.Horizontal_Resolution}x{self.Vertical_Resolution}")
        print(f"Actual resolution: {int(actual_width)}x{int(actual_height)}")

    def get_img_and_estimate_position(self):
        from math import sin, pi
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        
        while True:
            stat_time = time.time()
            ret, img = self.cap.read()
            if not ret: 
                print("Failed to read from camera")
                break
            
            # Split stereo image into left and right
            img1 = np.hsplit(img, 2)[0]
            img2 = np.hsplit(img, 2)[1]
            
            # Convert to grayscale and apply sharpening filter
            grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            grey1 = cv2.filter2D(grey1, -1, kernel)
            grey2 = cv2.filter2D(grey2, -1, kernel)

            # Detect markers in both images
            cor1, ids1 = self.detect_marker(grey1)
            cor2, ids2 = self.detect_marker(grey2)

            # Calculate centers
            center1 = self.calculate_center(cor1)
            center2 = self.calculate_center(cor2)

            # Display image (use color version for better visualization)
            display_img = img1.copy()

            # Calculate distance if markers found in both images
            if center1 is not None and center2 is not None:
                x1, y1 = center1
                x2, y2 = center2
                if abs(x1 - x2) > 0:  # Avoid division by zero
                    d = (self.focal_length * 12) / abs(x1 - x2)
                    hd1 = sin(pi / 4) * d * (x1 / (self.Horizontal_Resolution / 2))
                    vd1 = sin(pi / 6) * d * (y1 / self.Vertical_Resolution)
                    txt = f"D: {round(d, 2)} HD: {round(hd1, 2)} VD: {round(vd1, 2)} cm"
                    cv2.putText(display_img, txt, center1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Draw marker center
                    cv2.circle(display_img, center1, 5, (0, 255, 0), -1)

            # Calculate and display FPS
            elapsed_time = time.time() - stat_time
            if elapsed_time > 0:
                fps = 1 / elapsed_time
                cv2.putText(display_img, f"FPS: {round(fps, 1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Resize and display
            display_img = cv2.resize(display_img, (1280, 720))
            cv2.imshow("ArUco Position Estimation", display_img)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    pos = PositionEstimation()
    pos.init_cam(4320*2, 1440, 1400)
    print("Press 'q' to quit")
    pos.get_img_and_estimate_position()
