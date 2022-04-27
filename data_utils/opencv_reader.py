import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

class OpenCV2Reader:
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        # Check if camera opened successfully
        if (self.cap.isOpened()== False): 
            print(f"Error opening video file: {filename}")

    def read(self):
        frames = []
        # Read until video is completed
        while(self.cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret == True:
                frames.append(np.asarray(frame))
            # Break the loop
            else: 
                break
        # When everything done, release the video capture object
        self.cap.release()

        return frames