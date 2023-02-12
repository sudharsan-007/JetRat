import cv2
import os
import time
import pandas as pd
import numpy as np

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,  
    capture_height=720, 
    display_width=640,  
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

            
class Camera():
    
    def __init__(self):
        self.vid = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        
        self.image_dir_path = "./data/img"
        self.data_dir_path = "./data"
        self.mtx = np.load("calibration_matrix.npy")
        self.dist = np.load("distortion_coefficients.npy")
        
        self.check_dir()
        
        if not os.path.isfile(os.path.join(self.data_dir_path,"data.csv")):
            pd.DataFrame({},columns=["Epoch_time","Steering","Throttle"]).to_csv(f"{self.data_dir_path}/data.csv")
            print("No previous data found, creating new file")
        
        print("Camera initialised ")
        
    def return_frame(self):
        ret, frame = self.vid.read()
        frame = cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)
        return frame, ret 
    
    def check_dir(self):
        # checking if  images dir is exist not, if not then create images directory
        CHECK_DIR = os.path.isdir(self.image_dir_path)
        if not CHECK_DIR:
            os.makedirs(self.image_dir_path)
            print(f'"{self.image_dir_path}" Directory is created')
        else:
            print(f'"{self.image_dir_path}" Directory already Exists.')
            
    def record_data(self, curr_time, Steering, Throttle):
        data = {
                'Epoch_time': [str(curr_time)],
                'Steering': [str(Steering)],
                'Throttle': [str(Throttle)],
            }
        df = pd.DataFrame(data)
        df.to_csv(f"{self.data_dir_path}/data.csv", mode='a', index=False, header=False)
        
    def save_img_data(self, frame, fname):
        # fname = str(round(time.time())).replace(".","_")
        cv2.imwrite(f"{self.image_dir_path}/{fname}.png", frame)
        
    

def data_collect_mode():
    cam = Camera()
    interval = 1.0
    prev_time = time.time()
    while True:
        if time.time() - prev_time >= interval:
            prev_time = time.time()
            frame, ret = cam.return_frame()
            if interval >= 1:
                cam.record_data(str(int(time.time())), "throttle", "steering")
            else:
                cam.record_data(str(round(time.time())).replace(".","_"), "throttle", "steering")
            cam.save_img_data(frame,"asdf")
            
            
            print("I am running {}".format(frame.shape))
        
        


    
if __name__=="__main__":
    
    data_collect_mode()
    
    