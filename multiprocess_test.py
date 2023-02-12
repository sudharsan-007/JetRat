# !/usr/bin/env python3.6
# coding: utf-8

# Python imports
import time
import threading
import torch

# Imports
from Racestick import RaceStick
from nvidia_racecar import NvidiaRacecar
from camera import Camera
# from camera import record_img, dir_check
from autonomous import img_preprocess, img_test, NetworkNvidia, autonomous_preprocess
from Train import SelfTrainer



def autonomous_mode(frame):
    
    img = autonomous_preprocess(frame)
    # print(img.shape)
    throttle, _, _ = controls.GetThrottleSteering()
    net.eval()
    with torch.no_grad():
        out = net(img.to(device, dtype=torch.float))
    print("Predictions",out)
    Steering = float(out[0][0]*1.2)
    #Throttle = float(out[0][1]*1.2)

    car.RunCar(throttle, Steering)


def telemetry_mode():
    global throttle, steering, steeringoffset
    time.sleep(0.1)
    throttle, steering, steeringoffset = controls.GetThrottleSteering()
    steering = -steering
    car.steering_offset = steeringoffset
    car.RunCar(throttle, steering)


def data_collect_mode(interval):
    
    cam = Camera()
    global collectDataState
    
    while True:
        frame, ret = cam.return_frame()
        
        if interval >= 1:
            fname = str(int(time.time()))
            cam.record_data(fname, steering, throttle)
        if interval < 1:
            fname = str(round(time.time(),2)).replace(".","_")
            cam.record_data(fname, steering, throttle)
        cam.save_img_data(frame, fname)
            
        print("I am running {}".format(frame.shape))
        time.sleep(interval)
        
        if collectDataState == False:
            print("Data collection thread stopped")
            break
            

def training_mode():
    pass
    




if __name__=="__main__":
    
    # Initializing the car
    car = NvidiaRacecar()
    controls = RaceStick()
    cam = Camera()
    model_laod_flag = False

    throttle = 0.0
    steering = 0.0
    steeringOffset = 0.0
    interval = 1
    # dir_check(image_dir_path)
    
    ## Loading the model
    # PATH = "models/model2.pth"
    # device = torch.device('cuda')
    
    # net = NetworkNvidia()
    # checkpoint = torch.load(PATH)
    # print(checkpoint.keys())
    # net.load_state_dict(checkpoint)
    # net = net.to(device)
    # net.eval()
    # trainer = SelfTrainer(net, device)
    # print('Model loaded')
    
    
    
    if model_laod_flag:
        PATH = "models/nvidia_model.pth"
        device = torch.device('cuda')
        net = NetworkNvidia().to(device)
        checkpoint = torch.load(PATH, map_location=device)
        print("##### PRINTING KEYS BELOW ####")
        print(checkpoint.keys())
        print("------------------------------")
        net.load_state_dict(checkpoint)
        net.eval()
        trainer = SelfTrainer(net, device)
        print('Model loaded') 

    remoteControl, collectDataState, trainingState, autonomousState, forceStop = controls.GetDriveState()
    prevCollectState = collectDataState
    time.sleep(0.4)
    while not forceStop:
        
        # time.sleep(0.2)
        remoteControl, collectDataState, trainingState, autonomous, forceStop = controls.GetDriveState()
        
        if remoteControl == True:
            telemetry_mode()
            
        if autonomous == True:
            autonomous_mode()
            
        if trainingState == True:
            training_mode()


        if collectDataState == True and collectDataState != prevCollectState and autonomous != True:
            print("collecting data now")
            data_collection_process = threading.Thread(target = data_collect_mode, args=(interval,))
            data_collection_process.start()
            prevCollectState = collectDataState
            
        if collectDataState == False and collectDataState != prevCollectState:
            print("stopping data collection")
            data_collection_process.join()
            prevCollectState = collectDataState
            

        # This will break the loop and stop the program
        if forceStop == True:
            print("Force Stopping")
            break
            
        
            
        
        print(remoteControl, collectDataState, trainingState, autonomous, forceStop, throttle, steering)
        
    print("force stopped")
