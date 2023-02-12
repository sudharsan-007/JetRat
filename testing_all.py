# !/usr/bin/env python3.6
# coding: utf-8

# Python imports
import time
import Jetson.GPIO as GPIO
import board
import adafruit_vl53l0x
import threading
import torch
import numpy as np
# Imports
from Racestick import RaceStick
from nvidia_racecar import NvidiaRacecar
from camera import Camera
# from camera import record_img, dir_check
from autonomous import RunningDataset, autonomous_preprocess
from Models import NetworkNvidia, ResNet18
from Train import train_model



GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)



def autonomous_mode(frame):

    img = autonomous_preprocess(frame)
    net.eval()
    with torch.no_grad():
        out = net(img.to(device, dtype=torch.float))
    print("Predictions",out)

    car.steering = float(out[0][0]*1.0) 


def telemetry_mode():
    global throttle, steering, steeringoffset
    throttle, steering, steeringoffset = controls.GetThrottleSteering()
    steering = -steering
    car.steering_offset = steeringoffset
    car.RunCar(throttle, steering)


def data_collect_mode(frame):
    
    global prev_cap_time
    curr_time = time.time()
    if curr_time - prev_cap_time >= capture_interval:
        
        if capture_interval >= 1:
            fname = str(int(time.time()))    
        if capture_interval < 1:
            fname = str(round(time.time(),2)).replace(".","_")
        cam.save_img_data(frame, fname)
        cam.record_data(fname, steering, throttle)
            
        print("Capturing {}".format(frame.shape))
        prev_cap_time = curr_time

            

def training_mode():
    global doneTraining
    print("Training started now")
    train_gen = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)
    print("train_generation done")
    doneTraining = train_model(1, net, train_gen, criterion, optimizer, "cuda")

def training_helper():
    global doneTraining
    while True:
        if doneTraining: 
            doneTraining = training_mode()
        if trainingState == False:
            break 
        
    

def pwm_throttle():
    global prev_pwm_time
    curr_pwm_time = time.time()
    high_state = 0.2
    low_state = 0.25
    if curr_pwm_time-prev_pwm_time<=high_state:
        car.throttle = 0.8
    elif curr_pwm_time-prev_pwm_time<=high_state+low_state:
        car.throttle = 0.0
    else:
        prev_pwm_time = curr_pwm_time 


def update_cam_data(frame):
    global prev_data_time, data_time_interval, data
    curr_data_time = np.full((8,), time.time())
    where = np.where(curr_data_time - prev_data_time >= data_time_interval)[0]
    for i in where:
        img = frame
        img = autonomous_preprocess(frame)
        data[i] = (img[0], steering)
        prev_data_time[i] = curr_data_time[i] 
    

if __name__=="__main__":
    
    led_channels = [32, 36, 38, 40]
    GPIO.setup(led_channels, GPIO.OUT)
    
    batch_size = 8
    # Initializing the car
    car = NvidiaRacecar()
    controls = RaceStick()
    cam = Camera()
    i2c = board.I2C()
    sensor = adafruit_vl53l0x.VL53L0X(i2c)
    sensor.measurement_timing_budget = 20000 # Get faster but much lower accuracy results
    data = RunningDataset(batch_size) 
    model_laod_flag = False 
    doneTraining = True

    throttle = 0.0
    steering = 0.0
    steeringOffset = 0.0
    capture_interval = 0.2
    prev_pwm_time = time.time()
    prev_cap_time = time.time() 
    prev_data_time = np.full((8,), time.time())
    # data_time_interval = np.array([32, 16, 8, 4, 2, 1 , 0.5, 0.25])
    data_time_interval = np.array([64., 32., 16., 8., 4., 2., 1. , 0.5])
    
    if model_laod_flag:
        PATH = "models/Nvidia_net.pth"
        device = torch.device('cuda')
        net = NetworkNvidia().to(device)
        checkpoint = torch.load(PATH, map_location=device)
        # print("##### PRINTING KEYS BELOW ####")
        # print(checkpoint.keys())
        # print("------------------------------")
        net.load_state_dict(checkpoint)
        net.eval()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
        print('Model loaded') 

    remoteControl, collectDataState, trainingState, autonomousState, forceStop = controls.GetDriveState()
    prevTrainingState = trainingState

    time.sleep(0.4)
    
    while not forceStop: # This loop should run at 0.0084 seconds or 84 milli seconds 
        GPIO.output(led_channels[0], GPIO.HIGH)
        time.sleep(0.0062)
        
        #print("Range: {0}mm".format(sensor.range)) # Approx takes 20 milli seconds 
        if sensor.range<600:
            car.throttle = 0.0
            print("Object detected in the front")

        # time.sleep(0.2)
        frame,ret = cam.return_frame()
        # The rest of the code takes 0.0004 seconds or 4ms 
        start = time.time()
        update_cam_data(frame)
        remoteControl, collectDataState, trainingState, autonomous, forceStop = controls.GetDriveState()
        
        if controls.GetConsThrottle() == True:
            pwm_throttle()
            
        if remoteControl == True and autonomous == False:
            GPIO.output(led_channels[1], GPIO.HIGH) 
            telemetry_mode()
        
        if remoteControl == False:
            GPIO.output(led_channels[1], GPIO.LOW) 
            
        if autonomous == True and trainingState == False and remoteControl == False:
            autonomous_mode(frame)
            GPIO.output(led_channels[2], GPIO.HIGH) 
            
        if autonomous == False:
            GPIO.output(led_channels[2], GPIO.LOW) 
            
        if trainingState == True  and prevTrainingState!=trainingState and autonomous == False:
            print("starting training mode")
            training_process = threading.Thread(target = training_helper)
            training_process.start()
            GPIO.output(led_channels[3], GPIO.HIGH) 
            prevTrainingState = trainingState 
        
        if trainingState == False and prevTrainingState!=trainingState: 
            print("stopping training mode")
            training_process.join()
            GPIO.output(led_channels[3], GPIO.LOW) 
            prevTrainingState = trainingState


        if collectDataState == True:
            # GPIO.output(led_channels[3], GPIO.HIGH) 
            print("collecting data now")
            data_collect_mode(frame)
        if collectDataState == False:
            GPIO.output(led_channels[3], GPIO.LOW) 
            
        print(remoteControl, collectDataState, trainingState, autonomous, forceStop, throttle, steering)
        #print(start-end)
    else: 
        GPIO.output(led_channels, GPIO.LOW) 
        print("force stopped")
        GPIO.cleanup()
        
   
