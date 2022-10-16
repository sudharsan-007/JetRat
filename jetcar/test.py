#!/usr/bin/env python
# coding: utf-8

# Load the gamepad and time libraries
import Gamepad
import time
from nvidia_racecar import NvidiaRacecar

# Gamepad settings
gamepadType = Gamepad.example
pollInterval = 0.1
#buttonExit = 'PS'

def joystick_drive():
    speed =  -gamepad.axis(3)     # Speed is inverted
    steering = -gamepad.axis(0)   # Steering control (not inverted)
    print('%+.1f %% speed, %+.1f %% steering' % (speed * 100, steering * 100))
    # print("axis 0: {}, axis 1: {}, axis 2: {}, axis 3: {}, axis 4: {}, axis 5:{}"
    #         .format(gamepad.axis(0),gamepad.axis(1),gamepad.axis(2),gamepad.axis(3), gamepad.axis(4),gamepad.axis(5)))
    car.throttle = speed # * 0.7
    car.steering = steering * 0.7
    

# Initializing the car
car = NvidiaRacecar()
# Set some initial state
speed = 0.0
steering = 0.0

car.throttle_gain = 0.7
car.steering_gain = 0.6
car.steering_offset = 0.0



# Wait for a connection
if not Gamepad.available():
    print('Please connect your gamepad...')
    while not Gamepad.available():
        time.sleep(1.0)
gamepad = gamepadType()
print('Gamepad connected')



# Start the background updating
gamepad.startBackgroundUpdates()

# Joystick events handled in the background
try:
    while gamepad.isConnected():
        # Check for the exit button
        # if gamepad.beenPressed(buttonExit):
        #     print('EXIT')
        #     break
        joystick_drive()
        
        # time.sleep(pollInterval) # Sleep for our polling interval
finally:
    # Ensure the background thread is always terminated when we are done
    car.throttle = 0.0
    car.steering = 0.0
    gamepad.disconnect()
