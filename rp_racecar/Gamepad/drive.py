#!/usr/bin/env python
# coding: utf-8

# Load the gamepad and time libraries
import Gamepad
import time
import traitlets

from jetracer.nvidia_racecar import NvidiaRacecar

car = NvidiaRacecar()

# Gamepad settings 
gamepadType = Gamepad.PS4
pollInterval = 0.01
#buttonExit = 'PS'

# Wait for a connection
if not Gamepad.available():
    print('Please connect your gamepad...')
    while not Gamepad.available():
        time.sleep(1.0)
gamepad = gamepadType()
print('Gamepad connected')

# Set some initial state
speed = 0.0
steering = 0.0

# Start the background updating
gamepad.startBackgroundUpdates()

# Joystick events handled in the background
try:
    while gamepad.isConnected():
        # Check for the exit button
        if gamepad.beenPressed(4):
            print('EXIT')
            break

        # Update the joystick positions
        # Speed control (inverted)
        speed = -gamepad.axis(1)
        # Steering control (not inverted)
        steering = gamepad.axis(0)
        # print('%+.1f %% speed, %+.1f %% steering' % (speed * 100, steering * 100))
        print("axis 0: {}, axis 1: {}, axis 2: {}, axis 3: {}, axis 4: {}, axis 5:{}"
              .format(gamepad.axis(0),gamepad.axis(1),gamepad.axis(2),gamepad.axis(3),
                      gamepad.axis(4),gamepad.axis(5)))
        print("Button 0: {}, Button 1: {}, Button 2: {}, Button 3: {}".format(gamepad.isPressed(0),gamepad.isPressed(1),gamepad.isPressed(2),gamepad.isPressed(3)))
        
        left_link = traitlets.dlink((str(gamepad.axis(0)), 'value'), (car, 'steering'), transform=lambda x: -x)
        right_link = traitlets.dlink((str(gamepad.axis(1)), 'value'), (car, 'throttle'), transform=lambda x: -x)
        
        # print("axis 0: {}, axis 1: {}, axis 2: {}, axis 3: {}, axis 4: {}, axis 5:{}, axis 6:{}"
        #       .format(gamepad.axis(0),gamepad.axis(1),gamepad.axis(2),gamepad.axis(3),
        #               gamepad.axis(4),gamepad.axis(5), gamepad.axis(6)))

        # Sleep for our polling interval
        time.sleep(pollInterval)
finally:
    # Ensure the background thread is always terminated when we are done
    gamepad.disconnect()
