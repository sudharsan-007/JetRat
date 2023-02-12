import Jetson.GPIO as GPIO
import time 

GPIO.setmode(GPIO.BOARD)

# add as many as channels as needed. You can also use tuples: (18,12,13)
channels = [32, 36, 38, 40]
GPIO.setup(channels, GPIO.OUT)

GPIO.output(channels, GPIO.HIGH)
time.sleep(2)
GPIO.output(channels, GPIO.LOW)

GPIO.cleanup()


