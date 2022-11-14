from model import nvidia_model
import dataset

import socketio
import eventlet
import numpy as np
from flask import Flask
import torch
import base64
from io import BytesIO
from PIL import Image
import cv2


PATH = './checkpoint/ckpt.pth'
net = nvidia_model() 
sio = socketio.Server()
 
app = Flask(__name__) #'__main__'
speed_limit = 25
 
@sio.on('telemetry')
def telemetry(sio, data):
    print('Test: telementry')
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = dataset.img_train(image)
    steering_angle = net(image)
    # throttle = 1.0 - speed/speed_limit
    throttle = 1.0
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)
 
def send_control(steering_angle, throttle):
    print('Test: control')
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
 
if __name__ == '__main__':
    device = torch.device('cpu')
    net = nvidia_model()
    net.load_state_dict(torch.load(PATH, map_location=device))
    net.eval()
    print('Model loaded')
    
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
