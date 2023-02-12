import Gamepad
import time

# Gamepad settings
gamepadType = Gamepad.example
pollInterval = 0.1

class RaceStick():
    def __init__(self, gamepadType=Gamepad.example, pollInterval=0.05):
        self.gamepadType = gamepadType
        self.pollInterval = pollInterval
        self.gamepad = gamepadType()
        self.state = "init"
        self.throttle = 0.0
        self.steering = 0.0
        self.stop = False
        self.steeringOffset = 0.0
        self.cons_throttle = False
        
        self.collectData = False # set this to --> R1 for True; L1 for False
        self.remoteControl = False # 
        self. training = False
        self.autonomous = False
        
        self.forceStop = False

        # Wait for a connection
        if not Gamepad.available():
            print('Please connect your gamepad...')
            while not Gamepad.available():
                time.sleep(1.0)
                
        
        print('Gamepad connected')

        # Start the background updating
        self.gamepad.startBackgroundUpdates()
        print("bg update started")

    def PrintState(self):
        try:
            while self.gamepad.isConnected():
                
                time.sleep(pollInterval) # Sleep for our polling interval
                # Check for the exit button
                if self.gamepad.beenPressed(4):
                    print('EXIT')
                    break
                print("axis 0: {}, axis 1: {}, axis 2: {}, axis 3: {}, axis 4: {}, axis 5:{}"
                      .format(self.gamepad.axis(0),self.gamepad.axis(1),self.gamepad.axis(2),self.gamepad.axis(3), self.gamepad.axis(4),self.gamepad.axis(5)))
                
                # time.sleep(pollInterval) # Sleep for our polling interval
        finally:
            # Ensure the background thread is always terminated when we are done
            self.gamepad.disconnect()
    
    def GetDriveState(self):
        try:
            if self.gamepad.beenPressed(0):
                self.training = False
                
            if self.gamepad.beenPressed(1) and self.autonomous == False:
                self.training = True
                
            if self.gamepad.beenPressed(2):
                self.autonomous = False
                
            if self.gamepad.beenPressed(3) and self.training == False and self.remoteControl == False:
                self.autonomous = True
                
            if self.gamepad.beenPressed(4):
                self.remoteControl = False
                
            if self.gamepad.beenPressed(5):
                self.remoteControl = True
                
            if self.gamepad.beenPressed(6):
                self.collectData = False
                
            if self.gamepad.beenPressed(7):
                self.collectData = True
            
            if self.gamepad.beenPressed(8):
                self.forceStop = True
            
        except:
            print("the controller may be disconnected")
        return self.remoteControl, self.collectData, self.training, self.autonomous, self.forceStop
    
    def GetThrottleSteering(self):
        try:
            self.throttle, self.steering = -self.gamepad.axis(3), -self.gamepad.axis(0)
            self.steeringOffset += self.gamepad.axis(4) * 0.02
        except:
            print("the controller may be disconnected")
            return 0.0, 0.0, 0.0
        
        return self.throttle, self.steering, self.steeringOffset
    
    def GetConsThrottle(self):
        try:
            # print("controller input for 5 is {}".format(self.gamepad.axis(5)))
            if self.gamepad.axis(5) == -1:
                self.cons_throttle = True
            if self.gamepad.axis(5) == 1:
                self.cons_throttle = False
        except:
            print("the controller maybe Disconnected")
            self.cons_throttle = False
            return False
        return self.cons_throttle
            
        

                
if __name__ == "__main__":
    joystick = RaceStick(gamepadType,pollInterval)
    
    while True:
        time.sleep(0.08)
        # joystick.PrintState()
        # time.sleep(pollInterval) # Sleep for our polling interval
        print(joystick.GetConsThrottle())
        # throttle, steering, steeringOffset = joystick.GetThrottleSteering()
        # speed = joystick.GetConsThrottle()
        
        
        # print("throttle: {}, steering: {}, offset {}, speed {}".format(throttle, steering, steeringOffset, speed))
    