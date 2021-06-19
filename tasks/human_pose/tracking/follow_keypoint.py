import cv2
import tracking.PID
import xarm

class KeypointFollow:
    def __init__(self):
        self.Arm = xarm.Controller('/dev/ttyTHS1', False)
        self.target_servox=90
        self.target_servoy=45
        self.xservo_pid = tracking.PID.PositionalPID(2.5, 0.2, 0.35)
        self.yservo_pid = tracking.PID.PositionalPID(3.5, 0.4, 0.5)

    def follow_function(self, x, y):


        point_x = x
        point_y = y

        if not (self.target_servox>=180 and point_x<=320 or self.target_servox<=0 and point_x>=320):
            self.xservo_pid.SystemOutput = point_x
            self.xservo_pid.SetStepSignal(320)

            self.xservo_pid.SetInertiaTime(0.005, 0.01)
            print("PID system output: ", self.xservo_pid.SystemOutput)

            target_valuex = int(1500 + self.xservo_pid.SystemOutput)
            print("target_valuex: ", target_valuex)
            self.target_servox = int((target_valuex - 500) / 10)
            print("target_servox: ", self.target_servox)

            if self.target_servox > 180: self.target_servox = 180
            if self.target_servox < 0: self.target_servox = 0
        if not (self.target_servoy>=90 and point_y<=240 or self.target_servoy<=0 and point_y>=240):

            self.yservo_pid.SystemOutput = point_y
            self.yservo_pid.SetStepSignal(240)


            self.yservo_pid.SetInertiaTime(0.005, 0.01)
            print("PID system output: ", self.yservo_pid.SystemOutput)

            target_valuey = int(1500 + self.yservo_pid.SystemOutput)
            print("target_valuey: ", target_valuey)

            self.target_servoy = int((target_valuey - 500) / 10) - 20
            print("target_servoy: ", self.target_servoy)

            if self.target_servoy > 90: self.target_servoy = 90
            if self.target_servoy < 0: self.target_servoy = 0

        joints_0 = [90, 90, self.target_servoy / 2, self.target_servoy / 2, 55, self.target_servox]
        for i in range(len(joints_0)):
            print("Servo id {}, angle: {}".format(i+1, float(joints_0[i])))
            self.Arm.setPosition(i+1, float(joints_0[i]), 800)
        return
