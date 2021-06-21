import xarm

arm = xarm.Controller('/dev/ttyTHS1', True)

servo1 = xarm.Servo(1)
servo2 = xarm.Servo(2)
servo3 = xarm.Servo(3)
servo4 = xarm.Servo(4)
servo5 = xarm.Servo(5)
servo6 = xarm.Servo(6)

arm.setPosition(1, 90.0)
arm.setPosition(2, 90.0)
arm.setPosition(3, 90.0)
arm.setPosition(4, 0.0)
arm.setPosition(5, 55.0)
arm.setPosition(6, 180.0)


# arm.setPosition(1, 1500)
# arm.setPosition(2, 1500)
# arm.setPosition(3, 1500)
# arm.setPosition(4, 1500)
# arm.setPosition(5, 1000)
# arm.setPosition(6, 1500)


# for i in range(0,6):
#     arm.setPosition(i+1, 1700)
#     print(i)
# battery_voltage = arm.getBatteryVoltage()
# pos1 = arm.getPosition(6)

# print('Battery voltage (volts):', battery_voltage)
# print('pos1:', pos1)

# arm.setPosition([servo1, servo2, servo3, servo4, servo5, servo6]) 
