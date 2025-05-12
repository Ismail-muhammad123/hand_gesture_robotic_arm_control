import math
import time
import numpy as np
from pyfirmata import Arduino
# from your_servo_module import ServoMotor
from ik_solver import FiveDOF_IKSolver


class ServoMotor():
    def __init__(self, pin_number, name, home_angle,current_angle=0, max_angle=180, min_angle=0):
        self.name = name
        self.pin_number = pin_number
        self.home_angle = home_angle
        self.current_angle = current_angle
        self.max_angle = max_angle
        self.min_angle=min_angle
        
        self.pin = None

    def attach(self, board):
        servo_pin = board.get_pin(f'd:{self.pin_number}:s')
        self.pin = servo_pin

    
    def moveTo(self, target_angle, duration=1.0, steps=50):

        if self.pin is None:
            print(f"Servo {self.name} not attached to a board")

        try:
            current_read = self.pin.read()
            if current_read is not None:
                self.curren_angle = current_read
        except:
            pass

        current_angle = float(self.current_angle)
        target_angle = max(180, min(0, float(target_angle)))

        delta = target_angle - current_angle
        delay = duration / steps

        for i in range(steps + 1):
            t = i / steps  # 0 to 1
            s_curve = 3 * t**2 - 2 * t**3  # Smooth S-curve interpolation
            angle = current_angle + s_curve * delta
            self.pin.write(angle)
            self.curren_angle = angle
            time.sleep(delay)

        print(f"{self.name} smoothly moved to {target_angle}°")

    def setAngle(self, angle):
        angle = max(min(angle, self.max_angle), self.min_angle)
        if self.pin:
            self.pin.write(angle)
        self.current_angle = angle

    def goToHomePosition(self):
        self.moveTo(self.home_angle, 1, 50)


class RoboticArmDriver:
    def __init__(self, port, servos,  solver: FiveDOF_IKSolver):
        self.board = Arduino(port)
        time.sleep(2)
        self.servos = {}
        self.kinematics_solver = solver

        for pin_number, name, home_angle, current_angle, max_angle, min_angle in servos:
            servo = ServoMotor(pin_number, name, home_angle, current_angle, max_angle, min_angle)
            servo.attach(self.board)
            self.servos[name] = servo

    def move_joint(self, name, target_angle, duration=1.0, steps=50):
        if name in self.servos:
            servo = self.servos[name]
            servo.moveTo(target_angle, duration, steps)

    def goHome(self):
        for servo in self.servos.values():
            servo.goToHomePosition()


    def move_to_pose(self, position, euler_angles, duration=1.0, steps=100):
        angles = self.kinematics_solver.solve_ik(*position, euler_angles)
        angles = list(map(lambda r:round(math.degrees(r), 2) + 90,angles.tolist()))[:5]
        angles[1] = angles[1] - 45
        print(angles)
        joint_names = list(self.servos.keys())[:5]  # first 5 joints
        sleep_time = duration/steps
        for step in range(steps):
            for name, target in zip(joint_names, angles):
                current = self.servos[name].current_angle
                delta = (target - current) / (steps - step)
                self.servos[name].setAngle(current + delta)
            time.sleep(sleep_time)

    def get_pose(self):
        angles = [self.servos[name].current_angle for name in list(self.servos.keys())[:5]]
        return self.kinematics.forward_kinematics(angles)

    def open_gripper(self, width):
        servo = self.servos.get("gripper")
        if servo:
            angle = np.interp(width, [servo.min_angle, servo.max_angle], [servo.min_angle, servo.max_angle])
            # angle = np.clip(angle, servo.min_angle, servo.max_angle)
            servo.setAngle(angle)

    def cleanup(self):
        self.board.exit()
