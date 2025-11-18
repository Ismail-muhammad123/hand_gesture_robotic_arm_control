import argparse
import time
import sys
from pyfirmata import Arduino, util

#!/usr/bin/env python3
"""
/home/hawkeye/Desktop/projects/FInal Year Project/python/test.py

Control a servo on an Arduino running StandardFirmata using pyfirmata.
You can pass a serial port (e.g. /dev/ttyACM0, COM3) or a socket URL
(e.g. socket://192.168.1.100:3030) if a TCP server is bridging the Arduino.
"""


# try:
# except Exception as e:
#     sys.exit("pyfirmata is required. Install with: pip install pyfirmata\nError: " + str(e))

SERVO_CONFIG = {
    "gripper": {"pin": 9, "min_angle": 0, "max_angle": 70},
    "elbow": {"pin": 6, "min_angle": 50, "max_angle": 160},
    "shoulder": {"pin": 5, "min_angle": 40, "max_angle": 180},
    "base": {"pin": 3, "min_angle": 45, "max_angle": 225},
}






def run(port, angle, sweep, delay, joint="elbow"):

    pin = SERVO_CONFIG[joint]["pin"]
    min_angle = SERVO_CONFIG[joint]["min_angle"]
    max_angle = SERVO_CONFIG[joint]["max_angle"]

    board = None
    
    try:
        board = Arduino(port)
    except Exception as e:
        sys.exit(f"Failed to open port {port}: {e}")

    # Start iterator (recommended for pyfirmata to avoid overflow on some Firmata firmwares)
    it = util.Iterator(board)
    it.start()

    # Configure servo pin (digital pin, servo mode 's')
    servo = board.get_pin(f'd:{pin}:s')

    # Allow board to settle
    time.sleep(0.5)

    try:
        if sweep:
            # Sweep from 0 to 180 and back until interrupted
            while True:
                for a in range(min_angle, max_angle, 5):
                    servo.write(a)
                    print(f"ANGLE: {a} {''.join(['-'] * int(a/5))}")
                    time.sleep(delay)
                for a in range(max_angle, min_angle-1, -5):
                    servo.write(a)
                    print(f"ANGLE: {a} {''.join(['-'] * int(a/5))}")
                    time.sleep(delay)
        else:
            # Move to the target angle in steps
            current_angle = 0  # Start from 0 degrees
            target_angle = max(min_angle, min(max_angle, angle))
            step = 5  # Define the step size
            direction = 1 if target_angle > current_angle else -1
            
            while current_angle != target_angle:
                current_angle += step * direction
                current_angle = max(min_angle, min(max_angle, current_angle))
                servo.write(current_angle)
                print(f"ANGLE: {current_angle} {''.join(['-'] * int(current_angle/5))}")
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            board.exit()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Control a servo via pyfirmata/Firmata server")
    parser.add_argument("--port", "-p", required=True,
                        help="Serial port or socket URL. e.g. /dev/ttyACM0 or socket://192.168.1.100:3030")
    parser.add_argument("--joint", "-j", type=str, default="gripper", help="The Joint or the robot (default: gripper)")
    parser.add_argument("--angle", "-a", type=int, default=90, help="Angle to move servo to (min-max)")
    parser.add_argument("--sweep", "-s", action="store_true", help="Continuously sweep servo min-max-min")
    parser.add_argument("--delay", "-d", type=float, default=0.02, help="Delay between steps during sweep (seconds)")
    args = parser.parse_args()

    run(args.port, args.angle, args.sweep, args.delay, args.joint)


if __name__ == "__main__":
    main()