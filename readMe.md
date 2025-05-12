# Gesture-Controlled Robotic Arm with ESP32 and Raspberry Pi

This project allows you to control a 5DOF robotic arm using hand gestures captured from a camera. It uses MediaPipe for real-time hand tracking on a PC and sends commands via USB cable to an Arduino Uno controlling the robotic arm.

## 📦 Features

- Real-time gesture tracking using a camera.
- Serial communication between PC and Arduino Uno (using firmata).
- 5DOF inverse kinematics with optional end-effector orientation.
- Predefined routines via buttons (e.g., rest, start, reset).
- Python + Arduino implementation with IKPy library.

---

## 🧰 Hardware Requirements

- PC with webcam
- Arduino Uno
- 5DOF robotic arm (servo-driven)
- 12V power adapter
- 6v Buck Converter for servos

---

## 🧪 Software Requirements

### Raspberry Pi (Python side)
- Python 3.10+
- MediaPipe
- OpenCV
- NumPy
- SciPy
- IKPy
- PySerial (for debugging)
- pyFirmata

Install dependencies with:

```bash
pip install mediapipe opencv-python numpy scipy ikpy pyfirmata
