import queue
import threading
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ik_solver import FiveDOF_IKSolver
from robotics.robotic import RoboticArmDriver

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sampling_interval = 0.01
scaling_factor = 2
threashold_value = 5

# Constants for Gripper Control
MIN_GRIPPER_DISTANCE = 0.0  # Minimum gripper opening (cm)
MAX_GRIPPER_DISTANCE = 10.0  # Maximum gripper opening (cm)
TRACK_TWO_HANDS = False # True if two hand will be used
SHOW_SIMULATION=True

REACH_RANGE = [
    [0,50],
    [0,50],
    [0,50],
]


current_position = [0.02,0.02,0.02]
latest_angles = [0, 0, 0, 0, 0, 0]  
# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7, 
    max_num_hands=2 if TRACK_TWO_HANDS else 1,
    )
mp_draw = mp.solutions.drawing_utils

# Define Camera Feed
cap = cv2.VideoCapture(0)

# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)

def interpolate(z, z_min=0, z_max=5, new_min=0, new_max=10):
    z_clipped = max(min(z, z_max), z_min)  # Clamp z to avoid overflow
    return (z_clipped - z_min) / (z_max - z_min) * (new_max - new_min) + new_min

# Function to generate arm position command (x, y, z)
def generate_arm_position(hand_landmarks):
    # Extract landmarks of the hand (index finger tip, wrist, etc.)
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # The position of wrist gives us x, y, z coordinates
    x,y,z = wrist.x, wrist.y, wrist.z
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    z= euclidean_distance((wrist.x, wrist.y, wrist.z), (index_base.x, index_base.y,  index_base.z))
   
    
    x = x * 10  # Convert to cm or appropriate unit
    y = y * 10
    z = z * 10 

    y = 10-y # revses to make origin at the bottom


    # convert to two decimal places
    x = interpolate(x, z_min=0, z_max=10, new_min=-5, new_max=5)

    x = round(x, 1)
    y = round(y, 1)
    z = round(z, 1) 
    # z = round(interpolate(z, new_max=10), 1)

    # Swap x, y, z axes
    x, y, z = z, x, y

    print("-----------")
    print("POS X: ", x)
    print("POS Y: ", y)
    print("POS Z: ", z)
    print("-----------")
    return x, y, z

def midpoint(x1, y1, z1, x2, y2, z2):
    """
    Calculates the midpoint of a line segment in 3D space.

    Args:
        x1: X-coordinate of the first point.
        y1: Y-coordinate of the first point.
        z1: Z-coordinate of the first point.
        x2: X-coordinate of the second point.
        y2: Y-coordinate of the second point.
        z2: Z-coordinate of the second point.

    Returns:
        A tuple representing the midpoint (x, y, z).
    """
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    z_mid = (z1 + z2) / 2
    return (x_mid, y_mid, z_mid)

# Function to generate gripper command (distance based on thumb and index finger distance)
def generate_gripper_command(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Calculate Euclidean distance between thumb and index tip
    distance = euclidean_distance([thumb_tip.x, thumb_tip.y, thumb_tip.z],
                                  [index_tip.x, index_tip.y, index_tip.z])
    
    # Map this distance to a range of gripper opening (in cm)
    gripper_distance = np.interp(distance, [0.01, 0.2], [MIN_GRIPPER_DISTANCE, MAX_GRIPPER_DISTANCE])
    
    # Return gripper distance command
    return gripper_distance

# Function to check if only the thumb and index finger are open
def is_only_thumb_and_index_open(hand_landmarks):
    # Get the wrist and finger tip positions
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Calculate distances from wrist to the tips of the fingers
    middle_wrist_dist = euclidean_distance([wrist.x, wrist.y, wrist.z], [middle_tip.x, middle_tip.y, middle_tip.z])
    ring_wrist_dist = euclidean_distance([wrist.x, wrist.y, wrist.z], [ring_tip.x, ring_tip.y, ring_tip.z])
    pinky_wrist_dist = euclidean_distance([wrist.x, wrist.y, wrist.z], [pinky_tip.x, pinky_tip.y, pinky_tip.z])

    # Define distance threshold to consider finger as closed
    distance_threshold = euclidean_distance([wrist.x, wrist.y, wrist.z], [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z])  # Adjust this threshold based on your test

    # Check if thumb and index are open, and the other fingers are closed
    # if thumb_wrist_dist > distance_threshold and index_wrist_dist > distance_threshold and 
    if middle_wrist_dist < distance_threshold and ring_wrist_dist < distance_threshold and pinky_wrist_dist < distance_threshold:
        return True  # Only thumb and index are open
    return False


# ========================================================================
# ROBOTIC ARM CONTROL
SERVO_CONFIG = {
    "gripper": {"pin": 9, "min_angle": 0, "max_angle": 70},
    "elbow": {"pin": 6, "min_angle": 50, "max_angle": 160},
    "shoulder": {"pin": 5, "min_angle": 40, "max_angle": 180},
    "base": {"pin": 3, "min_angle": 45, "max_angle": 225},
}


# [pin_number, name, home_angle, current_angle, max_angle, min_angle]
servos_config = [(i[1]["pin"], i[0], (i[1]['max_angle'] -i[1]['min_angle'])/2, (i[1]['max_angle'] -i[1]['min_angle'])/2, i[1]['max_angle'], i[1]['min_angle']) for i in SERVO_CONFIG.items()] 
   

ik_solver =  FiveDOF_IKSolver()

# Initialize the arm
arm = RoboticArmDriver('/dev/ttyACM0', servos_config, ik_solver)

# Move servo to home position
arm.goHome()

# --- Command Queue ---
command_queue = queue.Queue()


# --- Command Worker ---
def command_worker():
    global current_position
    global latest_angles
    while True:
        position, orientation, gripper_distance = command_queue.get()  # blocks until item available
        try:
            # if any([abs(i) >= threashold_value for i in position_delta]):
            #     print("Position delta: ", position_delta)
            current_position = [round(position[i]/10,2) for i in range(len(position))] 
            # current_position[2]= -1 * current_position[2]
            current_position[1] = -1 * current_position[1]

            print("Position: ",    current_position)
            print("Orientation: ", orientation)
            print("gripper: ", int(gripper_distance))
            print("=======================================================")
            arm.move_to_pose(current_position, (0,0,0), 0, 1)
            arm.open_gripper(gripper_distance)

            # joint_angles = ik_solver.solve_ik(*current_position, orientation_euler=orientation)
            # print("Joint angles (degrees):", list(map(lambda r:round(math.degrees(r), 2),joint_angles.tolist()))[:5] )

        except Exception as e:
            print(f"Error moving to {position} and gripper distance {gripper_distance}: \n{e}")
        command_queue.task_done()  # marks command as completed


# init and start thread
worker_thread = threading.Thread(target=command_worker, daemon=True)
worker_thread.start()



# ==================================================================================
# Main Loop for Real-Time Hand Tracking in OpenCV

last_detected = None
last_position = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for better visualization
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dimensions of the frame
    height, width, _ = frame.shape

    # Draw the scale on the frame
    num_divisions = 10  # Number of divisions for the grid
    step_x = width // num_divisions
    step_y = height // num_divisions

    for i in range(num_divisions + 1):
        # Calculate the center of the frame
        center_x = width // 2
        center_y = height // 2

        # Draw vertical grid lines with center as zero
        cv2.line(frame, (center_x + (i - num_divisions // 2) * step_x, 0), (center_x + (i - num_divisions // 2) * step_x, height), color=(200, 200, 200), thickness=1)
        # Draw horizontal grid lines with center as zero
        cv2.line(frame, (0, center_y - (i - num_divisions // 2) * step_y), (width, center_y - (i - num_divisions // 2) * step_y), color=(200, 200, 200), thickness=1)
        # Add labels for X axis (right is positive)
        # Draw X axis labels at the bottom of the frame
        cv2.putText(
            frame,
            str((i - num_divisions // 2)),
            (center_x + (i - num_divisions // 2) * step_x + 5, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
        cv2.putText(frame, str(10-i), (5, i * step_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    
    # Process the frame and get the hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the thumb and index finger tip coordinates
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            

            thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_tip_coords = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
            

            # Check if only the thumb and index fingers are open
            if is_only_thumb_and_index_open(landmarks):
                box_color = (0, 255, 0)
                 

                # Generate arm position and gripper command
                arm_position = generate_arm_position(landmarks)
                gripper_distance = generate_gripper_command(landmarks)

                    


                # Draw a blue line connecting the thumb and index finger tips
                cv2.line(frame, thumb_tip_coords, index_tip_coords, (255, 0, 0), 2)


                command_queue.put((arm_position,(0,0,0), gripper_distance))

                # =================================================================


                # Display message that the gesture is being followed
                cv2.putText(frame, "Gesture Followed: Thumb & Index Open", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Display calsulated coordinates and gripper values
                cv2.putText(frame, f"x:{arm_position[0]:.2f}, y:{arm_position[1]:.2f}, z:{arm_position[2]}, gripper:{gripper_distance:.2f}", (10, 65), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            else:
                box_color = (0, 0, 255)

                # Display message that the gesture is not being followed
                cv2.putText(frame, "Gesture Not tracking,", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "Clinch all but the thumb and index finger to start following", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
            # Draw the bounding box
            cv2.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), box_color, 2)
    else:    
        # Display message that the gesture is not being followed
        cv2.putText(frame, "No hand detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Show the frame with hand landmarks and control values
    cv2.imshow("Hand Tracking", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
# arm.cleanUp()
cap.release()
cv2.destroyAllWindows()
