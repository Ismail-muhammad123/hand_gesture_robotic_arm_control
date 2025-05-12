import queue
import threading
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ik_solver import FiveDOF_IKSolver
from robotics.robotic import RoboticArmDriver

sampling_interval = 0.01
scaling_factor = 2
threashold_value = 5

# Constants for Gripper Control
MIN_GRIPPER_DISTANCE = 0.0  # Minimum gripper opening (cm)
MAX_GRIPPER_DISTANCE = 10.0  # Maximum gripper opening (cm)
TRACK_TWO_HANDS = False # True if two hand will be used
SHOW_SIMULATION=True


REACH_RANGE=[
    [0,50],
    [0,50],
    [0,50],
]

current_position = [20,20,20]

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

# Function to generate arm position command (x, y, z)
def generate_arm_position(hand_landmarks):
    # Extract landmarks of the hand (index finger tip, wrist, etc.)
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    # thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    # index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    
    # The position of wrist gives us x, y, z coordinates
    # x,y,z = midpoint(thumb_tip.x, thumb_tip.y, thumb_tip.z, index_tip.x, index_tip.y, index_tip.z)
    x,y,z = wrist.x, wrist.y, wrist.z
    x = x * 10  # Convert to cm or appropriate unit
    y = y * 10
    z = abs(z * 10**7)  # Depth/Distance from camera


    # convert to two decimal places
    x = round(x, 0)
    y = round(y, 0) 
    z = round(z, 0)


    # Subtract values from 100
    # x = REACH_RANGE[0][1] - x
    # y = REACH_RANGE[1][1] - y
    # z = REACH_RANGE[2][1] - z
    y = 10-y
    z=10-z

    # Swap x, y, z axes
    x, y, z = z, x, y
    # print("pos x: ", x)
    # print("pos y: ", y)
    # print("pos z: ", z)

    # Return arm position (x, y, z)
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
servos_config = [
    # [pin_number, name, home_angle, current_angle, max_angle, min_angle]
    (3, "base", 90, 90, 180, 0),
    (5, "shoulder", 120, 90, 180, 0),
    (6, "elbow", 170, 90, 180, 0),
    # (9, "wrist_pitch", 140, 45, 180, 0),
    # (10, "wrist_roll", 90, 90, 180, 0),
    # (11, "gripper", 50, 50, 50, 0),
]

ik_solver =  FiveDOF_IKSolver()

# Initialize the arm
# arm = RoboticArmDriver('/dev/ttyACM0', servos_config)

# # Move servo to home position
# arm.move_to_pose(current_position, (0,0,0), 1, 10)


# --- Command Queue ---
command_queue = queue.Queue()


# --- Command Worker ---
def command_worker():
    global current_position
    while True:
        position, orientation, gripper_distance = command_queue.get()  # blocks until item available
        try:
            # if any([abs(i) >= threashold_value for i in position_delta]):
            #     print("Position delta: ", position_delta)
            current_position = [position[i]/10 * 50 for i in range(len(position))] 
            # else:
            #     print("Position delta: ", [0,0,0])

            # arm.move_to_pose(position=current_position, euler_angles=(0,0,0), duration=0, steps=1)
            # arm.open_gripper(gripper_distance)
            
            # if SHOW_SIMULATION:
            #     ik_solver.move_chain_to_target(current_position)
            

            print("Position: ",  [round(i, 2) for i in  current_position])
            print("Orientation: ", orientation)
            print("gripper: ", int(gripper_distance))
            print("=======================================================")
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
        # Draw vertical grid lines
        cv2.line(frame, (i * step_x, 0), (i * step_x, height), color=(200, 200, 200), thickness=1)
        # Draw horizontal grid lines
        cv2.line(frame, (0, i * step_y), (width, i * step_y), color=(200, 200, 200), thickness=1)
        # Add labels for X and Y axes
        cv2.putText(frame, str(i*10), (i * step_x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, str(i*10), (5, i * step_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    
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


                # Send data to Arduino/ESP32 via serial (format: x,y,z,gripper_distance)
                # command = f"{arm_position[0]:.2f},{arm_position[1]:.2f},{arm_position[2]},{gripper_distance:.2f}\n"
                # print("COMMAND: ", command)

                # ============== Send command to the robotic arm ==================
                # now = time.time()
                # if last_detected is None or now - last_detected >= sampling_interval:
                    
                # position_difference = [0,0,0]
                
                # if last_position is None:
                #     position_difference = [0,0,0]
                # else:
                #     position_difference = [round(arm_position[i] - last_position[i], 2) for i in range(len(arm_position))]


                command_queue.put((arm_position,(0,0,0), gripper_distance))
                
                    # last_detected = now
                # last_position = arm_position

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
cap.release()
cv2.destroyAllWindows()
