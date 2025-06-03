import cv2
import mediapipe as mp
import serial
import time

# Try different COM ports to find Arduino
possible_ports = ['COM4', 'COM5', 'COM3', 'COM6', 'COM7', 'COM8']
arduino = None

for port in possible_ports:
    try:
        print(f"Trying to connect to {port}...")
        arduino = serial.Serial(port, 9600)
        print(f"Successfully connected to Arduino on {port}")
        break
    except serial.SerialException:
        print(f"Could not connect to {port}")
        continue

if arduino is None:
    print("No Arduino found. Running in webcam-only mode...")
    
time.sleep(2)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

def count_fingers(landmarks):
    """Count the number of extended fingers"""
    
    # Finger tip and pip (proximal interphalangeal) landmark indices
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
    
    fingers_up = 0
    
    # Thumb (special case - check x-coordinate)
    if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
        fingers_up += 1
    
    # Other fingers (check y-coordinate)
    for i in range(1, 5):
        if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
            fingers_up += 1
    
    return fingers_up

def send_finger_count(count):
    """Send finger count to Arduino"""
    command = str(count)
    if arduino:
        arduino.write(command.encode())
        print(f"Sent finger count to Arduino: {count}")
    else:
        print(f"SIMULATION - Would send finger count: {count}")

# Variables for stabilizing detection
last_finger_count = -1
count_stable_frames = 0
required_stable_frames = 5  # Number of consistent frames before sending command

print("Finger Control Robot Hand Started!")
print("Show your hand to the camera and raise different numbers of fingers")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(rgb_frame)
    
    finger_count = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Count fingers
            finger_count = count_fingers(hand_landmarks.landmark)
            
            # Draw finger count on screen
            cv2.putText(frame, f'Fingers: {finger_count}', 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Stability check - only send command if count is stable
            if finger_count == last_finger_count:
                count_stable_frames += 1
                if count_stable_frames >= required_stable_frames:
                    send_finger_count(finger_count)
                    count_stable_frames = 0  # Reset counter after sending
            else:
                last_finger_count = finger_count
                count_stable_frames = 0
    else:
        # No hand detected
        cv2.putText(frame, 'No hand detected', 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Add instructions to the frame
    cv2.putText(frame, 'Press Q to quit', 
               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the frame
    cv2.imshow('Finger Control Robot Hand', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
    print("Arduino connection closed.") 