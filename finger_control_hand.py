import cv2
import mediapipe as mp
import serial
import time

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

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

def count_fingers(landmarks):
    """Count the number of extended fingers"""
    
    finger_tips = [4, 8, 12, 16, 20]  
    finger_pips = [3, 6, 10, 14, 18]  
    
    fingers_up = 0
    
    if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
        fingers_up += 1
    
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

last_finger_count = -1
count_stable_frames = 0
required_stable_frames = 5  

print("Finger Control Robot Hand Started!")
print("Show your hand to the camera and raise different numbers of fingers")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    finger_count = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            finger_count = count_fingers(hand_landmarks.landmark)
            
            cv2.putText(frame, f'Fingers: {finger_count}', 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if finger_count == last_finger_count:
                count_stable_frames += 1
                if count_stable_frames >= required_stable_frames:
                    send_finger_count(finger_count)
                    count_stable_frames = 0  
            else:
                last_finger_count = finger_count
                count_stable_frames = 0
    else:
       
        cv2.putText(frame, 'No hand detected', 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.putText(frame, 'Press Q to quit', 
               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Finger Control Robot Hand', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
    print("Arduino connection closed.") 