import cv2
import serial
import time
import numpy as np

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

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def send_gesture(command, emotion):
    if arduino:
        arduino.write(command.encode())
        print(f"Sent command to Arduino: {command} (for {emotion})")
    else:
        print(f"SIMULATION - Would send command: {command} (for {emotion})")

last_sent_emotion = None
frame_count = 0
detection_history = []

print("🤖 Simple Face Control Robot Hand Started!")
print("This version uses OpenCV's built-in detection:")
print("😊 Smile detection → Thumbs up")
print("😐 No smile → Close hand")
print("👋 Wave your hand in front of face → Open hand")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    detected_emotion = "No face detected"
    confidence = 0.0
    
    if len(faces) > 0:
        face = max(faces, key=lambda x: x[2] * x[3])
        (x, y, w, h) = face
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        
        if len(smiles) > 0:
            detected_emotion = "happy"
            confidence = min(len(smiles) * 0.3, 1.0)
            
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
        else:
            detected_emotion = "neutral"
            confidence = 0.8
        
        if frame_count > 10:
            face_center = (x + w//2, y + h//2)
            
            if len(detection_history) > 0:
                last_center = detection_history[-1]
                movement = abs(face_center[0] - last_center[0]) + abs(face_center[1] - last_center[1])
                
                if movement > 50:
                    detected_emotion = "surprise"
                    confidence = min(movement / 100.0, 1.0)
            
            detection_history.append(face_center)
            if len(detection_history) > 5:
                detection_history.pop(0)
    
    if detected_emotion != "No face detected" and confidence > 0.5:
        if detected_emotion != last_sent_emotion and frame_count % 15 == 0:
            if detected_emotion == "happy":
                send_gesture('t', "smile")
                last_sent_emotion = detected_emotion
            elif detected_emotion == "neutral":
                send_gesture('c', "neutral")
                last_sent_emotion = detected_emotion
            elif detected_emotion == "surprise":
                send_gesture('o', "movement")
                last_sent_emotion = detected_emotion
    
    if detected_emotion == "happy":
        emotion_color = (0, 255, 0)
    elif detected_emotion == "surprise":
        emotion_color = (255, 0, 0)
    else:
        emotion_color = (255, 255, 255)
    
    emotion_text = f'Detection: {detected_emotion.title()}'
    confidence_text = f'Confidence: {confidence:.2f}'
    
    cv2.putText(frame, emotion_text, 
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)
    cv2.putText(frame, confidence_text, 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.putText(frame, 'Smile for thumbs up!', 
               (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, 'Wave for open hand!', 
               (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, 'Neutral face for close!', 
               (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    action_y = frame.shape[0] - 60
    if detected_emotion == "happy" and confidence > 0.5:
        cv2.putText(frame, '😊 -> Thumbs Up!', 
                   (10, action_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif detected_emotion == "surprise" and confidence > 0.5:
        cv2.putText(frame, '👋 -> Open Hand!', 
                   (10, action_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    elif detected_emotion == "neutral" and confidence > 0.5:
        cv2.putText(frame, '😐 -> Close Hand!', 
                   (10, action_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(frame, 'Press Q to quit', 
               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Simple Face Control Robot Hand", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
    print("Arduino connection closed.") 