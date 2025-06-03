import cv2
from fer import FER
import serial
import time
import numpy as np
import math

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
detector = FER(mtcnn=True)

def send_gesture(command, emotion):
    if arduino:
        arduino.write(command.encode())
        print(f"Sent command to Arduino: {command} (for {emotion})")
    else:
        print(f"SIMULATION - Would send command: {command} (for {emotion})")

def get_emotion_color(emotion):
    """Get color for emotion display"""
    colors = {
        'happy': (0, 255, 0),       
        'sad': (255, 0, 0),         
        'angry': (0, 0, 255),      
        'surprise': (255, 255, 0), 
        'fear': (0, 165, 255),     
        'disgust': (128, 0, 128), 
        'neutral': (255, 255, 255) 
    }
    return colors.get(emotion.lower(), (255, 255, 255))

emotion_history = []
history_length = 8  
confidence_threshold = 0.25  
last_sent_emotion = None
frame_count = 0

print("🎭 IMPROVED Face Control Robot Hand Started!")
print("Enhanced features:")
print("  • Lower sensitivity threshold (0.25) for subtle expressions")
print("  • Emotion smoothing over 8 frames")
print("  • Detailed emotion scores display")
print("  • Better lighting tolerance")
print("")
print("😊 Happy = Thumbs up")
print("😢 Sad = Close hand") 
print("😠 Angry = Close hand")
print("😲 Surprise = Open hand")
print("💡 TIP: Try making more exaggerated expressions at first!")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_frame = cv2.merge([l, a, b])
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)

    results = detector.detect_emotions(frame)
    results_enhanced = detector.detect_emotions(enhanced_frame)
    
    detected_emotion = "No face detected"
    confidence_score = 0.0
    all_emotions = {}
    
    if results or results_enhanced:
        combined_emotions = {}
        
        if results:
            for emotion, score in results[0]['emotions'].items():
                combined_emotions[emotion] = combined_emotions.get(emotion, []) + [score]
        
        if results_enhanced:
            for emotion, score in results_enhanced[0]['emotions'].items():
                combined_emotions[emotion] = combined_emotions.get(emotion, []) + [score]
        
        if combined_emotions:
            all_emotions = {emotion: np.mean(scores) for emotion, scores in combined_emotions.items()}
            detected_emotion = max(all_emotions, key=all_emotions.get)
            confidence_score = all_emotions[detected_emotion]
            
            emotion_history.append((detected_emotion, confidence_score))
            if len(emotion_history) > history_length:
                emotion_history.pop(0)
            
            if len(emotion_history) >= 3:
                recent_strong_emotions = [(e, c) for e, c in emotion_history if c > confidence_threshold]
                
                if recent_strong_emotions:
                    emotion_counts = {}
                    for emotion, conf in recent_strong_emotions:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    most_frequent = max(emotion_counts, key=emotion_counts.get)
                    if emotion_counts[most_frequent] >= 2:
                        detected_emotion = most_frequent
                        confidence_score = max([c for e, c in recent_strong_emotions if e == most_frequent])

    if detected_emotion != "No face detected" and confidence_score > confidence_threshold:
        if detected_emotion != last_sent_emotion and frame_count % 10 == 0:
            if detected_emotion == "happy":
                send_gesture('t', detected_emotion) 
                last_sent_emotion = detected_emotion
            elif detected_emotion in ["angry", "sad"]:
                send_gesture('c', detected_emotion) 
                last_sent_emotion = detected_emotion
            elif detected_emotion in ["surprise", "fear"]:
                send_gesture('o', detected_emotion) 
                last_sent_emotion = detected_emotion

    emotion_color = get_emotion_color(detected_emotion)
    
    emotion_text = f'Emotion: {detected_emotion.title()}'
    confidence_text = f'Confidence: {confidence_score:.2f}'
    
    cv2.putText(frame, emotion_text, 
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)
    cv2.putText(frame, confidence_text, 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    threshold_text = f'Threshold: {confidence_threshold} {"✓" if confidence_score > confidence_threshold else "✗"}'
    cv2.putText(frame, threshold_text, 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if confidence_score > confidence_threshold else (0, 0, 255), 1)
    
    if all_emotions:
        y_offset = 150
        cv2.putText(frame, 'All Emotions (avg):', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:4]
        for emotion, score in sorted_emotions:
            y_offset += 25
            color = get_emotion_color(emotion)
            thickness = 2 if score > confidence_threshold else 1
            cv2.putText(frame, f'{emotion}: {score:.3f}', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)
    
    action_y = frame.shape[0] - 120
    if detected_emotion.lower() == "happy" and confidence_score > confidence_threshold:
        cv2.putText(frame, '😊 -> Thumbs Up!', 
                   (10, action_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif detected_emotion.lower() in ["angry", "sad"] and confidence_score > confidence_threshold:
        cv2.putText(frame, f'😢😠 -> Close Hand!', 
                   (10, action_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif detected_emotion.lower() in ["surprise", "fear"] and confidence_score > confidence_threshold:
        cv2.putText(frame, f'😲 -> Open Hand!', 
                   (10, action_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(frame, 'Make a stronger expression...', 
                   (10, action_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    cv2.putText(frame, 'History:', (frame.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for i, (emotion, conf) in enumerate(emotion_history[-8:]):
        color = get_emotion_color(emotion)
        x = frame.shape[1] - 140 + i * 15
        y = 35
        radius = 6 if conf > confidence_threshold else 3
        cv2.circle(frame, (x, y), radius, color, -1)
    
    cv2.putText(frame, 'Press Q to quit | Try exaggerated expressions!', 
               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Improved Face Control Robot Hand", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
    print("Arduino connection closed.")
