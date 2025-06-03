import cv2
from fer import FER
import serial
import time
import numpy as np

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

# Initialize webcam and FER detector with improved settings
cap = cv2.VideoCapture(0)
# Try different FER configurations for better detection
detector = FER(mtcnn=True)  # Using MTCNN for better face detection

def preprocess_face_region(frame):
    """Enhance the frame for better emotion detection"""
    # Convert to grayscale and back to RGB for better contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better lighting
    enhanced = cv2.equalizeHist(gray)
    
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Slight gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_bgr, (3, 3), 0)
    
    return blurred

def send_gesture(command, emotion):
    if arduino:
        arduino.write(command.encode())
        print(f"Sent command to Arduino: {command} (for {emotion})")
    else:
        print(f"SIMULATION - Would send command: {command} (for {emotion})")

def get_emotion_color(emotion):
    """Get color for emotion display"""
    colors = {
        'happy': (0, 255, 0),      # Green
        'sad': (255, 0, 0),        # Blue  
        'angry': (0, 0, 255),      # Red
        'surprise': (255, 255, 0), # Cyan
        'fear': (0, 165, 255),     # Orange
        'disgust': (128, 0, 128),  # Purple
        'neutral': (255, 255, 255) # White
    }
    return colors.get(emotion.lower(), (255, 255, 255))

# Emotion stability tracking
emotion_history = []
history_length = 5  # Number of frames to consider
confidence_threshold = 0.3  # Lower threshold for better detection
last_sent_emotion = None

print("Improved Face Control Robot Hand Started!")
print("🎭 Enhanced emotion detection with:")
print("  • Better preprocessing for lighting")
print("  • Lower confidence thresholds")
print("  • Emotion smoothing over multiple frames")
print("😊 Happy = Thumbs up")
print("😢 Sad = Close hand") 
print("😠 Angry = Close hand")
print("😲 Surprise = Open hand")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    
    # Create enhanced version for detection
    enhanced_frame = preprocess_face_region(frame)

    # Detect emotions on both original and enhanced frames
    results_original = detector.detect_emotions(frame)
    results_enhanced = detector.detect_emotions(enhanced_frame)
    
    detected_emotion = "No face detected"
    confidence_score = 0.0
    all_emotions = {}
    
    # Combine results from both detection attempts
    all_results = []
    if results_original:
        all_results.extend(results_original)
    if results_enhanced:
        all_results.extend(results_enhanced)
    
    if all_results:
        # Get all emotion scores and average them
        emotion_scores = {}
        for result in all_results:
            for emotion, score in result['emotions'].items():
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = []
                emotion_scores[emotion].append(score)
        
        # Average the scores
        averaged_emotions = {}
        for emotion, scores in emotion_scores.items():
            averaged_emotions[emotion] = np.mean(scores)
        
        # Find the emotion with highest average score
        if averaged_emotions:
            detected_emotion = max(averaged_emotions, key=averaged_emotions.get)
            confidence_score = averaged_emotions[detected_emotion]
            all_emotions = averaged_emotions
            
            # Add to emotion history for smoothing
            emotion_history.append((detected_emotion, confidence_score))
            if len(emotion_history) > history_length:
                emotion_history.pop(0)
            
            # Get most frequent emotion from recent history with good confidence
            recent_emotions = [e for e, c in emotion_history if c > confidence_threshold]
            if recent_emotions:
                emotion_counts = {}
                for emotion in recent_emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                most_frequent = max(emotion_counts, key=emotion_counts.get)
                if emotion_counts[most_frequent] >= 2:  # Need at least 2 consistent detections
                    detected_emotion = most_frequent
                    confidence_score = max([c for e, c in emotion_history if e == most_frequent])

    # Send commands based on emotion (only if different from last sent)
    if detected_emotion != "No face detected" and confidence_score > confidence_threshold:
        if detected_emotion != last_sent_emotion:
            if detected_emotion in ["happy"]:
                send_gesture('t', detected_emotion)  # thumbs up
                last_sent_emotion = detected_emotion
            elif detected_emotion in ["angry", "sad"]:
                send_gesture('c', detected_emotion)  # close
                last_sent_emotion = detected_emotion
            elif detected_emotion in ["surprise", "fear"]:
                send_gesture('o', detected_emotion)  # open
                last_sent_emotion = detected_emotion

    # Display information on screen
    emotion_color = get_emotion_color(detected_emotion)
    
    # Main emotion display
    emotion_text = f'Emotion: {detected_emotion.title()}'
    confidence_text = f'Confidence: {confidence_score:.2f}'
    
    cv2.putText(frame, emotion_text, 
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)
    cv2.putText(frame, confidence_text, 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show all emotion scores if face detected
    if all_emotions:
        y_offset = 130
        cv2.putText(frame, 'All Emotions:', 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for emotion, score in sorted(all_emotions.items(), key=lambda x: x[1], reverse=True):
            y_offset += 25
            color = get_emotion_color(emotion)
            cv2.putText(frame, f'{emotion}: {score:.2f}', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Action indicators
    if detected_emotion.lower() == "happy":
        cv2.putText(frame, '😊 -> Thumbs Up', 
                   (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif detected_emotion.lower() in ["angry", "sad"]:
        cv2.putText(frame, f'{detected_emotion} -> Close Hand', 
                   (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif detected_emotion.lower() in ["surprise", "fear"]:
        cv2.putText(frame, f'{detected_emotion} -> Open Hand', 
                   (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Instructions
    cv2.putText(frame, 'Press Q to quit', 
               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show emotion history as small dots
    for i, (emotion, conf) in enumerate(emotion_history):
        color = get_emotion_color(emotion)
        cv2.circle(frame, (frame.shape[1] - 100 + i * 15, 30), 5, color, -1)

    # Show webcam feed
    cv2.imshow("Improved Face Control Robot Hand", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
    print("Arduino connection closed.") 