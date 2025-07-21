import cv2
import serial
import time
import numpy as np
import traceback
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

print("🤖 HSEmotion Face Control Robot Hand")
print("Connecting to Arduino...")

possible_ports = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8']
arduino = None

for port in possible_ports:
    try:
        print(f"Trying to connect to {port}...")
        arduino = serial.Serial(port, 9600, timeout=1)
        print(f"Successfully connected to Arduino on {port}")
        break
    except serial.SerialException:
        print(f"Could not connect to {port}")
        continue

if arduino is None:
    print("No Arduino found. Running in simulation mode...")
else:
    time.sleep(2)

def send_gesture(command, emotion):
    try:
        if arduino and arduino.is_open:
            arduino.write(command.encode())
            print(f"Sent to Arduino: {command} ({emotion})")
        else:
            print(f"SIMULATION: {command} ({emotion})")
    except Exception as e:
        print(f"Error sending gesture: {e}")

print("Initializing camera...")
cap = None

for i in range(3):
    print(f"Trying camera index {i}...")
    try:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            test_ret, test_frame = cap.read()
            if test_ret and test_frame is not None:
                print(f"Camera {i} works! Frame shape: {test_frame.shape}")
                break
            else:
                print(f"Camera {i} opened but can't read frames")
                cap.release()
                cap = None
        else:
            print(f"Camera {i} failed to open")
            cap = None
    except Exception as e:
        print(f"Error with camera {i}: {e}")
        cap = None

if cap is None or not cap.isOpened():
    print("ERROR: No working camera found!")
    print("Please check:")
    print("1. Camera is connected")
    print("2. No other apps are using the camera")
    print("3. Camera permissions are enabled")
    exit(1)

print("Camera opened successfully!")

try:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {actual_width}x{actual_height} @ {actual_fps}fps")
except Exception as e:
    print(f"Error setting camera properties: {e}")

print("Loading HSEmotion model...")

try:
    model_name = 'enet_b0_8_best_afew'
    emotion_recognizer = HSEmotionRecognizer(model_name=model_name)
    print(f"✓ HSEmotion model '{model_name}' loaded successfully")
except Exception as e:
    print(f"Error loading HSEmotion model: {e}")
    print("Falling back to OpenCV detection...")
    emotion_recognizer = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error: Could not load face detection model. Exiting...")
    exit()

emotion_colors = {
    'anger': (0, 0, 255),
    'contempt': (128, 0, 128),
    'disgust': (0, 128, 128),
    'fear': (0, 165, 255),
    'happiness': (0, 255, 0),
    'neutral': (255, 255, 255),
    'sadness': (255, 0, 0),
    'surprise': (255, 255, 0)
}

def get_emotion_color(emotion):
    return emotion_colors.get(emotion.lower(), (255, 255, 255))

def detect_emotions_hsemotion(face_img):
    try:
        if emotion_recognizer is None:
            return "neutral", 0.0, {}
        
        emotion, scores = emotion_recognizer.predict_emotions(face_img, logits=False)
        
        all_emotions = {
            'anger': scores[0],
            'disgust': scores[1], 
            'fear': scores[2],
            'happiness': scores[3],
            'neutral': scores[4],
            'sadness': scores[5],
            'surprise': scores[6]
        }
        
        confidence = max(scores)
        
        return emotion.lower(), confidence, all_emotions
        
    except Exception as e:
        print(f"HSEmotion detection error: {e}")
        return "neutral", 0.0, {}

print("HSEmotion detection loaded!")
print("Advanced features:")
print("- State-of-the-art emotion recognition")
print("- 7 emotion classes: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise")
print("- Real-time confidence scores")
print("- No TensorFlow dependency")
print("Instructions:")
print("- Happy emotions → Thumbs up")
print("- Sad/Angry emotions → Close hand")
print("- Surprise/Fear → Open hand")
print("- Press 'q' to quit")

frame_count = 0
last_emotion = None
emotion_history = []
confidence_threshold = 0.3
fps_counter = 0
fps_start_time = time.time()

try:
    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Camera read failed at frame {frame_count}")
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_emotion = "neutral"
            confidence_score = 0.0
            all_emotions = {}
            
            try:
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(50, 50),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    
                    if x >= 0 and y >= 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0]:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        
                        face_img = frame[y:y+h, x:x+w]
                        
                        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                            detected_emotion, confidence_score, all_emotions = detect_emotions_hsemotion(face_img)
                            
                            emotion_history.append((detected_emotion, confidence_score))
                            if len(emotion_history) > 5:
                                emotion_history.pop(0)
                            
                            if len(emotion_history) >= 3:
                                recent_emotions = [e for e, c in emotion_history if c > confidence_threshold]
                                if recent_emotions:
                                    emotion_counts = {}
                                    for emotion in recent_emotions:
                                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                                    if emotion_counts:
                                        most_frequent = max(emotion_counts, key=emotion_counts.get)
                                        if emotion_counts[most_frequent] >= 2:
                                            detected_emotion = most_frequent
                
                if detected_emotion != last_emotion and frame_count % 20 == 0 and confidence_score > confidence_threshold:
                    if detected_emotion in ["happiness"]:
                        send_gesture('t', "happy")
                    elif detected_emotion in ["anger", "sadness", "disgust"]:
                        send_gesture('c', "negative emotion")
                    elif detected_emotion in ["surprise", "fear"]:
                        send_gesture('o', "surprise/fear")
                    else:
                        send_gesture('c', "neutral")
                    last_emotion = detected_emotion
                
                emotion_color = get_emotion_color(detected_emotion)
                
                cv2.putText(frame, f'Emotion: {detected_emotion.title()}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)
                cv2.putText(frame, f'Confidence: {confidence_score:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'Faces: {len(faces)}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                threshold_text = f'Threshold: {confidence_threshold} {"✓" if confidence_score > confidence_threshold else "✗"}'
                cv2.putText(frame, threshold_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if confidence_score > confidence_threshold else (0, 0, 255), 1)
                
                if all_emotions:
                    y_offset = 200
                    cv2.putText(frame, 'Emotion Scores:', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:4]
                    for emotion, score in sorted_emotions:
                        y_offset += 25
                        color = get_emotion_color(emotion)
                        thickness = 2 if score > confidence_threshold else 1
                        cv2.putText(frame, f'{emotion}: {score:.3f}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)
                
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_counter = 0
                    cv2.putText(frame, f'FPS: {fps:.1f}', (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, f'Frame: {frame_count}', (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, 'HSEmotion Detection', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            except Exception as detection_error:
                print(f"Detection error at frame {frame_count}: {detection_error}")
                continue

            try:
                cv2.imshow("HSEmotion Face Control Robot Hand", frame)
            except Exception as display_error:
                print(f"Display error at frame {frame_count}: {display_error}")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit pressed")
                break
                
        except Exception as frame_error:
            print(f"Frame processing error at frame {frame_count}: {frame_error}")
            continue

except KeyboardInterrupt:
    print("Interrupted")
except Exception as e:
    print(f"Main error: {e}")
    traceback.print_exc()

finally:
    print("Shutting down...")
    try:
        if cap:
            cap.release()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass
    try:
        if arduino and arduino.is_open:
            arduino.close()
            print("Arduino disconnected")
    except:
        pass
    print("Done.") 