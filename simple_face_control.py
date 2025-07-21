import cv2
import serial
import time
import numpy as np
import traceback

print("🤖 Optimized Face Control Robot Hand")
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

print("Loading optimized face detection...")

face_cascade_main = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

detectors_loaded = 0
if not face_cascade_main.empty():
    detectors_loaded += 1
    print("✓ Main face detector loaded")
if not face_cascade_alt.empty():
    detectors_loaded += 1
    print("✓ Alternative face detector loaded")

if detectors_loaded == 0:
    print("Error: Could not load face detection models. Exiting...")
    exit()

print(f"Loaded {detectors_loaded} optimized face detectors!")

def fast_enhance_frame(gray):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
    enhanced = clahe.apply(gray)
    return enhanced

def detect_faces_optimized(gray, frame_count):
    faces = []
    
    if frame_count % 2 == 0:
        faces = face_cascade_main.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    else:
        if not face_cascade_alt.empty():
            faces = face_cascade_alt.detectMultiScale(
                gray, 
                scaleFactor=1.15, 
                minNeighbors=3,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
    
    return faces

def detect_smiles_fast(roi_gray):
    if smile_cascade.empty():
        return []
    
    smiles = smile_cascade.detectMultiScale(
        roi_gray, 
        scaleFactor=1.8, 
        minNeighbors=20,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return smiles

print("Optimized detection loaded!")
print("Performance improvements:")
print("- Alternating detection algorithms")
print("- Reduced computational overhead")
print("- Faster image enhancement")
print("- Optimized detection parameters")
print("Instructions:")
print("- Smile for thumbs up")
print("- Neutral face for close hand")
print("- Wave/move for open hand")
print("- Press 'q' to quit")

frame_count = 0
last_emotion = None
detection_history = []
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
            
            if frame_count % 3 == 0:
                gray = fast_enhance_frame(gray)
            
            detected_emotion = "neutral"
            
            try:
                faces = detect_faces_optimized(gray, frame_count)
                
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    
                    if x >= 0 and y >= 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0]:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        
                        roi_gray = gray[y:y+h, x:x+w]
                        
                        if roi_gray.shape[0] > 0 and roi_gray.shape[1] > 0:
                            if frame_count % 2 == 0:
                                smiles = detect_smiles_fast(roi_gray)
                                
                                if len(smiles) > 0:
                                    detected_emotion = "happy"
                                    for sx, sy, sw, sh in smiles:
                                        if sx >= 0 and sy >= 0 and sx+sw <= w and sy+sh <= h:
                                            cv2.rectangle(frame[y:y+h, x:x+w], (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                            
                            if frame_count > 10:
                                face_center = (x + w//2, y + h//2)
                                if detection_history:
                                    last_center = detection_history[-1]
                                    movement = abs(face_center[0] - last_center[0]) + abs(face_center[1] - last_center[1])
                                    if movement > 30:
                                        detected_emotion = "surprise"
                                
                                detection_history.append(face_center)
                                if len(detection_history) > 5:
                                    detection_history.pop(0)
                
                if detected_emotion != last_emotion and frame_count % 15 == 0:
                    if detected_emotion == "happy":
                        send_gesture('t', "smile")
                    elif detected_emotion == "surprise":
                        send_gesture('o', "wave")
                    else:
                        send_gesture('c', "neutral")
                    last_emotion = detected_emotion
                
                color = (0, 255, 0) if detected_emotion == "happy" else (255, 0, 0) if detected_emotion == "surprise" else (255, 255, 255)
                
                cv2.putText(frame, f'Emotion: {detected_emotion.title()}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f'Faces: {len(faces)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_counter = 0
                    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, f'Frame: {frame_count}', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, 'Optimized Detection', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            except Exception as detection_error:
                print(f"Detection error at frame {frame_count}: {detection_error}")
                continue

            try:
                cv2.imshow("Optimized Face Control Robot Hand", frame)
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