import cv2
import serial
import time
import numpy as np

print("🤖 Improved Finger Control Robot Hand")
print("Connecting to Arduino...")

possible_ports = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8']
arduino = None

for port in possible_ports:
    try:
        print(f"Trying to connect to {port}...")
        arduino = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)
        print(f"Successfully connected to Arduino on {port}")
        break
    except serial.SerialException:
        print(f"Could not connect to {port}")
        continue

if arduino is None:
    print("No Arduino found. Running in simulation mode...")
else:
    print("Waiting for Arduino to initialize...")
    time.sleep(3)

def send_finger_command(count):
    try:
        if arduino and arduino.is_open:
            if count == 0:
                command = 'c'
                action = "close hand (0 fingers)"
            elif count == 1:
                command = 't'
                action = "thumbs up (1 finger)"
            elif count >= 2:
                command = 'o'
                action = f"open hand ({count} fingers)"
            
            arduino.write(command.encode())
            print(f">>> SENT TO ARDUINO: '{command}' - {action}")
        else:
            print(f"SIMULATION: {count} fingers")
    except Exception as e:
        print(f"Error sending command: {e}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera opened successfully!")
print("Improved finger detection algorithm!")
print("Instructions:")
print("- Place your hand in the detection area (green box)")
print("- 0 fingers = Close hand")
print("- 1 finger = Thumbs up") 
print("- 2+ fingers = Open hand")
print("- Keep hand steady for detection")
print("- Press 'q' to quit")

last_finger_count = -1
stable_frames = 0
required_stability = 12
background = None
frame_count = 0

def create_skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    
    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    return mask

def count_fingers_improved(contour, defects):
    if defects is None:
        return 0
    
    finger_count = 0
    valid_defects = []
    
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        
        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        
        if b > 0 and c > 0:
            s_val = (b**2 + c**2 - a**2) / (2*b*c)
            s_val = max(-1, min(1, s_val))
            angle = np.arccos(s_val)
            
            angle_deg = np.degrees(angle)
            
            if angle_deg <= 80 and d > 15000:
                valid_defects.append((start, end, far, d, angle_deg))
                finger_count += 1
    
    finger_count = min(finger_count + 1, 5)
    
    if finger_count > 5:
        finger_count = 5
    elif finger_count < 0:
        finger_count = 0
    
    return finger_count, valid_defects

def get_hand_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        height, width = frame.shape[:2]
        
        roi_x, roi_y, roi_w, roi_h = width//4, height//4, width//2, height//2
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 3)
        cv2.putText(frame, 'Place hand here', (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        mask = create_skin_mask(roi)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        finger_count = 0
        hand_detected = False
        valid_defects = []
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            if area > 10000:
                hand_detected = True
                
                epsilon = 0.001 * cv2.arcLength(max_contour, True)
                approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)
                
                hull = cv2.convexHull(approx_contour, returnPoints=False)
                defects = cv2.convexityDefects(approx_contour, hull)
                
                finger_count, valid_defects = count_fingers_improved(approx_contour, defects)
                
                contour_roi = max_contour + [roi_x, roi_y]
                cv2.drawContours(frame, [contour_roi], -1, (255, 0, 0), 2)
                
                hull_points = cv2.convexHull(contour_roi)
                cv2.drawContours(frame, [hull_points], -1, (0, 0, 255), 2)
                
                center = get_hand_center(contour_roi)
                if center:
                    cv2.circle(frame, center, 8, (255, 255, 0), -1)
                
                for start, end, far, d, angle in valid_defects:
                    start_roi = (start[0] + roi_x, start[1] + roi_y)
                    end_roi = (end[0] + roi_x, end[1] + roi_y)
                    far_roi = (far[0] + roi_x, far[1] + roi_y)
                    
                    cv2.circle(frame, start_roi, 5, (0, 255, 0), -1)
                    cv2.circle(frame, end_roi, 5, (0, 255, 0), -1)
                    cv2.circle(frame, far_roi, 5, (255, 0, 255), -1)
                    cv2.line(frame, start_roi, far_roi, (0, 255, 255), 2)
                    cv2.line(frame, end_roi, far_roi, (0, 255, 255), 2)
        
        finger_count = max(0, min(finger_count, 5))
        
        if hand_detected:
            if finger_count == last_finger_count:
                stable_frames += 1
                if stable_frames >= required_stability:
                    send_finger_command(finger_count)
                    stable_frames = 0
            else:
                last_finger_count = finger_count
                stable_frames = 0
        else:
            stable_frames = 0
            last_finger_count = -1
        
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        stability_color = (0, 255, 0) if stable_frames > required_stability//2 else (0, 255, 255)
        
        cv2.putText(frame, f'Fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        cv2.putText(frame, f'Hand: {"YES" if hand_detected else "NO"}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f'Stability: {stable_frames}/{required_stability}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stability_color, 2)
        cv2.putText(frame, f'Defects: {len(valid_defects)}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if finger_count == 0:
            action_text = "CLOSE HAND"
            action_color = (0, 0, 255)
        elif finger_count == 1:
            action_text = "THUMBS UP"
            action_color = (0, 255, 0)
        else:
            action_text = "OPEN HAND"
            action_color = (255, 0, 0)
        
        cv2.putText(frame, f'Action: {action_text}', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)
        cv2.putText(frame, f'Last sent: {last_finger_count if last_finger_count >= 0 else "None"}', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, 'Press Q to quit', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Improved Finger Control', frame)
        cv2.imshow('Hand Mask', mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted")
except Exception as e:
    print(f"Error: {e}")

finally:
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    if arduino and arduino.is_open:
        arduino.close()
        print("Arduino disconnected")
    print("Done.") 