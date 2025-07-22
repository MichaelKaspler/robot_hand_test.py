import cv2
import mediapipe as mp
import serial
import serial.tools.list_ports
import time
import numpy as np

class FingerCountingController:
    def __init__(self, baud_rate=9600):
        
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  
            min_detection_confidence=0.8,  
            min_tracking_confidence=0.7,   
            model_complexity=1             
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        
        self.arduino = self.find_and_connect_arduino(baud_rate)
        
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Cannot open webcam")
        
        
        self.finger_tips = [4, 8, 12, 16, 20]  
        self.finger_pips = [3, 6, 10, 14, 18]  
        self.finger_mcp = [2, 5, 9, 13, 17]    
        self.last_finger_count = -1
        self.last_finger_states = [False] * 5  
        self.stable_count_frames = 0
        self.stable_state_frames = 0
        self.required_stable_frames = 8  
        self.finger_history = []
        self.history_size = 5
    
    def find_and_connect_arduino(self, baud_rate):
        
        available_ports = serial.tools.list_ports.comports()
        
        if not available_ports:
            print("No COM ports found!")
            return None
        
        for port in available_ports:
            try:
                print(f"Trying {port.device}...")
                arduino = serial.Serial(port.device, baud_rate, timeout=2)
                time.sleep(2)  
                arduino.write(b'test')
                arduino.flush()
                
                return arduino
                
            except (serial.SerialException, OSError) as e:
                print(f"Failed to connect to {port.device}: {e}")
                continue
        
        return None
        
    def detect_hand_orientation(self, landmarks):
       
        thumb_x = landmarks[4].x
        pinky_x = landmarks[20].x
        
        return 'Right' if thumb_x > pinky_x else 'Left'
    
    def is_finger_extended(self, landmarks, finger_idx, hand_orientation):
        
        if finger_idx == 0:  
            
            tip = landmarks[self.finger_tips[0]]
            pip = landmarks[self.finger_pips[0]]
            mcp = landmarks[self.finger_mcp[0]]
            
            if hand_orientation == 'Right':
                
                return tip.x > pip.x and tip.x > mcp.x
            else:
                
                return tip.x < pip.x and tip.x < mcp.x
        else:
            
            tip = landmarks[self.finger_tips[finger_idx]]
            pip = landmarks[self.finger_pips[finger_idx]]
            mcp = landmarks[self.finger_mcp[finger_idx]]
            
            tip_above_pip = tip.y < pip.y - 0.02  
            tip_above_mcp = tip.y < mcp.y
            
            return tip_above_pip and tip_above_mcp
    
    def count_fingers_enhanced(self, landmarks):
          
        try:
            
            hand_orientation = self.detect_hand_orientation(landmarks)
            
            
            finger_states = []
            for i in range(5):
                is_extended = self.is_finger_extended(landmarks, i, hand_orientation)
                finger_states.append(is_extended)
            
            
            self.finger_history.append(finger_states)
            if len(self.finger_history) > self.history_size:
                self.finger_history.pop(0)
            
            
            if len(self.finger_history) >= 3:
                smoothed_states = []
                for finger_idx in range(5):
                    finger_votes = [state[finger_idx] for state in self.finger_history]
                    
                    smoothed_states.append(sum(finger_votes) > len(finger_votes) // 2)
                finger_states = smoothed_states
            
            finger_count = sum(finger_states)
            return finger_count, finger_states
            
        except Exception as e:
            print(f"Error in finger detection: {e}")
            return 0, [False] * 5
    
    def send_finger_command(self, finger_count, finger_states):

        if self.arduino is None:
            print(f"Would send individual finger commands: {finger_states}")
            return
            
        try:
            
            command = ''
            for state in finger_states:
                command += '1' if state else '0'
            
            command += '\n'
            
            self.arduino.write(command.encode())
            print(f"Sent individual finger command: {command.strip()}")
            print(f"  Fingers: {finger_states} (Total: {finger_count})")
            
        except serial.SerialException as e:
            print(f"Error sending command to Arduino: {e}")
    
    def run(self):
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(frame_rgb)
            
            finger_count = 0
            finger_states = [False] * 5
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    finger_count, finger_states = self.count_fingers_enhanced(hand_landmarks.landmark)
                    
                    cv2.putText(frame, f'Total Fingers: {finger_count}', 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                    for i, (name, state) in enumerate(zip(finger_names, finger_states)):
                        color = (0, 255, 0) if state else (0, 0, 255)  
                        status = "UP" if state else "DOWN"
                        cv2.putText(frame, f'{name}: {status}', 
                                  (10, 110 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    orientation = self.detect_hand_orientation(hand_landmarks.landmark)
                    cv2.putText(frame, f'Hand: {orientation}', 
                              (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            count_stable = (finger_count == self.last_finger_count)
            states_stable = (finger_states == self.last_finger_states)
            
            if count_stable and states_stable:
                self.stable_count_frames += 1
            else:
                self.stable_count_frames = 0
                self.last_finger_count = finger_count
                self.last_finger_states = finger_states.copy()
            
            
            if self.stable_count_frames >= self.required_stable_frames:
                self.send_finger_command(finger_count, finger_states)
                self.stable_count_frames = 0  
                
                
                cv2.putText(frame, 'COMMAND SENT!', 
                          (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            cv2.putText(frame, 'Enhanced Finger Detection - Show hand to camera', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, 'Press Q to quit', 
                       (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Enhanced Finger Counting Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        
        self.cap.release()
        cv2.destroyAllWindows()
        if self.arduino:
            self.arduino.close()
        print("Cleanup completed")

def main():
    
    controller = FingerCountingController()
    controller.run()

if __name__ == "__main__":
    main() 