import cv2
import serial
import serial.tools.list_ports
import time
import numpy as np
from deepface import DeepFace
import threading

class EmotionController:
    def __init__(self, baud_rate=9600):
       
        self.arduino = self.find_and_connect_arduino(baud_rate)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Cannot open webcam")
        

        self.emotion_gestures = {
            'happy': '11111',    # Open hand (all fingers) 
            'sad': '00000',      # Fist/closed (all fingers closed) 
            'angry': '00000',    # Fist/closed (all fingers closed)
            'surprise': '11100', # Three fingers (thumb, index, middle)
            'fear': '11000',     # Two fingers (thumb, index)
            'disgust': '00000',  # Fist/closed (all fingers closed)
            'neutral': '11110'   # Four fingers (all except pinky)
        }
        
        
        self.current_emotion = 'neutral'
        self.last_emotion = 'neutral'
        self.emotion_confidence = 0.0
        self.stable_emotion_frames = 0
        self.required_stable_frames = 15 
        self.min_confidence = 0.6  
        
        
        self.emotion_lock = threading.Lock()
        self.emotion_thread_running = True
        
        emotion_descriptions = {
            'happy': 'Open hand (all fingers)',
            'sad': 'Fist (all closed)',
            'angry': 'Fist (all closed)',
            'surprise': '3 fingers (thumb, index, middle)',
            'fear': '2 fingers (thumb, index)',
            'disgust': 'Fist (all closed)',
            'neutral': '4 fingers (all except pinky)'
        }
        for emotion, gesture in self.emotion_gestures.items():
            description = emotion_descriptions.get(emotion, '')
            print(f"  {emotion.capitalize()}: {gesture} - {description}")
    
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
    
    def detect_emotion_thread(self, frame):
       
        try:
            
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion] / 100.0
            
            
            with self.emotion_lock:
                if confidence >= self.min_confidence:
                    self.current_emotion = dominant_emotion.lower()
                    self.emotion_confidence = confidence
                else:
                    self.current_emotion = 'neutral'
                    self.emotion_confidence = confidence
                    
        except Exception as e:
            
            with self.emotion_lock:
                self.current_emotion = 'neutral'
                self.emotion_confidence = 0.0
            print(f"Emotion detection error: {e}")
    
    def send_gesture_command(self, emotion):
        
        if self.arduino is None:
            print(f"Would send gesture for emotion: {emotion}")
            return
            
        try:
            
            command = self.emotion_gestures.get(emotion, '11110') 
            
            
            command_with_newline = command + '\n'
            
            self.arduino.write(command_with_newline.encode())
            print(f"Emotion: {emotion.capitalize()} -> Sent command '{command}'")
            print(f"  Finger pattern: {command} (T={command[0]} I={command[1]} M={command[2]} R={command[3]} P={command[4]})")
            
        except serial.SerialException as e:
            print(f"Error sending command to Arduino: {e}")
    
    def run(self):
        
        print("Look at the camera and show different emotions")
        print("The robotic hand will respond to your emotions")
        print("Press 'q' to quit")
        
        
        last_detection_time = time.time()
        detection_interval = 0.5  
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            
            frame = cv2.flip(frame, 1)
            current_time = time.time()
            
            
            if current_time - last_detection_time >= detection_interval:
                
                emotion_thread = threading.Thread(
                    target=self.detect_emotion_thread, 
                    args=(frame.copy(),)
                )
                emotion_thread.start()
                last_detection_time = current_time
            
            
            with self.emotion_lock:
                display_emotion = self.current_emotion
                display_confidence = self.emotion_confidence
            
            
            if display_emotion == self.last_emotion:
                self.stable_emotion_frames += 1
            else:
                self.stable_emotion_frames = 0
                self.last_emotion = display_emotion
            
            
            if self.stable_emotion_frames >= self.required_stable_frames:
                self.send_gesture_command(display_emotion)
                self.stable_emotion_frames = 0  
            
            
            emotion_text = f"Emotion: {display_emotion.capitalize()}"
            confidence_text = f"Confidence: {display_confidence:.2f}"
            gesture_pattern = self.emotion_gestures.get(display_emotion, '11110')
            gesture_text = f"Gesture: {gesture_pattern} (TIMRP)"
            
            
            cv2.putText(frame, emotion_text, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, gesture_text, 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            
            cv2.putText(frame, 'Show emotions to camera', 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, 'Press Q to quit', 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            
            legend_y = 200
            cv2.putText(frame, 'Emotion Gestures:', 
                       (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            legend_items = [
                'Happy: Open hand (11111)',
                'Sad/Angry: Fist (00000)', 
                'Surprise: 3 fingers (11100)',
                'Fear: 2 fingers (11000)',
                'Neutral: 4 fingers (11110)'
            ]
            
            for i, item in enumerate(legend_items):
                cv2.putText(frame, item, 
                           (10, legend_y + 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow('Emotion-Based Hand Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        
        self.emotion_thread_running = False
        self.cap.release()
        cv2.destroyAllWindows()
        if self.arduino:
            self.arduino.close()
        

def main():
    try:
        
        controller = EmotionController()
        controller.run()
    except Exception as e:
        print(f"Error: {e}")
    

if __name__ == "__main__":
    main() 