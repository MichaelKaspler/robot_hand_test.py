import cv2
import mediapipe as mp
import serial
import serial.tools.list_ports
import time
import numpy as np
import random

class RockPaperScissorsGame:
    def __init__(self, baud_rate=9600):
        # Initialize MediaPipe for hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Connect to Arduino
        self.arduino = self.find_and_connect_arduino(baud_rate)
        
        # Initialize webcam with larger resolution
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Cannot open webcam")
        
        # Set larger camera resolution for bigger window
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Hand landmark indices
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [3, 6, 10, 14, 18]
        self.finger_mcp = [2, 5, 9, 13, 17]
        
        # Game state
        self.player_score = 0
        self.computer_score = 0
        self.rounds_played = 0
        self.current_gesture = "None"
        self.computer_gesture = "None"
        self.game_result = ""
        self.game_state = "waiting"  # waiting, countdown, playing, result
        
        # Gesture patterns for robotic hand
        self.gesture_patterns = {
            'rock': '00000',     # Closed fist
            'paper': '11111',    # Open hand
            'scissors': '01100', # Index and middle finger
            'none': '11110'      # Default neutral
        }
        
        # Timing
        self.gesture_history = []
        self.history_size = 5
        self.stable_frames = 0
        self.required_stable_frames = 10
        self.countdown_start = 0
        self.result_display_time = 0
        
        print("🎮 Rock Paper Scissors Game Initialized!")
        print("📝 Game Rules:")
        print("   👊 Rock: Make a fist (0 fingers)")
        print("   ✋ Paper: Open hand (5 fingers)")
        print("   ✌️ Scissors: Index + middle finger (2 fingers)")
        print("   Press SPACE to start a round, Q to quit")

    def find_and_connect_arduino(self, baud_rate):
        """Find and connect to Arduino"""
        available_ports = serial.tools.list_ports.comports()
        
        if not available_ports:
            print("No COM ports found! Game will run without robotic hand.")
            return None
        
        for port in available_ports:
            try:
                print(f"Trying {port.device}...")
                arduino = serial.Serial(port.device, baud_rate, timeout=2)
                time.sleep(2)
                arduino.write(b'test')
                arduino.flush()
                print(f"✅ Connected to Arduino on {port.device}")
                return arduino
            except (serial.SerialException, OSError) as e:
                print(f"Failed to connect to {port.device}: {e}")
                continue
        
        print("No Arduino found! Game will run without robotic hand.")
        return None

    def detect_hand_orientation(self, landmarks):
        """Detect if hand is left or right"""
        thumb_x = landmarks[4].x
        pinky_x = landmarks[20].x
        return 'Right' if thumb_x > pinky_x else 'Left'

    def is_finger_extended(self, landmarks, finger_idx, hand_orientation):
        """Check if a specific finger is extended"""
        if finger_idx == 0:  # Thumb
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

    def count_fingers(self, landmarks):
        """Count extended fingers and determine gesture"""
        try:
            hand_orientation = self.detect_hand_orientation(landmarks)
            
            finger_states = []
            for i in range(5):
                is_extended = self.is_finger_extended(landmarks, i, hand_orientation)
                finger_states.append(is_extended)
            
            finger_count = sum(finger_states)
            
            # Determine gesture based on finger count and pattern
            gesture = self.classify_gesture(finger_count, finger_states)
            
            return finger_count, finger_states, gesture
            
        except Exception as e:
            print(f"Error in finger detection: {e}")
            return 0, [False] * 5, "none"

    def classify_gesture(self, finger_count, finger_states):
        """Classify the gesture as rock, paper, or scissors"""
        if finger_count == 0:
            return "rock"
        elif finger_count == 5:
            return "paper"
        elif finger_count == 2:
            # Check if it's index and middle finger (scissors)
            if finger_states[1] and finger_states[2]:  # Index and middle
                return "scissors"
            else:
                return "none"  # Two fingers but not scissors pattern
        else:
            return "none"

    def send_gesture_to_arduino(self, gesture):
        """Send gesture pattern to Arduino"""
        if self.arduino is None:
            print(f"Would send gesture: {gesture}")
            return
            
        try:
            pattern = self.gesture_patterns.get(gesture, '11110')
            command = pattern + '\n'
            self.arduino.write(command.encode())
            print(f"Sent {gesture}: {pattern}")
        except serial.SerialException as e:
            print(f"Error sending command to Arduino: {e}")

    def get_computer_choice(self):
        """Generate random computer choice"""
        choices = ['rock', 'paper', 'scissors']
        return random.choice(choices)

    def determine_winner(self, player, computer):
        """Determine the winner of the round"""
        if player == computer:
            return "tie"
        elif (player == "rock" and computer == "scissors") or \
             (player == "paper" and computer == "rock") or \
             (player == "scissors" and computer == "paper"):
            return "player"
        else:
            return "computer"

    def start_countdown(self):
        """Start a new round countdown"""
        self.game_state = "countdown"
        self.countdown_start = time.time()
        self.computer_gesture = self.get_computer_choice()
        print(f"\n🎲 Computer chose: {self.computer_gesture}")

    def run_game(self):
        """Main game loop"""
        print("\n🚀 Starting Rock Paper Scissors Game!")
        print("📌 Show your gesture when the countdown ends!")
        
        # Set up the window to be larger from the start
        cv2.namedWindow('Rock Paper Scissors Game', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Rock Paper Scissors Game', 1280, 720)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hands.process(rgb_frame)
            
            # Initialize current gesture
            current_gesture = "none"
            finger_count = 0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Count fingers and classify gesture
                    finger_count, finger_states, current_gesture = self.count_fingers(hand_landmarks.landmark)
            
            # Update gesture history for stability
            self.gesture_history.append(current_gesture)
            if len(self.gesture_history) > self.history_size:
                self.gesture_history.pop(0)
            
            # Get most frequent gesture
            if len(self.gesture_history) >= 3:
                gesture_counts = {}
                for gesture in self.gesture_history:
                    gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                stable_gesture = max(gesture_counts, key=gesture_counts.get)
                
                if stable_gesture == self.current_gesture:
                    self.stable_frames += 1
                else:
                    self.stable_frames = 0
                    self.current_gesture = stable_gesture
            
            # Game state management
            current_time = time.time()
            
            if self.game_state == "waiting":
                # Display instructions with larger text and better colors
                cv2.putText(frame, "Press SPACE to start round", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, f"Score - You: {self.player_score} | Computer: {self.computer_score}", 
                           (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
            elif self.game_state == "countdown":
                countdown_time = current_time - self.countdown_start
                if countdown_time < 3:
                    count = int(3 - countdown_time)
                    cv2.putText(frame, f"Get ready... {count}", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                else:
                    self.game_state = "playing"
                    
            elif self.game_state == "playing":
                cv2.putText(frame, "SHOW YOUR GESTURE!", (250, 360), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
                
                # Check for stable gesture
                if self.stable_frames >= self.required_stable_frames and self.current_gesture != "none":
                    # Round complete!
                    result = self.determine_winner(self.current_gesture, self.computer_gesture)
                    self.rounds_played += 1
                    
                    if result == "player":
                        self.player_score += 1
                        self.game_result = "YOU WIN!"
                    elif result == "computer":
                        self.computer_score += 1
                        self.game_result = "COMPUTER WINS!"
                    else:
                        self.game_result = "TIE!"
                    
                    # Show both gestures on robotic hand
                    print(f"\n🎯 Round {self.rounds_played} Results:")
                    print(f"   Player: {self.current_gesture}")
                    print(f"   Computer: {self.computer_gesture}")
                    print(f"   Result: {self.game_result}")
                    
                    # First show player gesture, then computer gesture
                    self.send_gesture_to_arduino(self.current_gesture)
                    
                    self.game_state = "result"
                    self.result_display_time = current_time
            
            elif self.game_state == "result":
                # Display results with larger text and colorful styling
                cv2.putText(frame, f"You: {self.current_gesture}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                cv2.putText(frame, f"Computer: {self.computer_gesture}", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
                cv2.putText(frame, self.game_result, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0) if "WIN" in self.game_result else (0, 0, 255), 4)
                cv2.putText(frame, "Press SPACE for next round", (50, 520), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
                # Show computer gesture after 2 seconds
                if current_time - self.result_display_time > 2:
                    self.send_gesture_to_arduino(self.computer_gesture)
                    if current_time - self.result_display_time > 4:
                        # Reset to neutral after showing both gestures
                        self.send_gesture_to_arduino('none')
            
            # Always display current gesture and finger count with larger text and colors
            cv2.putText(frame, f"Gesture: {self.current_gesture} ({finger_count} fingers)", 
                       (20, frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.putText(frame, f"Stability: {self.stable_frames}/{self.required_stable_frames}", 
                       (20, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
            
            # Display gesture legend with larger text and colorful styling
            legend_items = [
                ("Rock: Fist (0 fingers)", (0, 165, 255)),      # Orange
                ("Paper: Open hand (5 fingers)", (255, 255, 0)), # Cyan  
                ("Scissors: Index + middle (2 fingers)", (255, 0, 255)) # Magenta
            ]
            for i, (item, color) in enumerate(legend_items):
                cv2.putText(frame, item, (frame.shape[1] - 500, 50 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            cv2.imshow('Rock Paper Scissors Game', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar
                if self.game_state in ["waiting", "result"]:
                    self.start_countdown()
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        if self.arduino:
            self.arduino.close()
        print(f"\n🎮 Game Over! Final Score - You: {self.player_score} | Computer: {self.computer_score}")

def main():
    try:
        game = RockPaperScissorsGame()
        game.run_game()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 