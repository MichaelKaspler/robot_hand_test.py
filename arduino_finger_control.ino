/*
  Finger Control Robot Hand
  Receives finger count (0-5) via serial and controls servo motors
  to mimic the finger positions
*/

#include <Servo.h>

// Create servo objects for each finger
Servo thumbServo;
Servo indexServo;
Servo middleServo;
Servo ringServo;
Servo pinkyServo;

// Servo pin assignments (adjust based on your wiring)
const int THUMB_PIN = 3;
const int INDEX_PIN = 5;
const int MIDDLE_PIN = 6;
const int RING_PIN = 9;
const int PINKY_PIN = 10;

// Servo positions (adjust these values based on your servo range)
const int FINGER_CLOSED = 0;    // Position when finger is closed
const int FINGER_OPEN = 90;     // Position when finger is extended

void setup() {
  Serial.begin(9600);
  
  // Attach servos to pins
  thumbServo.attach(THUMB_PIN);
  indexServo.attach(INDEX_PIN);
  middleServo.attach(MIDDLE_PIN);
  ringServo.attach(RING_PIN);
  pinkyServo.attach(PINKY_PIN);
  
  // Initialize all fingers to closed position
  closeAllFingers();
  
  Serial.println("Robot Hand Ready!");
  Serial.println("Send finger count (0-5) to control hand");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    // Convert character to integer
    int fingerCount = command - '0';
    
    // Validate input
    if (fingerCount >= 0 && fingerCount <= 5) {
      Serial.print("Received finger count: ");
      Serial.println(fingerCount);
      
      // Control fingers based on count
      controlFingers(fingerCount);
    } else {
      Serial.println("Invalid command. Send 0-5 for finger count.");
    }
  }
}

void controlFingers(int count) {
  // Close all fingers first
  closeAllFingers();
  delay(200);
  
  // Open fingers based on count
  switch(count) {
    case 0:
      // All fingers closed (already done above)
      Serial.println("Closed fist");
      break;
      
    case 1:
      // Thumb up
      thumbServo.write(FINGER_OPEN);
      Serial.println("Thumb up");
      break;
      
    case 2:
      // Thumb and index finger
      thumbServo.write(FINGER_OPEN);
      indexServo.write(FINGER_OPEN);
      Serial.println("Peace sign");
      break;
      
    case 3:
      // Thumb, index, and middle finger
      thumbServo.write(FINGER_OPEN);
      indexServo.write(FINGER_OPEN);
      middleServo.write(FINGER_OPEN);
      Serial.println("Three fingers");
      break;
      
    case 4:
      // All fingers except pinky
      thumbServo.write(FINGER_OPEN);
      indexServo.write(FINGER_OPEN);
      middleServo.write(FINGER_OPEN);
      ringServo.write(FINGER_OPEN);
      Serial.println("Four fingers");
      break;
      
    case 5:
      // All fingers open
      openAllFingers();
      Serial.println("Open hand");
      break;
  }
}

void closeAllFingers() {
  thumbServo.write(FINGER_CLOSED);
  indexServo.write(FINGER_CLOSED);
  middleServo.write(FINGER_CLOSED);
  ringServo.write(FINGER_CLOSED);
  pinkyServo.write(FINGER_CLOSED);
}

void openAllFingers() {
  thumbServo.write(FINGER_OPEN);
  indexServo.write(FINGER_OPEN);
  middleServo.write(FINGER_OPEN);
  ringServo.write(FINGER_OPEN);
  pinkyServo.write(FINGER_OPEN);
} 