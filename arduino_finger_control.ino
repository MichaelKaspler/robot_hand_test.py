#include <Servo.h>

Servo thumbServo;
Servo indexServo;
Servo middleServo;
Servo ringServo;
Servo pinkyServo;


const int THUMB_PIN = 3;
const int INDEX_PIN = 5;
const int MIDDLE_PIN = 6;
const int RING_PIN = 9;
const int PINKY_PIN = 10;


const int FINGER_CLOSED = 0;    
const int FINGER_OPEN = 90;     

void setup() {
  Serial.begin(9600);
  
  
  thumbServo.attach(THUMB_PIN);
  indexServo.attach(INDEX_PIN);
  middleServo.attach(MIDDLE_PIN);
  ringServo.attach(RING_PIN);
  pinkyServo.attach(PINKY_PIN);
  

  closeAllFingers();
  
  Serial.println("Robot Hand Ready!");
  Serial.println("Send finger count (0-5) to control hand");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    

    int fingerCount = command - '0';
    
    
    if (fingerCount >= 0 && fingerCount <= 5) {
      Serial.print("Received finger count: ");
      Serial.println(fingerCount);
      
      
      controlFingers(fingerCount);
    } else {
      Serial.println("Invalid command. Send 0-5 for finger count.");
    }
  }
}

void controlFingers(int count) {

  closeAllFingers();
  delay(200);
  
  
  switch(count) {
    case 0:
      
      Serial.println("Closed fist");
      break;
      
    case 1:

      thumbServo.write(FINGER_OPEN);
      Serial.println("Thumb up");
      break;
      
    case 2:

      thumbServo.write(FINGER_OPEN);
      indexServo.write(FINGER_OPEN);
      Serial.println("Peace sign");
      break;
      
    case 3:

      thumbServo.write(FINGER_OPEN);
      indexServo.write(FINGER_OPEN);
      middleServo.write(FINGER_OPEN);
      Serial.println("Three fingers");
      break;
      
    case 4:
        
      thumbServo.write(FINGER_OPEN);
      indexServo.write(FINGER_OPEN);
      middleServo.write(FINGER_OPEN);
      ringServo.write(FINGER_OPEN);
      Serial.println("Four fingers");
      break;
      
    case 5:

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