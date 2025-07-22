#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Servo.h>

unsigned long startTime;
Servo gateServo;
const int gateServoPin = 3;
const int opengate = 15;
const int closegate = 120;
int count = 0;
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
const int numFingers = 5;

#define SERVOMIN  120
#define SERVOMAX  530

void setServo(int channel, int angle);
void fist();
void open_hand();
void wave();
void wait();
void mainwork();
void timer(unsigned long durationMillis);
void manual_control(int finger);
void hand_control();

void setup() {
  Serial.begin(9600);
  gateServo.attach(gateServoPin);
  gateServo.write(closegate);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);
  Serial.println("Setup complete. Gate closed.");
}

void loop() {
  
  hand_control();
}

void setServo(int channel, int angle) {
  angle = constrain(angle, 0, 180);
  int pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(channel, 0, pulse);
}


void finger4open() {
  int angle = 0;
  setServo(4, angle);
}

void finger3open() {
  int angle = 0;
  setServo(3, angle);
}

void finger2open() {
  int angle = 120;
  setServo(2, angle);
}

void finger1open() {
  int angle = 0;
  setServo(1, angle);
}

void finger0open() {
  int angle = 120;
  setServo(0, angle);
}

void finger4close() {
  int angle = 150;
  setServo(4, angle);
}

void finger3close() {
  int angle = 150;
  setServo(3, angle);
}

void finger2close() {
  int angle = 0;
  setServo(2, angle);
}

void finger1close() {
  int angle = 150;
  setServo(1, angle);
}

void finger0close() {
  int angle = 0;
  setServo(0, angle);
}


void hand_control() {
  if (Serial.available()) {
    char command = Serial.read();
    Serial.print("Received command: ");
    Serial.println(command);

    // Handle all finger counts from 0-5
    if (command == '0') {
      // Fist - all fingers closed
      finger4close();
      finger3close();
      finger2close();
      finger1close();
      finger0close();
      Serial.println("Gesture: Fist (0 fingers)");
    } 
    else if (command == '1') {
      // One finger up (thumb)
      finger4close();
      finger3close();
      finger2open();  // Thumb up
      finger1close();
      finger0close();
      Serial.println("Gesture: Thumb up (1 finger)");
    } 
    else if (command == '2') {
      // Two fingers up (thumb + index)
      finger4close();
      finger3close();
      finger2open();  // Thumb
      finger1close();
      finger0open();  // Index
      Serial.println("Gesture: Two fingers (2 fingers)");
    } 
    else if (command == '3') {
      // Three fingers up (thumb + index + middle)
      finger4close();
      finger3open();  // Middle
      finger2open();  // Thumb
      finger1close();
      finger0open();  // Index
      Serial.println("Gesture: Three fingers (3 fingers)");
    } 
    else if (command == '4') {
      // Four fingers up (all except pinky)
      finger4close(); // Pinky closed
      finger3open();  // Ring
      finger2open();  // Thumb
      finger1open();  // Middle
      finger0open();  // Index
      Serial.println("Gesture: Four fingers (4 fingers)");
    } 
    else if (command == '5') {
      // All fingers up (open hand)
      finger4open();  // Pinky
      finger3open();  // Ring
      finger2open();  // Thumb
      finger1open();  // Middle
      finger0open();  // Index
      Serial.println("Gesture: Open hand (5 fingers)");
    }
    else if (command == 't' || command == 'T') {
      // Test command - ignore (sent during connection test)
      Serial.println("Test command received - Arduino connected!");
    }
    else {
      Serial.print("Unknown command: ");
      Serial.println(command);
    }
  }
}

void fist() {
  for (int i = 0; i < numFingers; i++) {
    int angle = (i == 0 || i == 2) ? 0 : 150;
    setServo(i, angle);
  }
}

void open_hand() {
  for (int i = 0; i < numFingers; i++) {
    int angle = (i == 0 || i == 2) ? 150 : 0;
    setServo(i, angle);
  }
}

void wave() {
  fist();
  timer(650);
  open_hand();
  timer(650);
}

void thumpsup() {
  for (int i = 0; i < numFingers; i++) {
    int angle = (i == 2) ? 0 : 150;
    setServo(i, angle);
  }
}

void wait() {
  for (int i = 0; i < numFingers; i++) {
    int angle = (i == 3 || i == 4) ? 150 : 30;
    setServo(i, angle);
  }
}

void timer(unsigned long durationMillis) {
  unsigned long startTime = millis();
  while (millis() - startTime < durationMillis) {
  }
}

void mainwork() {
}

void manual_control(int finger) {
} 