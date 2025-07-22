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
void hand_control_individual();

void setup() {
  Serial.begin(9600);
  gateServo.attach(gateServoPin);
  gateServo.write(closegate);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);
  Serial.println("Setup complete. Individual finger control ready.");
}

void loop() {
  
  hand_control_individual();
}

void setServo(int channel, int angle) {
  angle = constrain(angle, 0, 180);
  int pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(channel, 0, pulse);
}


void finger0open() {  // THUMB
  int angle = 120;
  setServo(0, angle);
}

void finger0close() { // THUMB
  int angle = 0;
  setServo(0, angle);
}

void finger1open() {  // INDEX
  int angle = 120;
  setServo(1, angle);
}

void finger1close() { // INDEX
  int angle = 0;
  setServo(1, angle);
}

void finger2open() {  // MIDDLE
  int angle = 120;
  setServo(2, angle);
}

void finger2close() { // MIDDLE
  int angle = 0;
  setServo(2, angle);
}

void finger3open() {  // RING
  int angle = 0;       // Opens at 0
  setServo(3, angle);
}

void finger3close() { // RING
  int angle = 150;     // Closes at 150
  setServo(3, angle);
}

void finger4open() {  // PINKY
  int angle = 0;       // Opens at 0
  setServo(4, angle);
}

void finger4close() { // PINKY
  int angle = 150;     // Closes at 150
  setServo(4, angle);
}


void hand_control_individual() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim(); 
    
    Serial.print("Received command: ");
    Serial.println(command);
    
    
    if (command == "test") {
      Serial.println("Individual finger control ready!");
      return;
    }
    
    
    if (command.length() == 5) {
      Serial.println("Finger states:");

      if (command.charAt(0) == '1') {
        finger0open();
        Serial.println("  Thumb: OPEN");
      } else {
        finger0close();
        Serial.println("  Thumb: CLOSED");
      }
      
      if (command.charAt(1) == '1') {
        finger1open();
        Serial.println("  Index: OPEN");
      } else {
        finger1close();
        Serial.println("  Index: CLOSED");
      }
      
      if (command.charAt(2) == '1') {
        finger2open();
        Serial.println("  Middle: OPEN");
      } else {
        finger2close();
        Serial.println("  Middle: CLOSED");
      }
      
      if (command.charAt(3) == '1') {
        finger3open();
        Serial.println("  Ring: OPEN");
      } else {
        finger3close();
        Serial.println("  Ring: CLOSED");
      }
      
      if (command.charAt(4) == '1') {
        finger4open();
        Serial.println("  Pinky: OPEN");
      } else {
        finger4close();
        Serial.println("  Pinky: CLOSED");
      }
      
      Serial.println("Command completed!");
      
    } else {
      Serial.print("Invalid command length: ");
      Serial.print(command.length());
      Serial.println(" (expected 5 digits like 10100)");
      Serial.print("Received: '");
      Serial.print(command);
      Serial.println("'");
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
  while (millis() - startTime < durationMillis) {}
}

void mainwork() {}

void manual_control(int finger) {} 