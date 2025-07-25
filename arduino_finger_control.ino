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
  //mainwork();
  //fist();//full close
  //thumpsup();
  //open_hand();//full open
  //wave();
  //manual_control(4);
///////////////////////void hand_control() you want to use function and only change this function dont touch any thing else 

  
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

void finger0close() {
  int angle = 0;
  setServo(0, angle);
}
///////////////////////////////////////////////do here keep going till you do all finger combanitions 
void hand_control() {
  if (Serial.available()) {
    char command = Serial.read();
    Serial.print("Received command: ");
    Serial.println(command);

    if (command == '6') {
      finger4close();
      finger3open();
      finger2close();
      finger0open();
    } 
    else if (command == '5') {
      finger4close();
      finger3close();
      finger2open();
      finger0open();
    } 
    else if (command == '4') {
      finger4open();
      finger3close();
      finger2close();
      finger0close();
    } 
    else if (command == '3') {
      finger4close();
      finger3open();
      finger2close();
      finger0close();
    } 
    else if (command == '2') {
      finger4close();
      finger3close();
      finger2open();
      finger0close();
    } 
    else if (command == '0') {
      finger4close();
      finger3close();
      finger2close();
      finger0open();
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
