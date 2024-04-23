/***************************************************************
   Motor driver definitions

   Add a "#elif defined" block to this file to include support
   for a particular motor driver.  Then add the appropriate
   #define near the top of the main ROSArduinoBridge.ino file.

   *************************************************************/
void initMotorController() {
  pinMode(RIGHT_MOTOR_BACKWARD, OUTPUT);
  pinMode(LEFT_MOTOR_BACKWARD, OUTPUT);
  pinMode(RIGHT_MOTOR_FORWARD, OUTPUT);
  pinMode(LEFT_MOTOR_FORWARD, OUTPUT);

  ledcSetup(M1_C1, PWM_freq, 8);
  ledcSetup(M1_C2, PWM_freq, 8);
  ledcSetup(M2_C1, PWM_freq, 8);
  ledcSetup(M2_C2, PWM_freq, 8);

  ledcAttachPin(LEFT_MOTOR_BACKWARD, M1_C1);
  ledcAttachPin(LEFT_MOTOR_FORWARD, M1_C2);
  ledcAttachPin(RIGHT_MOTOR_BACKWARD, M2_C1);
  ledcAttachPin(RIGHT_MOTOR_FORWARD, M2_C2);
}


void setMotorSpeeds(int leftSpeed, int rightSpeed) {
  if (leftSpeed > 0) {
    ledcWrite(M1_C2, leftSpeed);
    ledcWrite(M1_C1, 0);
  }
  else if (leftSpeed < 0) {
    ledcWrite(M1_C1, -leftSpeed);
    ledcWrite(M1_C2, 0);
  } else {
    ledcWrite(M1_C1, 0);
    ledcWrite(M1_C2, 0);
  }


  if (rightSpeed > 0) {
    ledcWrite(M2_C2, rightSpeed);
    ledcWrite(M2_C1, 0);
  }
  else if (rightSpeed < 0) {
    ledcWrite(M2_C1, -rightSpeed);
    ledcWrite(M2_C2, 0);
  } else {
    ledcWrite(M2_C1, 0);
    ledcWrite(M2_C2, 0);
  }
}
