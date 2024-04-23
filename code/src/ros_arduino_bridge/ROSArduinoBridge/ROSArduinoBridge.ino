/*********************************************************************
    ROSArduinoBridge

    A set of simple serial commands to control a differential drive
    robot and receive back sensor and odometry data. Default
    configuration assumes use of an Arduino Mega + Pololu motor
    controller shield + Robogaia Mega Encoder shield.  Edit the
    readEncoder() and setMotorSpeed() wrapper functions if using
    different motor controller or encoder method.

    Created for the Pi Robot Project: http://www.pirobot.org
    and the Home Brew Robotics Club (HBRC): http://hbrobotics.org

    Authors: Patrick Goebel, James Nugen

    Inspired and modeled after the ArbotiX driver by Michael Ferguson

    Software License Agreement (BSD License)

    Copyright (c) 2012, Patrick Goebel.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

       Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
       Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

//#define USE_SERVOS  // Enable use of PWM servos as defined in servos.h
#undef USE_SERVOS     // Disable use of PWM servos

/* Serial port baud rate */
#define BAUDRATE     115200

/* Maximum PWM signal */
#define MAX_PWM        255

/* Include definition of serial commands */
#include "commands.h"

/* Motor driver function definitions */
#include "motor_driver.h"

/* Encoder driver function definitions */
#include "encoder_driver.h"

/* PID parameters and functions */
#include "diff_controller.h"

/* Run the PID loop at 30 times per second */
#define PID_RATE           100     // Hz

/* Convert the rate into an interval */
const int PID_INTERVAL = 1000 / PID_RATE;

/* Track the next time we make a PID calculation */
unsigned long nextPID = PID_INTERVAL;

/* Stop the robot if it hasn't received a movement command
  in this number of milliseconds */
#define AUTO_STOP_INTERVAL 2000
long lastMotorCommand = AUTO_STOP_INTERVAL;

/* Variable initialization */

// A pair of varibles to help parse serial commands (thanks Fergs)
int argu = 0;
int idx = 0;

// Variable to hold an input character
char chr;

// Variable to hold the current single-character command
char cmd_in;

// Character arrays to hold the first and second arguuments
char arguv1[16];
char arguv2[16];

// The arguuments converted to integers
int arg1;
int arg2;

/* Clear the current command parameters */
void resetCommand() {
  cmd_in = 0;
  memset(arguv1, 0, 16);
  memset(arguv2, 0, 16);
  arg1 = 0;
  arg2 = 0;
  argu = 0;
  idx = 0;
}

/* Run a command.  Commands are defined in commands.h */
void runCommand() {
  arg1 = atoi(arguv1);
  arg2 = atoi(arguv2);

  switch (cmd_in) {
    case GET_BAUDRATE:
      Serial.println(BAUDRATE);
      break;
    case PING:
      Serial.println("0");
      break;
    case READ_ENCODERS:
      Serial.print(readEncoder(LEFT));
      Serial.print(" ");
      Serial.println(readEncoder(RIGHT));
      break;
    case RESET_ENCODERS:
      resetEncoders();
      resetPID();
      Serial.println("OK");
      break;
    case MOTOR_SPEEDS:
      /* Reset the auto stop timer */
      lastMotorCommand = millis();
      if (arg1 == 0 && arg2 == 0) {
        setMotorSpeeds(0, 0);
        resetPID();
        moving = 0;
      }
      else moving = 1;
      // Swap the motor
      leftPID.TargetTicksPerFrame = arg2;
      rightPID.TargetTicksPerFrame = arg1;
      Serial.println("OK");
      break;
    case MOTOR_RAW_PWM:
      /* Reset the auto stop timer */
      lastMotorCommand = millis();
      resetPID();
      moving = 0; // Sneaky way to temporarily disable the PID
      setMotorSpeeds(arg1, arg2);
//      Serial.print("Reply Arg :");
//      Serial.print(arg1);
//      Serial.print(",");
//      Serial.println(arg2);
      Serial.println("OK");
      break;
    default:
      Serial.print("Invalid Command : ");
      Serial.println(cmd_in);
      break;
  }
}

/* Setup function--runs once at startup. */
void setup() {
  Serial.begin(BAUDRATE);

  // Initialize the motor controller if used */
  initEncoder();
  initMotorController();
  resetPID();
}

/* Enter the main loop.  Read and parse input from the serial port
   and run any valid commands. Run a PID calculation at the targuet
   interval and check for auto-stop conditions.
*/

void loop() {
  
  while (Serial.available() > 0) {
    // Read the next character
    chr = Serial.read();
    // Terminate a command with a CR
    if (chr == 13) {
      if (argu == 1) arguv1[idx] = NULL;
      else if (argu == 2) arguv2[idx] = NULL;
      
      runCommand();
      resetCommand();
    }
    // Use spaces to delimit parts of the command
    else if (chr == ' ') {
      // Step through the arguuments
      if (argu == 0) argu = 1;
      else if (argu == 1)  {
        arguv1[idx] = NULL;
        argu = 2;
        idx = 0;
      }
      continue;
    }
    else {
      if (argu == 0) {
        // The first argu is the single-letter command
        cmd_in = chr;
      }
      else if (argu == 1) {
        // Subsequent arguuments can be more than one character
        arguv1[idx] = chr;
        idx++;
      }
      else if (argu == 2) {
        arguv2[idx] = chr;
        idx++;
      }
    }
  }

  // If we are using base control, run a PID calculation at the appropriate intervals
  if (millis() > nextPID) {
    updatePID();
    nextPID += PID_INTERVAL;
  }

  // Check to see if we have exceeded the auto-stop interval
  if ((millis() - lastMotorCommand) > AUTO_STOP_INTERVAL) {
    setMotorSpeeds(0, 0);
    moving = 0;
  }
}
