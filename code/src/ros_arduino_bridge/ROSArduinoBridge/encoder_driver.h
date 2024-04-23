/* *************************************************************
   Encoder driver function definitions - by James Nugen
   ************************************************************ */
#define LEFT_ENC_PIN_A 32  
#define LEFT_ENC_PIN_B 33  

#define RIGHT_ENC_PIN_A 16  
#define RIGHT_ENC_PIN_B 17 

void initEncoder();
long readEncoder(int i);
void resetEncoder(int i);
void resetEncoders();
