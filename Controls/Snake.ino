#include <Servo.h>

// Pins
int pins1[5]= {2,3,4,5,6};
int pins2[5]= {9,10,11,12,13};
int x=0;

// Variables
Servo s1[5];
Servo s2[5];

int n=5;
float angles [5][2];

void setup() {
  for(int i=0; i<5; i++){
    s1[i].attach(pins1[i]);
    s2[i].attach(pins2[i]);
  }

  Serial.begin(9600);
}

void loop() {
  Serial.print("Enter...");
  for(int i=0; i<n; i++){
    for(int j=0; j<2; j++){
      while(Serial.available()==0);
      angles[i][j]=90.0 + Serial.parseFloat();
    }
  }



  
  Serial.print("Moving...");
  for(int i=0; i<n; i++){
    s1[(i+x)%n].write(angles[i][0]);
    s2[(i+x)%n].write(angles[i][1]);
    delay(250);
  }
  x++;
}
