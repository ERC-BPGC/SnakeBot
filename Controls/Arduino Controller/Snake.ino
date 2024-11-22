#include <Servo.h>

// Pins
int pins1[5]= {2,3,4,5,6};
int pins2[5]= {9,10,11,12,13};
int x=0;

// Variables
Servo s1[5];
Servo s2[5];

int n=5;
float angles[5][2];
float angles_test[5][2]= {{10,10},{0,0},{0,0},{0,0},{0,0}};
float angles_straight[5][2];
float angles_left[5][2];
float angles_right[5][2];

float off[5][2] = {{90,90},{90,90},{90,90},{90,90},{90,90}};

void setup() {
  for(int i=0; i<n; i++){
    s1[i].attach(pins1[i]);
    s2[i].attach(pins2[i]);
  }

  Serial.begin(9600);
}

void loop() {

  // Input real time calculations (when not in testing phase)
  // Serial.print("Enter...");
  // for(int i=0; i<n; i++){
  //   for(int j=0; j<2; j++){
  //     while(Serial.available()==0);
  //     angles[i][j]=Serial.parseFloat()-90.0;
  //   }
  // }

  // Serial.print("Moving...");
  for(int i=0; i<n; i++){
    s1[i].write(angles_test[(i+x)%n][0]+off[i][0]);
    s2[i].write(angles_test[(i+x)%n][1]+off[i][1]);
    // delay(250);
  }
  x++;
}
