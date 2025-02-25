#include <Servo.h>
const int numsegments = 2;// can support upto 6 segments
Servo myServoH[numsegments];
Servo myServoV[numsegments];
int horizontalpins[] = {2, 3, 4, 5, 6, 7};
int verticalpins[] = {8, 9, 10, 11, 12, 13};
void setup() {
    Serial.begin(115200);  // Start serial communication
    for(int i =0; i <numsegments;i++ )
    {
      myServoH[i].attach(horizontalpins[i]);
      myServoV[i].attach(verticalpins[i]);
      myServoH[i].write(90);    
      myServoV[i].write(90);  
    }

    Serial.println("Enter angle pairs (e.g., 121/130,140/150):");
}

void loop() {
  /*
  NOTE: 
  I Tried doing this same thing through strok() and strtol(), but for some reason it wasnt properly moving along the string received 
  Hence currently achieving the same thing using pointers
  */
    // the serial input recieved should be of format "horizontal1/vertical1,horizontal2/vertical2,..,horizontali/verticali\n" and so on then a '\n'  at the end of each set of angles
    if (Serial.available()) {
        String receivedtemp = Serial.readStringUntil('\n'); // Read input line
        Serial.println("Received: " + receivedtemp); // Debug print

        char received[100];
        receivedtemp.toCharArray(received, 100);  // Convert String to char array

        char* ptr = received;  // Pointer to iterate through received
        int i = 0;
        while (*ptr) {
            // Get horizontal value
            char* horizontal = ptr;
            while (*ptr && *ptr != '/') ptr++;  // Move to '/'
            if (*ptr) *ptr++ = '\0';  // Replace '/' with '\0' and move ahead

            // Get vertical value
            char* vertical = ptr;
            while (*ptr && *ptr != ',') ptr++;  // Move to ','
            if (*ptr) *ptr++ = '\0';  // Replace ',' with '\0' and move ahead

            if (*horizontal && *vertical) {
                Serial.print("Horizontal: ");
                Serial.print(horizontal);
                Serial.print("  Vertical: ");
                Serial.println(vertical);

            myServoH[i].write(atoi(horizontal));
            myServoV[i].write(atoi(vertical));
            i++;
            }
        }
    }
}
