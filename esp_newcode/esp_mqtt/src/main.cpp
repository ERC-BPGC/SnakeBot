#include <Arduino.h> // REQUIRED for PlatformIO
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include "time.h"

// --- CONFIGURATION ---
const char* ssid = "Snakebot";
const char* password = "12345678";
const char* mqtt_server = "172.20.10.2"; // Updated to your Hotspot IP

const char* THIS_ESP_ID = "ESP_01"; 

const int SERVO_PIN_1 = 18;
const int SERVO_PIN_2 = 19;

// --- MULTITASKING & QUEUE CONFIG ---
#define MAX_QUEUE_SIZE 50 

struct ServoCommand {
  unsigned long long timestamp;
  int angle1;
  int angle2;
  bool isValid;
};

ServoCommand commandQueue[MAX_QUEUE_SIZE];
volatile int queueHead = 0; 
volatile int queueTail = 0; 
volatile int queueCount = 0;
portMUX_TYPE queueMux = portMUX_INITIALIZER_UNLOCKED;

TaskHandle_t NetworkTaskHandle;
TaskHandle_t ServoTaskHandle;

WiFiClient espClient;
PubSubClient client(espClient);
Servo servo1;
Servo servo2;

const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 0;
const int   daylightOffset_sec = 0;

// ================================================================
//  FUNCTION PROTOTYPES (THE FIX)
//  These tell C++ that these functions exist further down
// ================================================================
void mqttCallback(char* topic, byte* payload, unsigned int length);
void reconnectMQTT();
unsigned long long getCurrentMillis();
void networkTask(void * parameter);
void servoTask(void * parameter);

// ================================================================
//  HELPER FUNCTIONS
// ================================================================

unsigned long long getCurrentMillis() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
}

void reconnectMQTT() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    String clientId = "ESP32-" + String(THIS_ESP_ID) + "-";
    clientId += String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("Connected");
      client.subscribe("servos/sync_command");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      delay(2000);
    }
  }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, payload, length);

  if (error) {
    Serial.print("JSON Error: "); Serial.println(error.f_str());
    return;
  }

  if (doc["data"].containsKey(THIS_ESP_ID)) {
    unsigned long long ts = doc["ts"];
    int a1 = doc["data"][THIS_ESP_ID][0];
    int a2 = doc["data"][THIS_ESP_ID][1];

    portENTER_CRITICAL(&queueMux);
    
    int nextTail = (queueTail + 1) % MAX_QUEUE_SIZE;
    if (nextTail != queueHead) {
      commandQueue[queueTail].timestamp = ts;
      commandQueue[queueTail].angle1 = a1;
      commandQueue[queueTail].angle2 = a2;
      commandQueue[queueTail].isValid = true;
      
      queueTail = nextTail;
      queueCount++;
      Serial.printf("Queued cmd for %llu (Total: %d)\n", ts, queueCount);
    } else {
      Serial.println("Queue FULL! Dropping command.");
    }
    
    portEXIT_CRITICAL(&queueMux);
  }
}

// ================================================================
//  TASKS
// ================================================================

void networkTask(void * parameter) {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected.");

  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  struct tm timeinfo;
  Serial.print("Syncing Time");
  while(!getLocalTime(&timeinfo)){
    Serial.print(".");
    delay(100);
  }
  Serial.println("\nTime Synced!");

  client.setServer(mqtt_server, 1883);
  client.setCallback(mqttCallback);

  for (;;) {
    if (!client.connected()) {
      reconnectMQTT();
    }
    client.loop(); 
    delay(10); 
  }
}

void servoTask(void * parameter) {
  servo1.attach(SERVO_PIN_1);
  servo2.attach(SERVO_PIN_2);

  for (;;) {
    unsigned long long currentMs = getCurrentMillis();
    bool shouldExecute = false;
    ServoCommand cmd;

    portENTER_CRITICAL(&queueMux);
    
    if (queueHead != queueTail) { 
      if (currentMs >= commandQueue[queueHead].timestamp) {
        cmd = commandQueue[queueHead];
        shouldExecute = true;
        queueHead = (queueHead + 1) % MAX_QUEUE_SIZE;
        queueCount--;
      }
    }
    
    portEXIT_CRITICAL(&queueMux);

    if (shouldExecute) {
      Serial.printf("Executing Move: %d, %d at %llu\n", cmd.angle1, cmd.angle2, currentMs);
      servo1.write(cmd.angle1);
      servo2.write(cmd.angle2);
    }

    delay(1);
  }
}

// ================================================================
//  MAIN SETUP & LOOP
// ================================================================

void setup() {
  Serial.begin(115200);

  xTaskCreatePinnedToCore(networkTask, "NetworkTask", 10000, NULL, 1, &NetworkTaskHandle, 0);
  xTaskCreatePinnedToCore(servoTask, "ServoTask", 10000, NULL, 1, &ServoTaskHandle, 1);
}

void loop() {
  vTaskDelete(NULL); 
}