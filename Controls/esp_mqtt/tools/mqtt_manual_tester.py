import time
import json
import paho.mqtt.client as mqtt

# --- CONFIGURATION ---
MQTT_BROKER = "172.20.10.2"  # CHANGE TO YOUR BROKER IP
MQTT_PORT = 1883
MQTT_TOPIC = "servos/sync_command"

# How far in the future to schedule the FIRST move (seconds)
INITIAL_DELAY = 0.5 

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"✓ Connected successfully to {MQTT_BROKER}")
    else:
        print(f"✗ Connection failed with code: {reason_code}")

def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    print(f"Disconnected with reason code: {reason_code}")

# Use CallbackAPIVersion.VERSION2 to fix deprecation warning
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_disconnect = on_disconnect

# Try to connect with better error handling
print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
except Exception as e:
    print(f"✗ Failed to connect: {e}")
    print("\nTroubleshooting steps:")
    print("1. Verify the MQTT broker is running")
    print("2. Check if 172.20.10.2 is the correct IP address")
    print("3. Ensure your firewall allows port 1883")
    print("4. Try pinging the broker: ping 172.20.10.2")
    exit(1)

# Give it a moment to connect
time.sleep(1)

def send_movement(esp1_angles, esp2_angles, delay_offset=0):
    """
    esp1_angles: [servo1, servo2]
    esp2_angles: [servo1, servo2]
    delay_offset: seconds from NOW to execute
    """
    current_time = time.time()
    target_timestamp = int((current_time + INITIAL_DELAY + delay_offset) * 1000)

    payload = {
        "ts": target_timestamp,
        "data": {
            "ESP_01": esp1_angles,
            "ESP_02": esp2_angles
        }
    }
    
    result = client.publish(MQTT_TOPIC, json.dumps(payload), qos=0)
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"✓ Sent: {payload['data']} for Time: {target_timestamp}")
    else:
        print(f"✗ Failed to send message")

try:
    while True:
        print("\n" + "="*50)
        print("1. Send Immediate Move")
        print("2. Send Sequence (3 moves)")
        print("3. Exit")
        print("="*50)
        choice = input("Select: ")

        if choice == '1':
            try:
                e1s1 = int(input("ESP1 S1 (0-180): "))
                e1s2 = int(input("ESP1 S2 (0-180): "))
                e2s1 = int(input("ESP2 S1 (0-180): "))
                e2s2 = int(input("ESP2 S2 (0-180): "))
                send_movement([e1s1, e1s2], [e2s1, e2s2])
            except ValueError:
                print("✗ Please enter valid numbers")
            
        elif choice == '2':
            print("\nSending 3 moves spaced 1 second apart...")
            
            # Move 1: t + 0.5s
            send_movement([0, 0], [0, 0], delay_offset=0) 
            time.sleep(0.1)  # Small delay between publishes
            
            # Move 2: t + 1.5s
            send_movement([90, 90], [90, 90], delay_offset=1.0) 
            time.sleep(0.1)
            
            # Move 3: t + 2.5s
            send_movement([180, 180], [180, 180], delay_offset=2.0)
            print("✓ All moves queued")
            
        elif choice == '3':
            break
        else:
            print("✗ Invalid choice")

except KeyboardInterrupt:
    print("\n\nShutting down...")
finally:
    client.loop_stop()
    client.disconnect()
    print("✓ Disconnected")