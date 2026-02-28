'''
new estimation method, sacrifices some accuracy for simpler and faster code
NOTE from here on the angles printed are now in (x,y) and not in (y,x) as in the codes before
'''
import numpy as np
import matplotlib.pyplot as plt
import time 
import math
import json
import paho.mqtt.client as mqtt
from math import atan, sin, cos, sqrt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev
from copy import deepcopy
import keyboard

# --- MQTT CONFIGURATION ---
MQTT_BROKER = "10.130.96.203"  # CHANGE TO YOUR BROKER IP
MQTT_PORT = 1883
MQTT_TOPIC = "servos/sync_command"
INITIAL_DELAY = 5 

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"✓ Connected successfully to {MQTT_BROKER}")
    else:
        print(f"✗ Connection failed with code: {reason_code}")

def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    print(f"Disconnected with reason code: {reason_code}")

def send_movement(calculated_pairs, delay_offset=0):
    """
    Dynamically maps pairs to ESP_01, ESP_02, ESP_03, etc., to preserve all segments.
    """
    current_time = time.time()
    target_timestamp = int((current_time + 4*INITIAL_DELAY + delay_offset) * 1000)

    # Dynamically build the payload based on how many angle pairs were calculated
    data_payload = {}
    for i, pair in enumerate(calculated_pairs):
        esp_key = f"ESP_{i+1:02d}"
        data_payload[esp_key] = pair

    payload = {
        "ts": target_timestamp,
        "data": data_payload
    }
    
    result = client.publish(MQTT_TOPIC, json.dumps(payload), qos=0)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"✗ Failed to send message")

# spline_func should be a function that returns x, y, z given a parameter `u`
def spline_func(u):
    return splev(u, tck)  # Assuming you have tck from splprep

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Initial point on the curve
x0, y0, z0 = 0, 0, 0  # You should choose an appropriate starting point

n=1
iterations =0
answers = []
error_margin=5
accuracylevel =100000
still_not_found =True
colors =['red','blue']

# User inputs for required number of segments, length, amplitudes, wavelength, and error margin
num_segments=5
req_length = 1.14
req_amplitude_z = 0.8
req_amplitude_y = 0
A1=0
num_path_points = 0
R=req_length/2
approximation_circle_dist=0.17

path_points_x = [0,3,4,20]
path_points_y = [0,3,4,20]
frequency = 2

# Convert lists to a numpy array
points = np.array([path_points_x, path_points_y])

# Create a parameterized spline
tck, u = splprep(points, s=0)

# Generate points along the spline
u_new = np.linspace(0, 1, accuracylevel)  
x_new, y_new = splev(u_new, tck)

# Calculate the derivatives (tangent vectors) along the spline
dx, dy = splev(u_new, tck, der=1)

# Normalize the tangent vectors
magnitude = np.sqrt(dx**2 + dy**2)
dx_normalized = -dy / magnitude
dy_normalized = dx / magnitude

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_box_aspect([1, 1, 1])  

# Plot the original spline in 3D (XY plane)
ax.plot(x_new, y_new, np.zeros_like(x_new), label='Original Spline', color='red')

# Initial sine wave plot in 3D
x_sine_wave = A1 + x_new + req_amplitude_y * np.sin(2 * np.pi *frequency * u_new) * dx_normalized
y_sine_wave = A1 + y_new + req_amplitude_y * np.sin(2 * np.pi *frequency * u_new) * dy_normalized
z_wave = req_amplitude_z * np.abs(np.sin(2 * np.pi *frequency * u_new))

sine_wave_line, = ax.plot(x_sine_wave, y_sine_wave, z_wave, label='3D Spline with Sine Waves', color='green')

angles_ground_apparent =[]
angles_ground_real=[]
angles_relative=[]
angles_real =[]
answers_per_iteration =[]
angles_relative_per_iteration=[]
angles_real_per_iteration=[]

offset =0
answers_per_iteration.append([0,0,0])

while(offset<accuracylevel):
    for i in range(offset,len(x_sine_wave)):
        if(sqrt((x_sine_wave[i]-answers_per_iteration[-1][0])**2+(y_sine_wave[i]-answers_per_iteration[-1][1])**2+(z_wave[i]-answers_per_iteration[-1][2])**2)>req_length):
            answers_per_iteration.append([x_sine_wave[i],y_sine_wave[i],z_wave[i]])
        if(len(answers_per_iteration)==num_segments+1):
            for j in range(0,len(answers_per_iteration)-1):
                angles_ground_apparent.append([atan((answers_per_iteration[j+1][1]-answers_per_iteration[j][1])/(answers_per_iteration[j+1][0]-answers_per_iteration[j][0]))*180/np.pi,atan((answers_per_iteration[j+1][2]-answers_per_iteration[j][2])/(answers_per_iteration[j+1][0]-answers_per_iteration[j][0]))*180/np.pi])
                angles_ground_real.append([atan((answers_per_iteration[j+1][1]-answers_per_iteration[j][1])/sqrt((answers_per_iteration[j+1][0]-answers_per_iteration[j][0])**2+(answers_per_iteration[j+1][2]-answers_per_iteration[j][2])**2))*180/np.pi,atan((answers_per_iteration[j+1][2]-answers_per_iteration[j][2])/sqrt((answers_per_iteration[j+1][0]-answers_per_iteration[j][0])**2+(answers_per_iteration[j+1][1]-answers_per_iteration[j][1])**2))*180/np.pi])
                if j:
                    angles_relative_per_iteration.append(tuple([round(180+angles_ground_apparent[j][0]-angles_ground_apparent[j-1][0],3),round(180+angles_ground_apparent[j][1]-angles_ground_apparent[j-1][1],3)]))
                    angles_real_per_iteration.append(tuple([round(180+angles_ground_real[j][0]-angles_ground_real[j-1][0],3),round(180+angles_ground_real[j][1]-angles_ground_real[j-1][1],3)]))
            break
            
    if(len(answers_per_iteration)<num_segments+1):
        break
        
    answers.append(deepcopy(answers_per_iteration))
    angles_real.append(deepcopy(angles_real_per_iteration))
    angles_relative.append(deepcopy(angles_relative_per_iteration))
    answers_per_iteration.clear()
    angles_real_per_iteration.clear()
    angles_ground_apparent.clear()
    angles_ground_real.clear()
    angles_relative_per_iteration.clear()
    offset+=int(accuracylevel/150)
    answers_per_iteration.append([x_sine_wave[offset],y_sine_wave[offset],z_wave[offset]])

max_dist =max(max(path_points_x),max(path_points_y))
ax.set_xlim(0,max_dist)
ax.set_ylim(0,max_dist)
ax.set_zlim(0,max_dist)

line, = ax.plot([], [], [], 'o-', lw=2)

def update(frame):
    data = np.array(answers[frame])  
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    line.set_data(x, y)
    line.set_3d_properties(z)
    return line,

ani = FuncAnimation(fig, update, frames=len(answers), interval=50, blit=False)

plt.legend()
plt.grid(True)
plt.show()

# --- INITIALIZE MQTT CLIENT ---
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_disconnect = on_disconnect

print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
except Exception as e:
    print(f"✗ Failed to connect: {e}")
    exit(1)

time.sleep(1) # Wait for connection to stabilize

# --- MAIN SENDING LOOP ---
tempcount = 1
wantToSend=True

try:
    while(wantToSend):
        for angles_set in angles_real:  
            calculated_pairs = []
            
            for angle_pair in angles_set:  
                '''this modified angle is due to the design of the modules in v2(virtual rolling joint)'''
                angle_1 = atan(R*sin((180-angle_pair[0])*np.pi/180)/(R*cos((180-angle_pair[0])*np.pi/180)-approximation_circle_dist))*180/np.pi
                angle_2 = atan(R*sin((180-angle_pair[1])*np.pi/180)/(R*cos((180-angle_pair[1])*np.pi/180)-approximation_circle_dist))*180/np.pi
                
                # Format exactly as before: min 70, max 110 constraints
                s1 = min(110,max(90-int(angle_1),70))
                s2 = min(110,max(90-int(angle_2),70))
                calculated_pairs.append([s2, s1]) # Note the order swap to match (x,y) -> (servo1, servo2)
            
            # Publish via MQTT instead of Serial
            send_movement(calculated_pairs)
            print(f"Sent angles: {calculated_pairs} - Count: {tempcount}")
            
            time.sleep(0.055)
            tempcount+=1
            
            if(keyboard.is_pressed('p')):
                print("paused")
                time.sleep(2)
                while(True):
                    if(keyboard.is_pressed('p')):
                        print("continuing")
                        time.sleep(2)
                        break
                    if(keyboard.is_pressed('q')):
                        print("quitting")
                        wantToSend=False
                        break
                        
            if(not wantToSend):
                break
                
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    client.loop_stop()
    client.disconnect()
    print("✓ MQTT Disconnected safely.")