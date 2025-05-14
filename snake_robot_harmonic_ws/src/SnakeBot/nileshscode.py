import mujoco
import mujoco.viewer
import time 
MJCF_PATH = "/home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/src/SnakeBot/scene.xml"

# Load the model
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)






# Run MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        
        
        mujoco.mj_step(model, data)
        viewer.sync()
        t += 0.01
        time.sleep(0.01)

