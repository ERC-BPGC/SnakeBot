import mujoco
import mujoco.viewer
import time 
from pathlib import Path

# Go up one directory from the script, then point to scene.xml
MJCF_PATH = Path(__file__).resolve().parent.parent / "scene.xml"


# Load the model
model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
data = mujoco.MjData(model)






# Run MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0.0
    while viewer.is_running():
        
        
        mujoco.mj_step(model, data)
        viewer.sync()
        t += 0.01
        time.sleep(0.01)

