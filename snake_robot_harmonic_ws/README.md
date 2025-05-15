# 🐍 Snake Robot MuJoCo Simulation Walkthrough

This guide provides a step-by-step walkthrough for setting up and running the **MuJoCo-based simulation** of a snake robot, located in your `snake_robot_harmonic_ws` workspace. This simulation is **standalone**—it does **not use ROS or Gazebo**—and is implemented purely using **MuJoCo and Python**.

---

## 📁 Project Structure Overview

```
snake_robot_harmonic_ws/
├── src/
│   └── SnakeBot/
│       ├── meshes/                                # STL files for the robot model
│       ├── ModelV2.xml                            # MuJoCo model of the snake robot
│       ├── scene.xml                              # MuJoCo scene (includes the robot and environment)
│       ├── nileshcode.py                          # Opens the MuJoCo model (no movement)
│       ├── multi_neuron_cpg_control.py            # Controller implementation (based on research)
│       ├── FLEXER_CPG_Sinusoidal_Control_Snake_Robot_Locomotion.py  # Another controller (FLEXER)
│       └── ...                                     # Other scripts and files
├── requirements.txt                                # Python dependencies
```

---

## 🧪 Setting Up the Python Virtual Environment

To ensure a clean and consistent development environment, we recommend using a virtual environment.

### Step 1: Navigate to the workspace

```bash
cd ~/snake_robot_harmonic_ws
```

### Step 2: Create and activate the virtual environment

```bash
python3 -m venv mujoco_env
source mujoco_env/bin/activate
```

> 💡 **Windows users**: Use `mujoco_env\Scripts\activate` instead.

### Step 3: Install required Python packages

Install dependencies listed in `requirements.txt`.  

> ⚠️ **Note**: This file may contain extra packages (e.g. ROS, CUDA, reinforcement learning libraries).  
> You can manually edit it to keep only the relevant packages for MuJoCo if needed.

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Simulation

After activating the virtual environment, you can run any of the following simulation scripts.

> ✅ **Important**: Make sure you're using the Python interpreter from the virtual environment.  
> If you're using an IDE or script launcher, confirm it uses the correct interpreter path.  
> You can also run scripts explicitly using the full path to the environment's Python:

```bash
./mujoco_env/bin/python path/to/script.py
```

### A. Launch MuJoCo Viewer (No Movement)

Opens the simulation in MuJoCo with the robot model loaded.

```bash
python src/SnakeBot/nileshcode.py
```

### B. Run Multi-Neuron CPG Controller

Based on a central pattern generator (CPG) research model.

```bash
python src/SnakeBot/multi_neuron_cpg_control.py
```

### C. Run FLEXER CPG Sinusoidal Controller

Another research-based controller using FLEXER and sinusoidal inputs.

```bash
python src/SnakeBot/FLEXER_CPG_Sinusoidal_Control_Snake_Robot_Locomotion.py
```

---

## 🧩 Notes and Recommendations

- ✅ Ensure that `scene.xml` and `ModelV2.xml` correctly reference STL files from the `meshes/` folder.
- 🧠 The control scripts implement algorithms from published research papers. You can customize the control parameters or logic for further experimentation.
- 💾 Consider creating a cleaned version of `requirements.txt` by running:

  ```bash
  pip freeze > requirements_clean.txt
  ```

---

## 📌 Summary

| Step | Description |
|------|-------------|
| ✅ 1 | Create & activate the `mujoco_env` virtual environment |
| ✅ 2 | Install dependencies from `requirements.txt` |
| ✅ 3 | Run desired control script (`nileshcode`, `multi_neuron_cpg_control`, or `FLEXER...`) |
| ✅ 4 | **Ensure you're using the correct Python interpreter from the virtual environment** (use full path if needed) |
| ✅ 5 | Validate STL and XML references for simulation correctness |

---
