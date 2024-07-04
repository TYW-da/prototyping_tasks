import time
import mujoco
import mujoco.viewer
import numpy as np
import csv

# Define parameters
amplitude = 0.04  # 4 cm converted to meters
max_velocity = 0.4  # m/s

# Initialize Mujoco model and data
model = mujoco.MjModel.from_xml_path('example.xml')
data = mujoco.MjData(model)

# Calculate trajectory points
timestep = model.opt.timestep
num_points = int(2 * amplitude / max_velocity / timestep) + 1
time_points = np.linspace(0, 2 * np.pi, num_points)
x_trajectory = amplitude * np.sin(time_points)
y_trajectory = amplitude * np.cos(time_points)

# Find the site ID of the target site
target_site_id = model.site_name2id('target')

# Main simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    for i in range(len(time_points)):
        # Set desired position in cartesian space
        data.site_xpos[target_site_id] = [x_trajectory[i], y_trajectory[i], 0]

        # Set desired position in joint space (assuming joint index 0 here)
        data.qpos[0] = x_trajectory[i]
        data.qvel[0] = max_velocity * np.cos(time_points[i])  # Velocity in joint space

        # Step forward the simulation
        step_start = time.time()
        mujoco.mj_step(model, data)
        mujoco.mj_inverse(model, data)
        print("Position:", data.qpos[0])
        print("Velocity:", data.qvel[0])
        print("Acceleration:", data.qacc[0])
        print("Cartesian Position:", data.site_xpos[target_site_id])
        print("Motor Torques:", data.qfrc_inverse[0])

        viewer.sync()

        # Control the time step
        time_until_next_step = timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        # Check for simulation timeout
        if time.time() - start >= 30:
            break
