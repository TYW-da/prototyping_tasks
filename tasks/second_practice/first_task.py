import time
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('example.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        data.qpos = np.deg2rad([45, 100, 100])
        data.qvel = [0, 0, 0]
        data.qacc = [0, 0, 0]
        # tau: data.q_qrc_inverse, 4
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)