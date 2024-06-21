import time
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('example.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        mujoco.mj_step(model, data)


        data.qvel(1)
        mujoco.mj_inverse(model,data)

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)