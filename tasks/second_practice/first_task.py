import time
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('example.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        data.xpos = [15, 15, 15]

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)