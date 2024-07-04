import time
import mujoco
import mujoco.viewer
import numpy as np
import csv

model = mujoco.MjModel.from_xml_path('example.xml')
data = mujoco.MjData(model)


def all_conf(i0, i1, i2, i3):
    if i3 > 360:
        i3 = 0
        i2 += 25
    if i2 > 360:
        i2 = 0
        i1 += 25
    if i1 > 360:
        i1 = 0
        i0 += 25
    if i0 > 360:
        exit(0)
    i3 += 25
    return i0, i1, i2, i3


csv_filename = 'qfrc_inverse_data.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    i0, i1, i2, i3 = 0, 0, 0, 0
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        i0, i1, i2, i3 = all_conf(i0, i1, i2, i3)
        data.qpos = np.deg2rad([i0, i1, i2, i3])
        mujoco.mj_step(model, data)
        mujoco.mj_inverse(model, data)
        if data.contact:
            print('contact')
        else:
            with open(csv_filename, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(data.qfrc_inverse)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
