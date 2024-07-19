from math import sqrt

import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import numpy as np

xml_path = 'example.xml'
sim_end = 2

q0_init = [0, 0, 0, 0]

q0_end = [4, np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]
q1_end = [-4, np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]
q2_end = [0, -4, np.deg2rad(0), np.deg2rad(0)]
q3_end = [0, 4, np.deg2rad(0), np.deg2rad(0)]

t_init = 0
t_end = 4

t = []
qact0 = []
qref0 = []
qact1 = []
qref1 = []
qact2 = []
qref2 = []
qact3 = []
qref3 = []


def generate_trajectory(t0, tf, q0, qf):
    sign = -1 if qf < q0 else 1
    T = sign * 15 / 8 * sqrt(abs(qf - q0)) / 0.3

    a0 = 0

    a1 = 0

    a2 = 0

    a3 = 0 if T == 0 else 10 / T ** 3

    a4 = 0 if T == 0 else 15 / T ** 4

    a5 = 0 if T == 0 else 6 / T ** 5

    return a0, a1, a2, a3, a4, a5


def init_controller(q_init, q_end):
    global a_jnt0, a_jnt1, a_jnt2, a_jnt3

    a_jnt0 = generate_trajectory(
        t_init, t_end, q_init[0], q_end[0])

    a_jnt1 = generate_trajectory(
        t_init, t_end, q_init[1], q_end[1])

    a_jnt2 = generate_trajectory(
        t_init, t_end, q_init[2], q_end[2])

    a_jnt3 = generate_trajectory(
        t_init, t_end, q_init[3], q_end[3])


def controller(model, data):
    global a_jnt0, a_jnt1, a_jnt2, a_jnt3

    time = data.time

    if (time > t_end):
        time = t_end

    if (time < t_init):
        time = t_init

    q_ref0 = a_jnt0[0] + a_jnt0[1] * time + \
             a_jnt0[2] * (time ** 2) + a_jnt0[3] * (time ** 3)

    qdot_ref0 = a_jnt0[1] + 2 * a_jnt0[2] * \
                time + 3 * a_jnt0[3] * (time ** 2)

    q_ref1 = a_jnt1[0] + a_jnt1[1] * time + \
             a_jnt1[2] * (time ** 2) + a_jnt1[3] * (time ** 3)

    qdot_ref1 = a_jnt1[1] + 2 * a_jnt1[2] * \
                time + 3 * a_jnt1[3] * (time ** 2)

    q_ref2 = a_jnt2[0] + a_jnt2[1] * time + \
             a_jnt2[2] * (time ** 2) + a_jnt2[3] * (time ** 3)

    qdot_ref2 = a_jnt2[1] + 2 * a_jnt2[2] * \
                time + 3 * a_jnt2[3] * (time ** 2)

    q_ref3 = a_jnt3[0] + a_jnt3[1] * time + \
             a_jnt3[2] * (time ** 2) + a_jnt3[3] * (time ** 3)

    qdot_ref3 = a_jnt3[1] + 2 * a_jnt3[2] * \
                time + 3 * a_jnt3[3] * (time ** 2)

    M = np.zeros((4, 4))
    mj.mj_fullM(model, M, data.qM)
    f0 = data.qfrc_bias[0]
    f1 = data.qfrc_bias[1]
    f2 = data.qfrc_bias[2]
    f3 = data.qfrc_bias[3]
    f = np.array([f0, f1, f2, f3])

    kp = 500
    kd = 2 * np.sqrt(kp)
    pd_0 = -kp * (data.qpos[0] - q_ref0) - kd * (data.qvel[0] - qdot_ref0)
    pd_1 = -kp * (data.qpos[1] - q_ref1) - kd * (data.qvel[1] - qdot_ref1)
    pd_2 = -kp * (data.qpos[2] - q_ref2) - kd * (data.qvel[2] - qdot_ref2)
    pd_3 = -kp * (data.qpos[3] - q_ref3) - kd * (data.qvel[3] - qdot_ref3)
    pd_control = np.array([pd_0, pd_1, pd_2, pd_3])
    tau_M_pd_control = np.matmul(M, pd_control)
    tau = np.add(tau_M_pd_control, f)
    data.ctrl[0] = tau[0]
    data.ctrl[1] = tau[1]
    data.ctrl[2] = tau[2]
    data.ctrl[3] = tau[3]

    t.append(data.time)
    qact0.append(data.qpos[0])
    qref0.append(q_ref0)
    qact1.append(data.qpos[1])
    qref1.append(q_ref1)
    qact2.append(data.qpos[2])
    qref2.append(q_ref2)
    qact3.append(data.qpos[3])
    qref3.append(q_ref3)

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cam.azimuth = 0
cam.elevation = 4
cam.distance = 1
cam.lookat = np.array([0.0, 0.1, 0.1])

data.qpos[0] = q0_init[0]
data.qpos[1] = q0_init[1]
data.qpos[2] = q0_init[2]
data.qpos[3] = q0_init[3]

init_controller(q0_init, q0_end)

mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while data.time - time_prev < 1.0 / 60.0:
        mj.mj_step(model, data)

    if data.time >= sim_end:
        if sim_end > t_end:
            init_controller(q0_init, q1_end)
        if sim_end > t_end * 2:
            init_controller(q0_init, q2_end)
        if sim_end > t_end * 3:
            init_controller(q0_init, q3_end)
        if sim_end > t_end * 4:
            plt.figure(1)
            plt.subplot(4, 1, 1)
            plt.plot(t, np.subtract(qref0, qact0), 'k')
            plt.ylabel('error position joint 0')
            plt.subplot(4, 1, 2)
            plt.plot(t, np.subtract(qref1, qact1), 'k')
            plt.ylabel('error position joint 1')
            plt.subplot(4, 1, 3)
            plt.plot(t, np.subtract(qref2, qact2), 'k')
            plt.ylabel('error position joint 2')
            plt.subplot(4, 1, 4)
            plt.plot(t, np.subtract(qref3, qact3), 'k')
            plt.ylabel('error position joint 3')
            plt.show(block=False)
            plt.pause(10)
            plt.close()
            break
        mj.set_mjcb_control(controller)
        sim_end += t_end

    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)

    glfw.poll_events()

glfw.terminate()
