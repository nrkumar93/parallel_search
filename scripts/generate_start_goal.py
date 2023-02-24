import os
import mujoco as mp
from mujoco import MjData, MjModel
import mujoco_viewer
from time import sleep
import numpy as np
from numpy import genfromtxt
from abb_fkik_solver import ABBFkIkSolver

print(fkik.computeFK([0,0,0,0,0,0]))
print(fkik.computeIK([0.815,0,0.9615], [0,0,0]))


#### Load mujoco model
model_dir = '../third_party/mujoco-2.3.2/model/abb/irb_1600'
urdf = os.path.join(model_dir, 'irb1600_6_12_generated.urdf')
mjcf_arm = 'irb1600_6_12_shield.xml'
arm_model = MjModel.from_xml_path(os.path.join(model_dir, mjcf_arm))
arm_data = MjData(arm_model)

starts = np.empty((0,arm_model.nq))
goals = np.empty((0,arm_model.nq))

### Volume to do IK
vol = np.array([[[1.2, 1.8], [-1.2, 0], [0.2, 1.2]],
                [[0, 1.2], [0, 0.6], [0.2, 1.2]],
                [[-0.6, 0], [-1.2, 0], [0.2, 1.2]],
                [[0, 1.2], [-1.8, -1.2], [0.2, 1.2]]])

data_size = 500

fkik = ABBFkIkSolver(urdf, base_position=[0,0,0])

def isCollisionFree(q):
    if not ((arm_model.jnt_range[:,0] <= q) and (arm_model.jnt_range[:,1] >= q)):
        return False

    arm_data.qpos[:] = q[:]
    mp.mj_forward(arm_model, arm_data)

    if arm_data.ncon > 0:
        return False
    else:
        return True


def ik(x):
    ### your code here
    q = fkik.computeIK(x)
    return q

while s < data_size:
    sid = 0
    gid = 0
    while 1:
        id = np.random.choice(4, 2)
        if id[0] != id[1]:
            sid = id[0]
            gid = id[1]
            break

    st_seed = np.random.random_sample((3,))
    go_seed = np.random.random_sample((3,))
    # start = np.array([(vol[sid,0,1]-vol[sid,0,0])*st_seed[0]+vol[sid,0,0],
    #                   (vol[sid,1,1]-vol[sid,1,0])*st_seed[1]+vol[sid,1,0],
    #                   (vol[sid,2,1]-vol[sid,2,0])*st_seed[2]+vol[sid,2,0]])
    # goal = np.array([(vol[gid,0,1]-vol[gid,0,0])*go_seed+vol[gid,0,0],
    #                  (vol[gid,1,1]-vol[gid,1,0])*go_seed+vol[gid,1,0],
    #                  (vol[gid,2,1]-vol[gid,2,0])*go_seed+vol[gid,2,0]])
    start = np.array([(vol[sid,:,1]-vol[sid,:,0])*st_seed+vol[sid,:,0]])
    goal = np.array([(vol[gid,:,1]-vol[gid,:,0])*go_seed+vol[gid,:,0]])

    st_q = ik(start)
    go_q = ik(goal)

    if isCollisionFree(st_q) and isCollisionFree(go_q):
        starts = np.append(starts, start, axis=0)
        goals = np.append(goals, goal, axis=0)

    s = s+1

start_file = '../examples/manipulation/resources/shield/starts.txt'
goal_file = '../examples/manipulation/resources/shield/goals.txt'
np.savetxt(start_file, starts, delimiter=',')
np.savetxt(goal_file, goals, delimiter=',')
