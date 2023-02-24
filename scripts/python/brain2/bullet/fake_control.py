# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import time

class FakeControl(object):
    """
    Fake control for bullet simulation
    """

    def __init__(self, obj_ref, iface=None):
        self.iface = iface
        self.obj = obj_ref
        self.ik = self.obj.ik_solver

    def visualize_pose(self, T):
        ee_ref = self.obj.ee_ref
        ee_ref.set_pose_matrix(T)
        #input('--- press enter ---')

    def go_local(self, T=None, q=None, wait=False, speed=None):
        if q is not None:
            print(q)
            self.obj.set_joint_positions(q)
        else:
            raise NotImplementedError()

    def forward_kinematics(self, q):
        q0 = self.obj.get_joint_positions()
        self.obj.set_joint_positions(q)
        T = self.obj.get_ee_pose()
        self.obj.set_joint_positions(q0)
        return T

    def visualize_trajectory(self, *args, **kwargs):
        pass

    def execute_joint_trajectory(self, trajectory, timings=None, entity_state=None,
                                 timestep=0.2,
                                 wait_at_end=True):
        for pt in trajectory:
            self.obj.set_joint_positions(pt)
            time.sleep(timestep)

    def close_gripper(self, wait=False, obj_state=None):
        print(self.obj.dof)
        print(self.obj.allowed_collisions)
        if obj_state is None:
            self.obj.close_gripper()
        else:
            self.obj.close_gripper_around(obj_state.ref)

    def open_gripper(self):
        self.obj.open_gripper()

