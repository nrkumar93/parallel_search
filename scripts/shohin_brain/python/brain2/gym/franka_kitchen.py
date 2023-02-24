# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from carbongym import gymapi
from carbongym import gymutil
from carbongym import rlbase

# Local import
from brain2.gym.franka_data import SharedData
from brain2.gym.experiment import Experiment

class FrankaKitchenEnv(rlbase.Environment):
    """ Wraps a cartpole environment instance """

    def __init__(self, shared_data, **base_args):
        super(FrankaKitchenEnv, self).__init__(**base_args)

        # Kitchen pose
        pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(-1., 0.6, 0)
        pose.p = gymapi.Vec3(-1., 1.36, 0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        self.kitchen_handle = self._gym.create_actor(self._envPtr,
                shared_data.kitchen_asset, pose, "indigo", self._envIndex, 0)

        # Table pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(2., 0.375)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        self.table_handle = self._gym.create_actor(self._envPtr,
                shared_data.table_asset, pose, "table", self._envIndex, 0)

        # Frnka position
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(1.25, 0.0, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        self.franka_handle = self._gym.create_actor(self._envPtr,
                shared_data.mobile_asset, pose, "franka", self._envIndex, 1)
                #shared_data.franka_asset, pose, "franka", self._envIndex, 1)

        # get useful handles
        self.franka_hand = self._gym.get_rigid_handle(self._envPtr, "franka", "franka_hand")
        self.franka_dof_props = self._gym.get_actor_dof_properties(self._envPtr, self.franka_handle)
        self.franka_lower_limits = self.franka_dof_props['lower']
        self.franka_upper_limits = self.franka_dof_props['upper']

        # # set joint control type to effort
        # props = self._gym.get_actor_dof_properties(self._envPtr, self.cartpole_handle)
        # props["driveMode"][self.poleJoint] = gymapi.DOF_MODE_EFFORT
        # props["driveMode"][self.sliderJoint] = gymapi.DOF_MODE_EFFORT
        # props['stiffness'].fill(0)
        # props['damping'].fill(0)
        # self._gym.set_actor_dof_properties(self._envPtr, self.cartpole_handle, props)

        # remember initial transforms
        self._initialHandTransform = self._gym.get_rigid_transform(self._envPtr, self.franka_hand)

    def step(self, actions):
        # # apply action as force along x-axis
        # actions = np.clip(actions, -1, 1)
        # actions = 200 * actions
        # force = actions[self._envIndex]
        # self._gym.apply_joint_effort(self._envPtr, self.sliderJoint, force)
        pass

    def num_actions(self):
        """ Six or seven dof control. Probably joint position controlled to be honest. """
        # Control is:
        # 7 dof for the arm
        # 1 dof for the gripper
        # 2 dof for the base
        return 7 + 1 + 2

    def get_null_action(self):
        return np.zeros(self.num_actions())

    def create_shared_data(gym, sim, **kwargs):
        return SharedData(gym, sim, **kwargs)


class FrankaKitchenDefaultTask(rlbase.Task):
    """ Boring default task for a robot to do """
    def __init__(self, envs, **base_args):
        super(FrankaKitchenDefaultTask, self).__init__(envs, **base_args)
        self._envs = envs

        # Set up parameters
        self._numActions = self._envs[0].num_actions()
        self._numEnvs = len(self._envs)
        self._numObs = self.num_observations()

        # buffers for the last step needed for reward calculation
        self.observationBuffer = np.zeros((self._numEnvs, self._numObs)).astype(float)
        self.rewardBuffer = np.zeros((self._numEnvs)).astype(float)
        self.donesBuffer = np.zeros((self._numEnvs)).astype(bool)
        self.actionsBuffer = np.zeros((self._numEnvs, self._numActions))
        self.terminated_envs = np.array([])

    def num_observations(self):
        #  x-coordinate of cart, cart velocty, cosine of pole angle, pole velocity
        return 4

    def fill_observations(self, actions):
        # TODO
        # fill and return observation
        return self.observationBuffer

    def fill_rewards(self, actions):
        # TODO
        #  fill and return rewards
        self.rewardBuffer[:] = 1
        if len(self.terminated_envs) > 0:
            self.rewardBuffer[self.terminated_envs] = np.zeros(np.shape(self.terminated_envs))
        return self.rewardBuffer

    def fill_dones(self, actions):
        # TODO
        #  fill and return dones
        self.donesBuffer = np.zeros((self._numEnvs)).astype(bool)
        if len(self.terminated_envs) > 0:
            self.donesBuffer[self.terminated_envs] = np.ones(np.shape(self.terminated_envs)).astype(bool)
        return self.donesBuffer

    def reset(self, kills):
        # TODO
        # Callback to re-initialize the episode
        pass

if __name__ == "__main__":
    exp = Experiment(FrankaKitchenEnv, FrankaKitchenDefaultTask)

    exp.spin()
    exp.shutdown()
