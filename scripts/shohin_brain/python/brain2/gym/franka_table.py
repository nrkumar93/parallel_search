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

from brain2.gym.experiment import Experiment
from brain2.gym.franka_data import SharedData

class FrankaTableEnv(rlbase.Environment):
    """ Wraps a cartpole environment instance """

    def __init__(self, shared_data, **base_args):
        super(FrankaTableEnv, self).__init__(**base_args)
        # create cartpole instance
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.4, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        # add franka
        pose.p.y = 0.0
        pose.p.x = 1.25
        self.franka_handle = self._gym.create_actor(self._envPtr,
                shared_data.franka_asset, pose, "franka", self._envIndex, 1)

        # get useful handles
        self.franka_hand = self._gym.get_rigid_handle(self._envPtr, "franka", "franka_hand")
        self.franka_dof_props = self._gym.get_actor_dof_properties(self._envPtr, self.franka_handle)
        self.franka_lower_limits = self.franka_dof_props['lower']
        self.franka_upper_limits = self.franka_dof_props['upper']

        # remember initial transforms
        self._initialHandTransform = self._gym.get_rigid_transform(self._envPtr,
                                                                   self.franka_hand)

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
        return 7 + 1

    def get_null_action(self):
        return np.zeros(self.num_actions())

    def create_shared_data(gym, sim, **kwargs):
        return SharedData(gym, sim, **kwargs)


class FrankaTableDefaultTask(rlbase.Task):
    def __init__(self, envs, **base_args):
        super(FrankaTableDefaultTask, self).__init__(envs, **base_args)
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

        envIndicies = np.arange(self._numEnvs)
        self.terminated_envs = self.check_termination(self.cartPose, self.poleAngle)
        non_terminated_envs = np.delete(np.arange(self._numEnvs), self.terminated_envs)
        if len(non_terminated_envs) > 0:
            self.observationBuffer[non_terminated_envs, 0] = self.cartPose[non_terminated_envs]
            self.observationBuffer[non_terminated_envs, 1] = self.cartVelocity[non_terminated_envs]
            self.observationBuffer[non_terminated_envs, 2] = np.cos(self.poleAngle[non_terminated_envs])
            self.observationBuffer[non_terminated_envs, 3] = self.poleVelocity[non_terminated_envs]
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
    exp = Experiment(FrankaTableEnv, FrankaTableDefaultTask)

    exp.spin()
    exp.shutdown()
