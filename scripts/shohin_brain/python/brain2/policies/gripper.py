# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np

import brain2.utils.status as status

from brain2.task.action import Policy


class BlockingCloseGripper(Policy):
    """ Simple gripper close policy """

    def enter(self, world_state, actor, goal):
        return status.SUCCESS

    def exit(self, world_state, actor, goal):
        return status.SUCCESS

    def step(self, world_state, actor, goal):
        world_state[actor].ctrl.close_gripper(wait=True)
        return status.SUCCESS


class BlockingOpenGripper(Policy):
    """ Simple gripper open policy """

    def enter(self, world_state, actor, goal):
        return status.SUCCESS

    def exit(self, world_state, actor, goal):
        return status.SUCCESS

    def step(self, world_state, actor, goal):
        world_state[actor].ctrl.open_gripper(wait=True)
        world_state[actor].detach()
        return status.SUCCESS

