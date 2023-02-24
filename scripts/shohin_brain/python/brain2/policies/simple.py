# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

from brain2.task.action import Policy

from brain2.policies.planning import LinearPlan
from brain2.policies.planning import JointSpaceLinearPlan


class CartesianMotionPolicy(Policy):
    """ Move to a particular end position. Doesn't quite matter what it is."""

    def __init__(self, default_q = None):
        """ Doesn't really need to do anything """
        self.default_q = default_q

    def enter(self, world_state, actor, goal):
        return status.SUCCESS

    def exit(self, world_state, actor, goal):
        return status.SUCCESS

    def step(self, world_state, actor, goal):
        T = None
        actor.ctrl.go_local(T)
        return status.SUCCESS
