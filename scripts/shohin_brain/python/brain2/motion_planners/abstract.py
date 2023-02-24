# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function


class GoalSampler(object):
    """ This class represents a goal sampler for a particular set of entities. """
    def __init__(self, domain):
        """ Do any early work and generate the set of things we should be saving. """
        self.domain = domain

    def get_sampler(self, world_state):
        """ Return a likely sampler. Compute metrics here. """
        raise NotImplementedError()

    def apply(self, goal):
        """ Apply goal to world state and get another world state. After this the world state should
        be good and the effects should be true, or we failed. """
        raise NotImplementedError()


class GoalConnector(object):
    pass


class MotionPlanner(object):
    """ This class will generate a motion plan to a particular world state. It usually wraps some
    planning logic from the other stuff. """

    def __init__(self, problem):
        """ Store the planning configuration parameters """
        self.problem = problem

    def plan(self, goal):
        """ Return a plan to the provided goal"""
        raise NotImplementedError()
