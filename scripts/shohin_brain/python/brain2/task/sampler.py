# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

class Sampler(object):
    """
    Draw a sample from a set of options.
    """
    def __init__(self):
        pass

    def __call__(self, domain, world_state, actor, goal, attempt=0):
        raise NotImplementedError('you should fill this in')

class LookupTableSampler(object):
    """
    Draw a sample from the lookup table.
    """
    def __init__(self, table, sequence_idx=0, goal_heuristic=None):
        self.table = table
        self.goal_heuristic = goal_heuristic
        self.sequence_idx = sequence_idx

    def __call__(self, domain, world_state, actor, goal, attempt=0):
        """
        This one chooses the next way to pick up an object.
        """
        pass


