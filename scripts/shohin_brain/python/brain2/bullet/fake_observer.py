# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import pybullet as pb
import numpy as np

import brain2.utils.transformations as tra
import brain2.robot.entity as entity

from brain2.task.world_state import WorldStateObserver

class FakeObserver(WorldStateObserver):
    """ Fake observer that creates world states out of the given PyBullet interface. """

    def __init__(self, domain, iface=None):
        super(FakeObserver, self).__init__(domain)

        # Store the pybullet interface
        if iface is None:
            # Assume that the domain owns one because it really should
            iface = domain.iface
        self.iface = iface
        self.seq = 0

    def update(self, entities=None):
        """ Simple update function. Loops over all objects, seeing if they are visible from the
        robot's cameras, and moves it around. """

        # Assume we wanted to update everything
        if entities is None:
            entities = self.current_state.keys()

        self.seq += 1
        self.current_state.time = self.seq

        # Update a full set of entities here
        for name in entities:
            state = self.current_state.entities[name]
            state.observed = True
            state.updated = True
            obj = state.ref
            if obj is None:
                entity.pose = state.surface.update(matrix=True)
            elif obj.entity_type == entity.OBJECT:
                pos, rot = pb.getBasePositionAndOrientation(obj.id)
                pose = tra.quaternion_matrix(rot)
                pose[:3, 3] = pos
                state.set_base_pose(pose)
                if obj.dof > 0:
                    state.set_config(obj.get_joint_positions())

                # Get joint states
                if name == self.domain.robot:
                    # Update both base position and end effector position
                    state.q = obj.get_joint_positions()
                    state.set_ee_pose(obj.get_ee_pose())
                elif name == "kitchen":
                    # Update joint positions etc.
                    state.q = obj.get_joint_positions()

            elif obj.entity_type == entity.JOINT:
                # print("joint entity =", name)
                state.q = obj.get_joint_position()
                state.pose = obj.get_pose(matrix=True)
                #print(state.pose)
                #raw_input()
            else:
                raise RuntimeError('object type ' + str(obj.entity_type) + ' not understood')

        # We can always update the world state
        return True

