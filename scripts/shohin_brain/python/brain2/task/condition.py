# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import numpy as np


class Condition(object):

    """
    We evaluate a condition to determine a particular variable in our logical
    states. Conditions may describe either a relation between two entities or a
    quality of a single entitiy.
    """

    def __init__(self, domain, name, function, from_entities, to_entities):
        self.domain = domain
        self.name = name
        self.function = function
        self.from_entities = from_entities
        self.to_entities = to_entities

    def get_name(self, e1, e2=None):
        if e2 is None:
            return "{}({})".format(self.name, e1)
        else:
            return "{}({}, {})".format(self.name, e1, e2)

    def apply(self, world_state, idx):
        """
        Run through all templated situations in which we want to apply this
        condition. Update current world state to have this vector.
        """

        for e1 in self.from_entities:
            if self.to_entities is None:

                # Evaluate the condition
                result = self.function(world_state, e1)

                # Update logical state
                world_state.logical_state[idx] = result
                idx += 1

                if np.any(np.isnan(world_state.logical_state)):
                    print(idx, self.name, e1, result)
                    print(world_state.logical_state)
                    raise RuntimeError('nans are not allowed')
            else:
                for e2 in self.to_entities:
                    # Do not add relations between object and itself that does not
                    # make any sense
                    if e1 == e2:
                        continue

                    # Setup things here
                    result = self.function(world_state, e1, e2)

                    # Update logical state
                    world_state.logical_state[idx] = result
                    idx += 1

                    if np.any(np.isnan(world_state.logical_state)):
                        print(idx, self.name, e1, e2, result)
                        print(world_state.logical_state)
                        raise RuntimeError('nans are not allowed')

        return idx
