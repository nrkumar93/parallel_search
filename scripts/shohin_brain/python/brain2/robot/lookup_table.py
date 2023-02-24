# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

class LookupTable(object):
    """
    This is the base class for a lookup table for sampling grasps or other
    continuous parameterizations of symbolic actions. These tables are checked
    at runtime (and possibly at planning time) in order to determine if actions
    can be executed.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, ws, obj, actor=None):
        """
        Right now we expect this to return a batch of possible options, where
        each option is a list of poses that constitute a possible grasp. In
        other words, this should return a list of lists.
        """
        raise NotImplementedError('must implement lookup(world_state, obj)')

    def has(self, ws, obj):
        raise NotImplementedError(
            'must implement lookup.has(world_state, obj)')


class BasicLookupTable(LookupTable):
    """
    Create a simple class for lookup tables, so that we can override it to get
    more interesting types of behaviors. This one takes in a dictionary that
    contains all the grasps and poses worth considering.
    """

    def __init__(self, table, by_type=True, unroll=False,):
        self.table = table
        if unroll:
            # This is a table filled with single grasps that needs to be
            # pre-processed to be in the standard, list-based form.
            for k, options in self.table.items():
                new_options = []
                if isinstance(options, list):
                    for grasp in options:
                        # Only a single entry in the sequence
                        new_options.append([grasp])
                else:
                    new_options.append([options])
                self.table[k] = new_options
            if not self.table[k][0][0].shape == (4, 4):
                print(self.table[k])
                print(self.table[k][0])
                print(self.table[k][0][0])
                raise RuntimeError('invalid size')
        self.by_type = by_type

    def __call__(self, ws, obj, actor=None):

        if not self.has(ws, obj):
            return None

        if self.by_type:
            return self.table[ws[obj].obj_type]
        else:
            return self.table[obj]

    def has(self, ws, obj):
        if self.by_type:
            return ws[obj].obj_type in self.table
        else:
            return obj in self.table
