# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

from brain2.robot.domain import RobotDomainDefinition


class RelationDomainDefinition(RobotDomainDefinition):
    """
    This captures information about the robot and where things are getting put
    down in relation to one another.
    """

    def _get_objs(self, control, num_objs=4):
        objs = {"robot": {"control": robot_control},
                "table": {},
                "drawer": {},
                }
        self.drawers = ["drawer"]
        self.surfaces = ["table"]
        self.moveable = []
        # Create a set of objects for each entity we might be dealing with
        for n in range(num_objs):
            name = "obj%02d" % n
            objs[name] = {"idx:", n}
            self.moveable.append(name)
        return objs
    
    def __init__(self, iface, predictor=None, robot_control=None, hands=False, ik_solver=None):
        super(CartBlocksDomainDefinition, self).__init__(iface,
                                                         self._get_objs(robot_control),
                                                         ik_solver,
                                                         self._get_hands(hands),
                                                         add_default_predicates=False)
        # Store reference to the predictor that we need here
        self.predictor = predictor


    def init_actions(self):
        """ Create the full set of predicate actions. """
        # For each predicate plus diagonal
        pass
