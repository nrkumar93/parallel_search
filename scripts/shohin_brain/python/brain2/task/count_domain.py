# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

import copy
import numpy as np
from .domain import DomainDefinition
from .world_state import EntityState
from .world_state import WorldStateObserver


class CountPoint(EntityState):

    def __init__(self, pos):
        self.pos = pos

    def copy(self):
        return copy.copy(self)


class CountRobot(CountPoint):

    '''
    Set up the actor to do some stuff. Actor has policies to either increase or
    decrease its current associated state.
    '''

    def __init__(self, *args, **kwargs):
        super(self, CountRobot).__init__(*args, **kwargs)
        self.gripper_closed = False
        self.obj = None


class CountSimObserver(WorldStateObserver):

    '''
    This is an example of how a world state listener would work. It produces
    new world state observations when we call observe().
    '''

    def __init__(self, domain):
        # Takes in a particular domain definition
        super(CountSimObserver, self).__init__(domain)
        self.a = (10 + np.random.random() * 10)
        self.b = (self.a - (2 + np.random.random() * 8))
        self.c = (self.b + (1 + np.random.random() * 8))
        self.robot = (np.random.random() * 10)
        self.gripper_closed = False
        self.gripper_holding = None
        self.obj = None
        self.iter = 0

    def print_info(self):
        self.iter += 1
        print("=== WORLD STATE ===")
        print("a =", self.a)
        print("b =", self.b)
        print("c =", self.c)
        print("robot at", self.robot, "and has", str(self.gripper_holding))

    def update(self, entities):
        '''
        This is our "perception" and it gets the current state of the world
        '''
        if self.gripper_closed:
            self.current_state.entities['robot'].obj = self.gripper_holding
            if self.gripper_holding is not None:
                self.set_obj_position(self.gripper_holding, self.robot)
        else:
            self.current_state.entities['robot'].obj = None
        self.current_state.entities['a'].pos = self.a
        self.current_state.entities['b'].pos = self.b
        self.current_state.entities['c'].pos = self.c
        self.current_state.entities['robot'].pos = self.robot
        self.current_state.entities['robot'].gripper_closed = self.gripper_closed
        return True

    def set_obj_position(self, obj_name, pos):
        if obj_name == 'a':
            self.a = pos
        elif obj_name == 'b':
            self.b = pos
        elif obj_name == 'c':
            self.c = pos


def make_world():
    # This is the actual world manager class
    domain = DomainDefinition()

    # Add the list of entities so we know what to get from the update hook
    domain.add_entity("a", CountPoint, 0)
    domain.add_entity("b", CountPoint, 1)
    domain.add_entity("c", CountPoint, 2)
    domain.add_entity("robot", CountPoint, 3)
    tol = 0.01

    # ----------------------------------
    # CONDITIONS:
    # Divided between properties (single object vs. world) and relations (two
    # objects, plus world).
    # Define our logical world state here
    def over(ws, x, y):
        return ws[x].pos > ws[y].pos + tol

    def under(ws, x, y):
        return ws[x].pos < ws[y].pos - tol

    def at(ws, x, y):
        return abs(ws[x].pos - ws[y].pos) <= tol

    def positive(ws, x):
        return ws[x].pos > 0

    def negative(ws, x):
        return ws[x].pos < 0

    def is_zero(ws, x):
        return ws[x].pos == 0

    def is_holding(ws, x, y):
        return ws[x].obj is not None and ws[x].obj == ws[y]

    def is_free(ws, x):
        return ws[x].obj is None

    def is_held(ws, x):
        return ws["robot"].obj is not None and ws["robot"].obj == x

    # Get the set of actors
    domain.add_relation("is_over", over)
    domain.add_relation("is_under", under)
    domain.add_relation("is_at", at)
    domain.add_relation("is_holding", is_holding, "robot", ["a", "b", "c"])
    domain.add_property("is_positive", positive)
    domain.add_property("is_negative", negative)
    domain.add_property("is_free", is_free, "robot")
    domain.add_property("is_held", is_held, ["a", "b", "c"])

    # High level dynamics
    def apply_pickup(ws, actor, obj):
        # TODO
        raise NotImplementedError()

    def apply_PLACEHOLDER(ws, actor, obj):
        # TODO
        raise NotImplementedError()

    # Control policies
    def pickup(ws, actor, obj):
        # This needs to compute the control and actually send it
        # This policy only works if we are at the right position
        sim.gripper_closed = True
        sim.gripper_holding = obj
        sim.set_obj_position(obj, ws[actor].pos)
        return domain.success

    def release(ws, actor, obj):
        sim.gripper_closed = False
        sim.gripper_holding = None
        return domain.success

    def move_above(ws, actor, obj):
        # move with some amount of noise
        sim.robot += 2 + np.random.randn() * tol
        return domain.success

    def move_below(ws, actor, obj):
        # move with some amount of noise
        sim.robot -= 2 + np.random.randn() * tol
        return domain.success

    def move_to(ws, actor, obj):
        # Compute step and move
        diff = ws[obj].pos - ws[actor].pos
        max_speed = 2
        if diff < -1 * max_speed:
            diff = -1 * max_speed
        elif diff > max_speed:
            diff = max_speed
        # apply dynamics
        sim.robot += diff + np.random.randn() * tol
        return domain.success

    # Operations: these are a set of 2d actions that we can "step"
    domain.add_operator("pickup",
                        preconditions=[
                            ("is_free(robot)", True),
                            ("is_at(robot, {})", True),
                        ],
                        effects=[
                            ("is_free(robot)", False),
                            ("is_held({})", True),
                        ],
                        policy=pickup,
                        actor="robot",
                        to_entities=["a", "b", "c"])
    domain.add_operator("move_above",
                        policy=move_above,
                        preconditions=[("is_under(robot, {})", True)],
                        effects=[
                            ("is_over(robot, {})", True),
                            ("is_under(robot, {})", False)],
                        actor="robot",
                        to_entities=["a", "b", "c"])
    domain.add_operator("move_below",
                        policy=move_below,
                        preconditions=[("is_over(robot, {})", True)],
                        effects=[
                            ("is_over(robot, {})", False),
                            ("is_under(robot, {})", True)],
                        actor="robot",
                        to_entities=["a", "b", "c"])
    domain.add_operator("move_below_with_b",
                        policy=move_below,
                        preconditions=[
                            ("is_over(robot, {})", True),
                            ("is_under(robot, {})", False),
                            ("is_held(b)", True)],
                        effects=[
                            ("is_under(robot, {})", True),
                            ("is_over(robot, {})", False),
                            ("is_under(b, {})", True),
                            ("is_over(b, {})", False), ],
                        actor="robot",
                        to_entities=["a", "c"])
    domain.add_operator("move_to_with_a",
                        policy=move_to,
                        # this actually sends controls to the robot
                        effects=[
                            ("is_at(a, {})", True),
                            ("is_at({}, a)", True),
                            ("is_at(robot, {})", True),
                            ("is_at({}, robot)", True),
                        ],
                        # functor for task planning
                        preconditions=[
                            ("is_at(robot, {})", False),
                            ("is_held(a)", True),
                        ],
                        actor="robot",
                        to_entities=["b", "c"])
    domain.add_operator("move_to",
                        policy=move_to,
                        preconditions=[("is_at(robot, {})", False)],
                        effects=[
                            ("is_at(robot, {})", True),
                            ("is_at({}, robot)", True),
                        ],
                        actor="robot",
                        to_entities=["a", "b", "c"])
    domain.add_operator("release",
                        preconditions=[("is_free(robot)", False)],
                        effects=[("is_free(robot)", True)],
                        policy=release,
                        actor="robot",
                        to_entities=None)
    domain.add_operator("release_at_b",
                        preconditions=[
                            ("is_free(robot)", False),
                            ("is_at(robot, b)", True)],
                        effects=[("is_free(robot)", True)],
                        policy=release,
                        actor="robot",
                        to_entities=None)
    domain.add_operator("release_below_a",
                        preconditions=[
                            ("is_free(robot)", False),
                            ("is_under(robot, a)", True)],
                        effects=[("is_free(robot)", True), ],
                        policy=release,
                        actor="robot",
                        to_entities=None)

    # build all connections
    domain.planning_operators = domain.operators.keys()
    domain.compile()

    # This produces the update hook we'll be using
    sim = CountSimObserver(domain)

    return sim, domain
