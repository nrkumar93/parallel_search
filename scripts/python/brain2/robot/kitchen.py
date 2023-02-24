# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import print_function

import numpy as np

import brain2.robot.heuristics as heuristics
import brain2.utils.axis as axis
import brain2.utils.status as status
import brain2.utils.pose as pose_tools
import brain2.utils.transformations as tra
from brain2.utils.extend import simple_extend

from brain2.task.domain import DomainDefinition
from brain2.motion_planners.problem import MotionPlanningProblem

# Imported as a placeholder
from brain2.task.action import Policy 

# Brain2 policies and conditions
from brain2.task.execution import PlanExecutionPolicy
from brain2.policies.gripper import BlockingOpenGripper, BlockingCloseGripper
from brain2.policies.planned import RelativeGoPlanned
from brain2.policies.planned import WaypointGoPlanned
from brain2.policies.planned import BlockingGraspObject
from brain2.policies.planned import LiftObject
from brain2.conditions.approach import ApproachRegionCondition

# Task and motion planning
from brain2.robot.samplers import *
from brain2.robot.connectors import *
from brain2.robot.heuristics import *

class StackInfo(object):
    """ Keeps track of whether objects have been stacked based on their poses """

    def __init__(self, objs, size=0.05):
        self.ws = None
        self.t = None
        self.objs = objs
        self.obj_status = {}
        self.obj_on = {}
        self.verbose = False
        self.size = size
        self.min_step = self.size * 0.8
        self.max_step = self.size * 1.2
        self.xy_step = self.size

    def update(self, ws):
        if ws == self.ws and self.t == ws.time:
            return False
        # Actually have to update now
        self.obj_status = {}
        self.obj_on = {}
        # Get the full list of objects from the current world state
        objs = [ws[obj] for obj in self.objs]
        # Loop over the objects and update all this
        for obj in objs:
            self.obj_on[obj.name] = None
        for obj in objs:
            self.obj_status[obj.name] = None
            for obj2 in objs:
                if obj == obj2:
                    continue
                # Get pose of the objcet
                xyz1 = obj.pose[:3, 3]
                xyz2 = obj2.pose[:3, 3]
                dxy = np.linalg.norm(xyz1[:2] - xyz2[:2])
                # obj2 on top of obj1
                dz = xyz2[2] - xyz1[2]
                # print(obj.name, obj2.name, "dxy =", dxy, "dz =", dz)
                check_z = dz > self.min_step and dz < self.max_step
                check_xy = dxy < self.xy_step
                # print("   cz =", check_z, "cxy =", check_xy)
                if check_z and check_xy:
                    self.obj_status[obj.name] = obj2.name
                    self.obj_on[obj2.name] = obj.name
                
        self.ws = ws
        self.t = ws.time
        #self.print_info()
        #input('<>>>>> enter to continue')

    def print_info(self):
        for obj, top in self.obj_status.items():
            print(str(top), "was on top of obj", obj)
    
    def is_clear(self, ws, x):
        """ is this object clear so we can put something on it? """
        # Recompute if necessary
        self.update(ws)
        if self.verbose:
            print("is clear", x, "=", self.obj_status[x] is None)
        return self.obj_status[x] is None

    def is_on_something(self, ws, x):
        self.update(ws)
        if self.verbose:
            print("is on something", x, "=", self.obj_on[x] is None)
        return self.obj_on[x] is not None

    def stacked(self, ws, x, y):
        """ are these two objects stacked or what """
        # Recompute if necessary
        self.update(ws)
        if self.verbose:
            print("stacked on", x, y, "=", self.obj_status[x] == y)
        return self.obj_status[x] == y


def DefineKitchenDomain(domain):
    """
    Creates all the necessary stuff for the kitchen domain definition
    """
    ik = domain.ik_solver 

    # Are we at the home position?
    def at_home(ws, x):
        x = ws[x]
        if x.ref.home_q is None or x.q is None:
            return False
        return np.all(np.abs(x.q[:7] - x.ref.home_q[:7]) < 0.1)

    domain.add_property("at_home", at_home, domain.robot)
    domain.add_property("observed", lambda ws, x: ws[x].observed)
    domain.add_property("stable", lambda ws, x: ws[x].stable)
    domain.add_property("has_anything",
                      lambda ws, x: ws[x].attached_to is not None,
                      domain.robot)
    domain.add_property("is_attached",
                      lambda ws, x: ws[x].attached_to is not None,
                      domain.manipulable_objs + domain.handles)
    domain.add_property("gripper_moving",
                      lambda ws, x: ws[x].gripper_moving,
                      domain.robot)
    domain.add_property("gripper_fully_closed",
                      lambda ws, x: ws[x].gripper_fully_closed,
                      domain.robot)
    domain.add_property("gripper_fully_open",
                      lambda ws, x: ws[x].gripper_fully_open,
                      domain.robot)

    # Create operations for grasping + moving objects around
    domain.add_relation("on_surface",
                      lambda ws, x, y: ws[y].overlap_check(ws[x].ref),
                      domain.manipulable_objs,
                      domain.surfaces+domain.handles)
    #print(domain.manipulable_objs)
    #print(domain.surfaces + domain.handles)
    # Add a predicate for if we've grasped an object or not 
    domain.add_relation("has_obj",  
                      lambda ws, x, y: ws[x].attached_to == y,
                      domain.robot, domain.manipulable_objs+domain.handles)

    # have we generated a motion plan for this?
    # TODO make sure this works properly
    domain.add_relation("has_plan",
                      lambda ws, x, y: ws[x].goal_obj == y,
                      domain.robot, domain.manipulable_objs + domain.hands)

    # TODO use hands
    if len(domain.hands) > 0:
        # add hand predicates
        domain.add_property("hand_over_table", 
                          lambda ws, x: (ws[x].observed
                              and over_table.overlap_check(ws[x].ref)),
                          domain.hands)
        domain.add_property("hand_has_obj",
                          #lambda ws, x: ws[x].obj_pose is not None,
                          lambda ws, x: ws[x].has_obj is not None,
                          domain.hands)

    # Structure creation
    # Create an object that'll track and update all this stuff
    stackable = StackInfo(domain.stackable_objs)
    # Is there anything on top of object?
    top_is_clear = lambda ws, x: stackable.is_clear(ws, x)
    domain.add_property("top_is_clear", top_is_clear, domain.stackable_objs)
    is_on_something = lambda ws, x: stackable.is_on_something(ws, x)
    domain.add_property("on_something", is_on_something, domain.stackable_objs)
    stacked = lambda ws, x, y: stackable.stacked(ws, x, y)
    domain.add_relation("stacked", stacked, domain.stackable_objs)
    aligned_with = lambda ws, x, y: False
    domain.add_relation("aligned_with", aligned_with, domain.stackable_objs)

    # Is the arm currently close to geometry so we should be careful?
    # This is sort of a hack to make placement/approaching/grasping nice
    is_free = lambda ws, x: True
    domain.add_property("is_free", is_free, domain.robot)
    has_goal = lambda ws, x: ws[x].goal_pose is not None
    domain.add_property("has_goal", has_goal, domain.robot)

    #near = lambda ws, x, y: False
    #domain.add_relation("near", near, domain.robot, domain.surfaces + domain.handles
    #                    + domain.manipulable_objs)
    is_open = lambda ws, x: ws[x].q > ws[x].ref.max * 0.95
    domain.add_property("is_open", is_open, domain.handles)
    is_closed = lambda ws, x: ws[x].q <= ws[x].ref.min * 0.01
    domain.add_property("is_closed", is_closed, domain.handles)

    # Planning conditions
    hand_appr_dist = -0.2 # How far away we go to stand-off from the hand
    approach_region = ApproachRegionCondition(approach_distance=2*hand_appr_dist,
                                              approach_direction=axis.Z,
                                              verbose=False,
                                              slope=10.,
                                              pos_tol=2e-2,
                                              max_pos_tol=5e-2,
                                              theta_tol=np.radians(30))
    domain.add_relation("in_approach_region", approach_region, domain.robot,
                        domain.handles+domain.manipulable_objs)

    # ---------------------------------------------------
    # Create operators for the kitchen domain
    # ---------------------------------------------------
    actor = domain.robot
     # Just go home. Can be interrupted.
    go_home = WaypointGoPlanned(step_size=0.25)
    # NOTE: this operator doesn't make sense.
    domain.add_operator("observe_obj",
                        policy=go_home,
                        preconditions=[("observed({})", False)],
                        effects=[("observed({})", True)],
                        task_planning=True,
                        to_entities=domain.manipulable_objs)

    # Approach object
    plan_approach = Policy() # placeholder
    domain.add_operator("approach_obj", policy=plan_approach,
                        preconditions=[
                            ("top_is_clear({})", True),
                            ("in_approach_region(%s, {})" % actor, False),
                            ("observed({})", True),
                            ("has_anything(%s)" % domain.robot, False),
                            ("is_free(%s)" % domain.robot, True),
                            #("near(%s, {})" % (domain.robot), True),
                        ],
                        effects=[
                            ("in_approach_region(%s, {})" % actor, True),
                            ("is_free(%s)" % domain.robot, False),
                            ("has_goal(%s)" % domain.robot, True),
                        ],
                        task_planning=True,
                        to_entities=domain.manipulable_objs+domain.handles,
                        subgoal_sampler=DiscreteGoalSampler("grasp",
                                                            standoff=0.1,
                                                            cost=BasicCost(),
                                                            attach=status.NO_CHANGE),
                        subgoal_connector=RRTConnector(),
                        )

    # Cage object + grasp it
    # This is a blocking policy so that we don't need to worry about anything interfering for now
    grasp_obj = BlockingGraspObject(step_size=0.05, retreat=True)
    domain.add_operator("grasp_obj", policy=grasp_obj,
                        preconditions=[
                            ("top_is_clear({})", True),
                            ("in_approach_region(%s, {})" % actor, True),
                            ("has_anything(%s)" % actor, False),
                            ("has_goal(%s)" % domain.robot, True),
                            ("on_something({})", False),
                        ],
                        effects=[
                            ("has_anything(%s)" % actor, True),
                            ("has_obj(%s, {})" % (domain.robot), True),
                            ("in_approach_region(%s, {})" % actor, False),
                            ("is_free(%s)" % domain.robot, True),
                            ("has_goal(%s)" % domain.robot, False),
                        ],
                        task_planning=True,
                        planning_cost=0,
                        to_entities=domain.manipulable_objs+domain.handles,
                        subgoal_sampler=AttachSampler(status.ATTACH))

    # TODO: I wish we did not need actions like this one; I think they make our life a lot more
    # complex for no good reason at all.
    for obj in domain.stackable_objs:
        domain.add_operator("grasp_obj_from_%s" % obj, policy=grasp_obj,
                            preconditions=[
                                ("top_is_clear({})", True),
                                ("in_approach_region(%s, {})" % actor, True),
                                ("has_anything(%s)" % actor, False),
                                ("on_something({})", True),
                                ("has_goal(%s)" % domain.robot, True),
                                ("stacked(%s, {})" % obj, True),
                            ],
                            effects=[
                                ("has_anything(%s)" % actor, True),
                                ("has_obj(%s, {})" % (domain.robot), True),
                                ("in_approach_region(%s, {})" % actor, False),
                                ("is_free(%s)" % domain.robot, True),
                                ("has_goal(%s)" % domain.robot, False),
                                ("top_is_clear(%s)" % obj, True),
                                ("stacked(%s, {})" % obj, False),
                                ("on_something({})", False),
                            ],
                            task_planning=True,
                            planning_cost=0,
                            to_entities=[o for o in domain.stackable_objs if o != obj], #+domain.handles,
                            subgoal_sampler=AttachSampler(status.ATTACH))


    # Define pickup operator
    for surface in domain.surfaces + domain.handles:
        # Lift object up off of a surface. Needs to work for ALL surfaces.
        lift_obj = LiftObject(step_size=0.05)
        lift_obj_name = "lift_obj_from_%s" % surface
        domain.add_operator(lift_obj_name, policy=lift_obj,
                            preconditions=[
                                ("has_obj(%s, {})" % (domain.robot), True),
                                ("on_surface({}, %s)" % surface, True),
                            ],
                            effects=[
                                ("on_surface({}, %s)" % surface, False),
                                ("is_free(%s)" % domain.robot, True),
                                ("has_goal(%s)" % domain.robot, False),
                                #("on_any_surface({})", False),
                            ],
                            task_planning=True,
                            to_entities=domain.manipulable_objs,
                            subgoal_sampler=LinearSampler(axis.Z, 0.1, in_world_frame=True),
                            subgoal_connector=LinearConnector(axis.Z),)

    # Open gripper if it was empty.
    open_gripper = BlockingOpenGripper()
    domain.add_operator("open_gripper_grasp_failed", policy=open_gripper,
                        preconditions=[
                            ("has_anything(%s)" % actor, True),
                            ("gripper_fully_closed(%s)" % actor, True)],
                        effects=[
                            ("has_anything(%s)" % actor, False),
                            ("gripper_fully_closed(%s)" % actor, False)],
                        task_planning=True,
                        )
    # Align with object to stack the block
    # And then place it. These are created separately due to some complexity here.
    for obj in domain.stackable_objs:
        _stackable_objs = [obj2 for obj2 in domain.stackable_objs if obj != obj2]
        align_stack = Policy()
        domain.add_operator("align_%s_with" % obj, policy=align_stack,
                            preconditions=[
                                ("top_is_clear({})", True),
                                ("has_obj(%s, %s)" % (domain.robot, obj), True),
                                ("aligned_with({}, %s)" % obj, False),
                                ("is_free(%s)" % domain.robot, True),
                            ],
                            effects=[
                                ("aligned_with({}, %s)" % obj, True),
                                ("is_free(%s)" % domain.robot, False),
                                ("has_goal(%s)" % domain.robot, True),
                            ],
                            task_planning=True,
                            to_entities=_stackable_objs,
                            subgoal_sampler=DiscreteGoalSampler("stack",
                                                                standoff=0.1,
                                                                cost=PlaceCost(),
                                                                attach=status.NO_CHANGE,
                                                                threshold=10.),
                            subgoal_connector=RRTConnector())
        place_stack = Policy() 
        domain.add_operator("stack_%s_on" % obj, policy=place_stack,
                            preconditions=[
                                ("top_is_clear({})", True),
                                ("aligned_with({}, %s)" % obj, True),
                                ("has_obj(%s, %s)" % (domain.robot, obj), True),
                                ("has_goal(%s)" % domain.robot, True),
                                #("stacked({}, %s)" % obj, False),
                            ],
                            effects=[
                                ("top_is_clear({})", False),
                                ("aligned_with({}, %s)" % obj, True),
                                ("stacked({}, %s)" % obj, True),
                                ("has_obj(%s, %s)" % (domain.robot, obj), False),
                                ("has_anything(%s)" % domain.robot, False),
                                ("is_free(%s)" % domain.robot, True),
                                ("has_goal(%s)" % domain.robot, False),
                            ],
                            task_planning=True,
                            planning_cost=0,
                            to_entities=_stackable_objs,
                            subgoal_sampler=AttachSampler(status.DETACH))
                            #subgoal_connector=LinearConnector())

    # Place object on a surface
    for surface in domain.surfaces:
        place_surface = Policy()
        domain.add_operator("place_on_%s" % surface, policy=place_surface,
                            preconditions=[
                                #("near(%s, %s)" % (domain.robot, surface), True),
                                ("has_obj(%s, {})" % domain.robot, True),
                            ],
                            effects=[
                                ("on_surface({}, %s)" % "tabletop", True),
                                ("on_surface({}, %s)" % surface, True),
                                ("has_obj(%s, {})" % domain.robot, False),
                                ("has_anything(%s)" % domain.robot, False),
                            ],
                            task_planning=True,
                            subgoal_sampler=SurfacePlacementSampler(surface,
                                                                    standoff=0.,
                                                                    cost=PlaceCost(),
                                                                    ),
                            to_entities=domain.manipulable_objs)
    # Placing in drawers
    for surface in domain.handles:
        place_surface = Policy()
        domain.add_operator("place_on_%s" % surface, policy=place_surface,
                            preconditions=[
                                ("is_open(%s)" % surface, True),
                                ("is_closed(%s)" % surface, False),
                                #("near(%s, %s)" % (domain.robot, surface), True),
                                ("has_obj(%s, {})" % domain.robot, True),
                            ],
                            effects=[
                                ("on_surface({}, %s)" % surface, True),
                                ("has_obj(%s, {})" % domain.robot, False),
                                ("has_anything(%s)" % domain.robot, False),
                                ("gripper_fully_open(%s)" % actor, True),
                            ],
                            task_planning=True,
                            subgoal_sampler=SurfacePlacementSampler(surface,
                                                                    standoff=0.,
                                                                    cost=PlaceCost()),
                            to_entities=domain.manipulable_objs)

    # Move the base around in the kitchen area
    # This operator will optimistically assume that we can move to a position that will reach the
    # object
    #move_to_object = Policy()
    #domain.add_operator("move_to", policy=move_to_object,
    #                    preconditions=[],
    #                    effects=[("near(%s, {})" % domain.robot, True)],
    #                    to_entities=domain.manipulable_objs+domain.handles+domain.surfaces,
    #                    task_planning=True,
    #                    )

    # Pull open -- drawers and doors
    pull_open = Policy() # Placeholder
    domain.add_operator("pull_open", policy=pull_open,
                        preconditions=[
                            ("is_open({})", False),
                            ("has_obj(%s, {})" % domain.robot, True),
                        ],
                        effects=[
                            ("is_open({})", True),
                            ("is_closed({})", False),
                        ],
                        to_entities=domain.handles,
                        task_planning=True,
                        subgoal_sampler=DrawerOffsetSampler(axis.Z, open_drawer=True),
                        subgoal_connector=LinearConnector(),
                        )
    # Push closed -- drawers and doors
    push_closed = Policy() # Placeholder
    domain.add_operator("push_closed", policy=push_closed,
                        preconditions=[
                            ("is_closed({})", False),
                            ("has_obj(%s, {})" % domain.robot, True),
                        ],
                        effects=[
                            ("is_open({})", False),
                            ("is_closed({})", True),
                        ],
                        to_entities=domain.handles,
                        task_planning=True,
                        subgoal_sampler=DrawerOffsetSampler(axis.Z, open_drawer=False),
                        subgoal_connector=LinearConnector(),
                        )
    # Open gripper if it was empty.
    open_gripper = BlockingOpenGripper()
    domain.add_operator("release", policy=open_gripper,
                        preconditions=[
                            ("has_anything(%s)" % actor, False),
                            ("has_obj(%s, {})" % actor, True),
                        ],
                        effects=[
                            ("has_anything(%s)" % actor, False),
                            ("has_obj(%s, {})" % actor, False),
                            ("gripper_fully_open(%s)" % actor, True),
                        ],
                        to_entities=domain.handles + domain.manipulable_objs,
                        task_planning=True,
                        subgoal_sampler=AttachSampler(status.DETACH, static=True),
                        )

    
def DefineYcbObjects(objs=None):
    """ Define object definitions for the YCB objects we're using in the kitchen scenario. This
    would ideally be done in a yml configuration file somewhere else instead of right here, so that
    we can do this a bit more nicely. """

    config = {
            "spam": {},
            "sugar": {},
            "cracker_box": {},
            "tomato_soup": {},
            "mustard": {}}

    return config

def TaskStackKitchen(iface, seed=None):
    """ Set up a planner and tell it to build a stack at a particular location """
    if seed is not None:
        np.random.seed(seed)
    
    # Place the blocks in differnet places
    colors = ["red", "green", "blue", "yellow"]
    block_names = [color + "_block" for color in colors]
    kitchen = iface.get_object("kitchen")
    surfaces = [kitchen.get_surface("indigo"), kitchen.get_surface("hitman")]
    valid = False
    while not valid:
        # Randomize positions of all the objects
        for block_name in block_names:
            obj = iface.get_object(block_name)
            surface = np.random.choice(surfaces)
            pose = surface.sample_pose(np.eye(4))
            p = pose[:3, 3] + np.array([0, 0, 0.05])
            r = tra.quaternion_from_matrix(pose)
            print(block_name, "was assigned pose =", (p, r))
            obj.set_pose(p, r, wxyz=True)

        # Check to see if blocks are in collision or generally just too close to each other. If
        # so, then we need to re-sample their positions.
        valid = True


def TaskSortIndigoKitchen(iface, seed=None):
    """ Set up sorting task on indigo alone, with a fixed camera. This task has you getting out
    some common set of objects from the drawers and putting them out on the table. """

    if seed is not None:
        np.random.seed(seed)

    # These are the objects we'll be using for now.
    # They're sort of simple but that's fine
    objs = ["cracker_box", "sugar", "tomato_soup", "mustard", "spam"]
    colors = ["red", "green", "blue", "yellow"]
    blocks = [color + "_block" for color in colors]
    kitchen = iface.get_object("kitchen")
    table = iface.get_object("table")
    surfaces = [kitchen.get_surface("indigo"),
                kitchen.get_surface("indigo_top_drawer"),
                kitchen.get_surface("indigo_bottom_drawer")]
    ignored = table.get_surface("top")
    drawers = [] # TODO
    valid = False

    # Randomize cabinets
    q = np.random.random(3) * 0.4
    drawer_names = ["indigo_drawer_top", "indigo_drawer_bottom", "hitman_drawer_top"]
    drawer_idx = [kitchen.get_link_index(name) for name in drawer_names]
    kitchen.set_joint_positions(q, drawer_idx)

    while not valid:
        # Randomize positions of all the objects
        for obj_name in objs + blocks:
            obj = iface.get_object(obj_name)
            if obj_name in blocks:
                surface = ignored
            else:
                surface = np.random.choice(surfaces)
            pose = surface.sample_pose(np.eye(4))
            p = pose[:3, 3]
            r = tra.quaternion_from_matrix(pose)
            print(obj_name, "was assigned pose =", (p, r))
            obj.set_pose(p, r, wxyz=True)

        # Check to see if blocks are in collision or generally just too close to each other. If
        # so, then we need to re-sample their positions.
        valid = True

def _setCameraIndigo(iface):
    # pos, quat = [0.509, 0.993, 0.542], [-0.002, 0.823, -0.567, -0.010]
    # pos1, quat1 = [1.2, 0.0, 0.25], [0, 0, 0, 1]
    # pos2, quat2 = [0.830, 1.180, 0.240], [0.505, 0.704, -0.396, -0.303]
    # - Translation: [0.235, -0.380, 0.945]
    # - Rotation: in Quaternion [0.704, -0.505, 0.303, -0.396]
    #pos1, quat1 = [-0.5, 1., 0.25], [0, 0, 0, 1]
    pos2, quat2 = [-0.25, 1.5, 1.5],  [0.704, -0.505, 0.303, 0.396]
    #T1 = pose_tools.make_pose(pos1, quat1)
    #T2 = tra.inverse_matrix(pose_tools.make_pose(pos2, quat2))
    #T2 = pose_tools.make_pose(pos2, quat2)
    iface.set_camera((pos2, quat2), matrix=False)


def _placeReachable(surface, robot, obj, z):
    """ Make sure any object we actually want to grasp is close enough to the robot base that it's
    feasible that we could reach it if we wanted to. """

    # pose = surface.sample_pose(np.eye(4))
    pose = surface.sample_pose(obj.default_pose, height=z, var_theta=1.)
    #pose = surface.sample_pose(None, height=0.05)
    p = pose[:3, 3]
    r = tra.quaternion_from_matrix(pose)
    print(obj.name, "was assigned pose =", p, "on surface", surface.name)
    obj.set_pose(p, r, wxyz=False)

    #print(robot.get_pose()[:2, 3])
    #print(pose[:2, 3])
    dist_to_base = np.linalg.norm(robot.get_pose()[:2, 3] - pose[:2, 3])
    #print("dist =", dist_to_base)
    if dist_to_base > 0.9:
        return False
    return True


def _chooseRobotPoseForIndigo(iface, kitchen, robot):
    """ Randomize robot position for data collection """
    q = [0, 0, 1, 0]
    #p = [1.03, 0.8, -1.23]
    p = [0.88, 0.8, -1.23]
    T = tra.quaternion_matrix(q)
    T[:3, 3] = p
    T_kitchen = kitchen.get_pose(matrix=True)
    # drift the pose slightly
    T[:2, 3] += 0.02 * np.random.randn(2)
    
    robot_pose = T_kitchen.dot(T)
    robot.set_pose_matrix(robot_pose)


def _placeAll(iface, robot, surfaces, ignored, objs, all_objs, blocks, max_tries=100, z=0.05):
    """ Create a scene that's known to be valid. Make sure everything makes sense and is somewhat
    reasonable. Don't create impossible scenes. """
    kitchen = iface.get_object("kitchen")
    valid = False
    tries = 0
    while not valid:
        tries += 1
        if tries > max_tries:
            input('Press enter to terminate')
            raise RuntimeError('Could not create environment!')
        i = 0
        
        _chooseRobotPoseForIndigo(iface, kitchen, robot)

        placed_objs = []
        for obj_name in all_objs + blocks:
            obj = iface.get_object(obj_name)
            if obj_name not in objs:
                surface = ignored
                _skip = True
            else:
                _skip = False
                surface = surfaces[0]
    
            # Don't check ignored objects
            placed = False
            while not placed and i < 100:
                # Place objects and make sure they're close enough if that's important
                placed = _placeReachable(surface, robot, obj, z) or _skip
                print('- placed', obj_name, '=', placed)
                if placed:
                    # Check to see if blocks are in collision or generally just too close to each
                    # other. If so, then we need to re-sample their positions.
                    placed = _checkCollisions(iface, robot, placed_objs + [obj_name])
                    print('- placed', obj_name, 'collision check =', placed)
                if placed:
                    placed_objs.append(obj_name)
                    break
                i += 1

        valid = placed
    return valid


def _checkCollisions(iface, robot, objs):
    """ Make sure we only create environments where objects aren't in crazy overlapping
    configurations. We can create environments via rejection sampling for now. """

    for obj in objs:
        ref = iface.get_object(obj)
        collides = robot.check_pairwise_collisions(ref, 0.01)
        if collides:
            print("- collision robot -->", obj)
            return False

        for obj2 in objs:
            if obj2 == obj:
                continue
            ref2 = iface.get_object(obj2)
            collides = ref2.check_pairwise_collisions(ref, 0.01)
            if collides:
                print("- collision", obj2, "-->", obj)
                return False
    return True


def TaskStack(iface, seed=None):
    """ Set up a planner and tell it to build a stack at a particular location """
    if seed is not None:
        np.random.seed(seed)

    robot = iface.get_object("robot")
    robot.set_pose(np.array([1.2, 0.0, 0.25]),
                   np.array([0, 0, 0, 1,]),
                   wxyz=False)
    
    # Place the blocks in differnet places
    colors = ["red", "green", "blue", "yellow"]
    block_names = [color + "_block" for color in colors]
    kitchen = iface.get_object("kitchen")
    surfaces = [kitchen.get_surface("indigo"), kitchen.get_surface("hitman")]
    valid = False
    while not valid:
        # Randomize positions of all the objects
        for block_name in block_names:
            obj = iface.get_object(block_name)
            surface = np.random.choice(surfaces)
            pose = surface.sample_pose(np.eye(4))
            p = pose[:3, 3] + np.array([0, 0, 0.05])
            r = tra.quaternion_from_matrix(pose)
            print(block_name, "was assigned pose =", (p, r))
            obj.set_pose(p, r, wxyz=True)

        # Check to see if blocks are in collision or generally just too close to each other. If
        # so, then we need to re-sample their positions.
        valid = True


def TaskBlocksDrawer(iface, seed=None):
    """
    Get goal information and configure the world
    """

    if seed is not None:
        np.random.seed(int(seed))

    #all_objs = ["cracker_box", "sugar", "tomato_soup", "mustard", "spam"]
    all_objs = []
    colors = ["red", "green", "blue", "yellow"]
    blocks = [color + "_block" for color in colors]
    kitchen = iface.get_object("kitchen")
    table = iface.get_object("table")
    surfaces = [kitchen.get_surface("indigo"),
                kitchen.get_surface("indigo_top_drawer"),
                kitchen.get_surface("indigo_bottom_drawer")]
    # ignored = table.get_surface("top")
    ignored = kitchen.get_surface("hitman")
    drawers = [] # TODO
    valid = False

    robot = iface.get_object("robot")
    robot.set_pose(np.array([-0.2, 0.8, 0.25]),
                   [0, 0, 1, 0],
                   wxyz=False)

    _setCameraIndigo(iface)

    goal = [
            ("on_surface(blue_block, indigo_top_drawer)", True),
            #("on_surface(spam, indigo)", True),
            #("has_obj(robot, spam)", True),
            ("is_open(indigo_top_drawer)", True),
            #("is_closed(indigo_top_drawer)", True),
            # ("near(robot, indigo)", True)
    ]
    objs = ["blue_block", "red_block"]
    _placeAll(iface, robot, surfaces, ignored, objs, all_objs, blocks)
    return objs, goal


def TaskSpamDrawer(iface, seed=None):
    """
    Get goal information and configure the world
    """

    if seed is not None:
        np.random.seed(int(seed))

    all_objs = ["cracker_box", "sugar", "tomato_soup", "mustard", "spam"]
    colors = ["red", "green", "blue", "yellow"]
    blocks = [color + "_block" for color in colors]
    kitchen = iface.get_object("kitchen")
    table = iface.get_object("table")
    surfaces = [kitchen.get_surface("indigo"),
                kitchen.get_surface("indigo_top_drawer"),
                kitchen.get_surface("indigo_bottom_drawer")]
    # ignored = table.get_surface("top")
    ignored = kitchen.get_surface("hitman")
    drawers = [] # TODO

    robot = iface.get_object("robot")
    robot.set_pose(np.array([-0.2, 0.8, 0.25]),
                   [0, 0, 1, 0],
                   wxyz=False)

    goal = [
            ("on_surface(spam, indigo_top_drawer)", True),
            #("on_surface(spam, indigo)", True),
            #("has_obj(robot, spam)", True),
            ("is_open(indigo_top_drawer)", True),
            #("is_closed(indigo_top_drawer)", True),
            # ("near(robot, indigo)", True)
    ]
    objs = ["spam", "sugar"]
    _placeAll(iface, robot, surfaces, ignored, objs, all_objs, blocks)
    return objs, goal


def TaskSortIndigo(iface, seed=None):
    """ Set up sorting task on indigo alone, with a fixed camera. This task has you getting out
    some common set of objects from the drawers and putting them out on the table. """

    if seed is not None:
        np.random.seed(seed)

    # These are the objects we'll be using for now.
    # They're sort of simple but that's fine
    all_objs = ["cracker_box", "sugar", "tomato_soup", "mustard", "spam"]
    colors = ["red", "green", "blue", "yellow"]
    blocks = [color + "_block" for color in colors]
    kitchen = iface.get_object("kitchen")
    table = iface.get_object("table")
    surfaces = [kitchen.get_surface("indigo"),
                kitchen.get_surface("indigo_top_drawer"),
                kitchen.get_surface("indigo_bottom_drawer")]
    ignored = table.get_surface("top")
    drawers = [] # TODO
    valid = False

    robot = iface.get_object("robot")
    robot.set_pose(np.array([-0.5, 1., 0.25]),
                   np.array([0, 0, 0, 1,]),
                   wxyz=False)

    _setCameraIndigo(iface)

   # Randomize cabinets
    q = (np.random.random(3) * 0.4) * [1., 0., 1.]
    drawer_names = ["indigo_drawer_top", "indigo_drawer_bottom", "hitman_drawer_top"]
    drawer_idx = [kitchen.get_link_index(name) for name in drawer_names]
    kitchen.set_joint_positions(q, drawer_idx)

    goal = [
            ("on_surface({}, indigo_top_drawer)", True),
            #("on_surface(spam, indigo)", True),
            #("has_obj(robot, spam)", True),
            ("is_open(indigo_top_drawer)", True),
            #("is_closed(indigo_top_drawer)", True),
    ]
    #num_objs = np.random.randint(3) + 1
    num_objs = np.random.randint(2) + 1
    obj_idx = np.arange(len(all_objs))
    np.random.shuffle(obj_idx)
    objs = [all_objs[i] for i in obj_idx[:num_objs]]
    goal_obj = objs[0] # np.random.randint(len(objs))]
    goal = [(pred.format(goal_obj), val) for pred, val in goal]
    _placeAll(iface, robot, surfaces, ignored, objs, all_objs, blocks)
    return objs, goal


def GetHandleEndPoses():
    """
    This should really be loaded from a configuration file. Get a set of poses for grabbing the
    handle.

    From ROS:
    At time 1479.217
    - Translation: [0.352, 0.108, 0.022]
    - Rotation: in Quaternion [0.514, -0.490, -0.483, 0.513]
                in RPY (radian) [1.578, -0.006, -1.517]
                in RPY (degree) [90.434, -0.323, -86.910]

    At time 1552.183
    - Translation: [0.361, -0.104, 0.022]
    - Rotation: in Quaternion [0.514, -0.489, -0.483, 0.513]
                in RPY (radian) [1.578, -0.006, -1.515]
                in RPY (degree) [90.390, -0.327, -86.821]
    """
    pose1 = tra.euler_matrix(np.pi/2, 0, -1*np.pi/2)
    pose1[:3, 3] = np.array([0.31, 0.1, -0.02])
    pose2 = tra.euler_matrix(np.pi/2, 0, -1*np.pi/2)
    pose2[:3, 3] = np.array([0.31, -0.1, -0.02])
    return pose1, pose2
