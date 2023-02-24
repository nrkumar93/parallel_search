# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import brain2.robot.heuristics as heuristics
import brain2.utils.axis as axis
import brain2.utils.pose as pose_tools
import brain2.utils.status as status
import brain2.utils.transformations as tra
from brain2.utils.extend import simple_extend

from brain2.task.domain import DomainDefinition
from brain2.motion_planners.problem import MotionPlanningProblem

import brain2.robot.entity as entity
from brain2.robot.cube import CubeGraspLookupTable
from brain2.robot.world_state import RobotEntityState
from brain2.robot.world_state import HumanHandState
from brain2.robot.surface import SurfaceState
from brain2.robot.surface import DrawerState
from brain2.robot.kitchen import DefineKitchenDomain

# For creating environments
# TODO: this should be replaced by something else in the future
import brain2.bullet.problems as problems


class RobotDomainDefinition(DomainDefinition):

    def apply_action(self, prev_world_state, world_state, params, verbose=False, debug=False):
        """
        Apply the effects of a subgoal to the world.
        """

        # THIS IS WHERE THE PROBLEM IS
        # TODO: this could be faster
        # Update cleanly from previous entry
        #for entity1, (k2, entity2) in zip(prev_world_state.entities.values(),
        #                                  world_state.entities.items(),):
        for k1, entity1 in prev_world_state.entities.items():
            world_state[k1] = entity1.copy()

        actor = world_state[params.actor]
        ee_moved = False
        # print(params.q, params.ee_pose)
        # print("actor", actor.ee_pose)
        if params.q is not None:
            if verbose:
                print(params.actor, "moving to config =", params.q)
            actor.set_config(params.q)
            actor.ref.set_joint_positions(params.q)
            ee_moved = True

        # Update the pose of the robot base
        # This means we need to update some other infrmation as well
        if params.pose is not None:
            actor.set_base_pose(params.pose)
            actor.ref.set_pose_matrix(params.pose)
            ee_moved = True

        # Read end effector pose from the simulation backend if we lost it
        if ee_moved:
            if verbose:
                print("End effector moved for:", actor, params.actor, params.goal)
            actor.set_ee_pose(actor.ref.get_ee_pose(matrix=True))
        
        # Handling attachment
        # This part is a little hacky, would be better if the physics engine would handle this.
        # TODO: look into supporting with constraints
        # If we have an attached object move it as well
        if ee_moved and actor.attached_to is not None:
            if verbose:
                print("Attached to =", actor.attached_to)
            attached_obj = world_state[actor.attached_to]
    
            # First part is for handling opening drawers.
            if attached_obj.ref.entity_type == entity.JOINT:
                if verbose:
                    print("Attached to joint. Opening motion =", params.linear_dist)
                if params.linear_dist is not None :
                    # Set the joint position
                    attached_obj.q += -1 * params.linear_dist
                    attached_obj.ref.set_joint_position(attached_obj.q)
                    attached_obj.pose = attached_obj.ref.get_pose()
                    attached_obj.surface.pose = attached_obj.surface.update(matrix=True)

                for obj in attached_obj.objs:
                    # update this thing's pose as well
                    obj.set_base_pose(attached_obj.pose.dot(obj.attached_pose))
            else:
                # Free-floating objects just get their base position set.
                attached_obj.set_base_pose(actor.ee_pose.dot(actor.attached_inv_pose))
                # TODO: should we be updating the scene here? I think we want to
                # do that, yes.
                # So -- do we really need to do this? Updates the backend
                # version of the world in addition to the one we have in the
                # world state.
                attached_obj.ref.set_pose_matrix(actor.ee_pose.dot(actor.attached_inv_pose))

        # Set goal pose for the motion
        if params.goal is not None:
            obj = world_state[params.goal]
            if params.ee_pose is not None:
                actor.set_goal(params.goal, params.relative_pose)
        else:
            obj = None

        # Handle attachment
        if params.attach == status.ATTACH:
            if obj is None:
                raise RuntimeError('must provide a goal to attach to')
            if params.ee_pose is None:
                attach_pose = obj.goal_pose
                raise RuntimeError('why would you do this')
            else:
                attach_pose = params.relative_pose
            if attach_pose is None:
                raise RuntimeError('did not provide way to attach to ' + str(params.goal))
            
            if verbose:
                print("attaching", params.attach_to, "to", params.actor, "with pose =\n", attach_pose)
            actor.attach(params.attach_to, attach_pose)
            actor.ref.close_gripper_around(obj.ref)
        elif params.attach == status.DETACH:
            if verbose:
                print(params.actor, "releasing object")
            actor.detach()
            actor.ref.open_gripper()
        elif params.attach == status.PLACE:
            actor.detach()
            if params.goal is None:
                raise RuntimeError('tried to place on nothing')
            if verbose:
                print("attaching", params.attach_to, "to", params.goal, "with pose =\n", params.relative_pose)
            obj.attach(params.attach_to, params.relative_pose)
            actor.ref.open_gripper()
        elif params.attach == status.NO_CHANGE:
            pass
        else:
            raise RuntimeError('attachment status code not supported: ' + str(params.attach))

        # Pause after applying action
        if debug:
            self.iface.update(world_state)
            input('waiting for [ENTER]')

    def set_robot_control(self, ctrl):
        """ Set the control interface used for robot execution """
        self.root[self.robot].set_ctrl(ctrl)

    def _add_surfaces(self, surface_names, surface_refs):
        """
        Add defined surface entities to the world state.

        Track where all the various surfaces are as part of the state. Since
        these are some of our affordances after all
        """
        if self.compiled:
            raise RuntimeError('cannot add new surface entities after compilation')
        for name, surface in zip(surface_names, surface_refs):
            self.add_entity(name, SurfaceState, surface)
        #    print("adding surface", name, surface)
        #input('----')

    def reset_world_state(self, world_state):
        self.iface.update(world_state)

    def get_backend(self):
        return self.iface

    def _get_hands(self, hands):
        if hands:
            return ["left", "right"]
        else:
            return []
 
    def __init__(self, iface, objects=[], hands=[], robot="robot", verbose=0,
            add_default_predicates=True):
        """
        iface: planning environment with necessary world information; should
               contain collision checking etc.
        iksolver: inverse kinematics for the primary actor (the robot!)
        objects: physical things in the world
        hands: humans, basically.
        robot: unique name id for the thing doing the planning
        """
        super(RobotDomainDefinition, self).__init__()
        self.iface = iface
        self.robot = robot
        self.verbose = verbose

        # Store object references I guess
        self.references = {}

        # Add objects to world state register
        for obj, params in objects.items():
            obj_ref = iface.get_object(obj)
            domain_obj = self.add_entity(obj, RobotEntityState, obj_ref, **params)
            self.references[obj] = obj_ref

        self.objects = objects

        # Add human objects
        self.hands = hands
        for hand in hands:
            # No human hand object?
            obj_ref = iface.get_object(hand)
            domain_obj = self.add_entity(hand, HumanHandState, obj_ref, **params)
            self.references[hand] = obj_ref

        # Create some basic values for an example here
        if add_default_predicates:

            self.manipulable_objs = [obj for obj in objects.keys() if obj not in ["table", self.robot]]
            # Are we at the home position?
            def at_home(ws, x):
                x = ws[x]
                if x.ref.home_q is None or x.q is None:
                    return False
                return np.all(np.abs(x.q[:7] - x.ref.home_q[:7]) < 0.1)

            has_goal = lambda ws, x: ws[x].goal_pose is not None
            self.add_property("has_goal", has_goal, self.robot)
            self.add_property("at_home", at_home, self.robot)
            self.add_property("observed", lambda ws, x: ws[x].observed)
            self.add_property("stable", lambda ws, x: ws[x].stable)
            self.add_property("has_anything",
                              lambda ws, x: ws[x].attached_to is not None,
                              self.robot)
            self.add_property("gripper_moving",
                              lambda ws, x: ws[x].gripper_moving,
                              self.robot)
            self.add_property("gripper_fully_closed",
                              lambda ws, x: ws[x].gripper_fully_closed,
                              self.robot)
            self.add_property("gripper_fully_open",
                              lambda ws, x: ws[x].gripper_fully_open,
                              self.robot)

            # ------------------------
            # Create some positional predicates based on table affordances
            table = self.references["table"]
            top = table.get_surface("top")
            over = table.get_surface("over")

            # Create operations for grasping + moving objects around
            self.add_property("on_table_top",
                              lambda ws, x: top.overlap_check(ws[x].ref),
                              self.manipulable_objs)
            # Add a predicate for if we've grasped an object or not 
            self.add_relation("has_obj",  
                              lambda ws, x, y: ws[x].attached_to == y,
                              self.robot, self.manipulable_objs)
            # have we generated a motion plan for this?
            # TODO make sure this works properly
            self.add_relation("has_plan",
                              lambda ws, x, y: ws[x].goal_obj == y,
                              self.robot, self.manipulable_objs + self.hands)

            if len(hands) > 0:
                # add hand predicates
                self.add_property("hand_over_table", 
                                  lambda ws, x: ws[x].observed and over.overlap_check(ws[x].ref),
                                  self.hands)
                self.add_property("hand_has_obj",
                                  #lambda ws, x: ws[x].obj_pose is not None,
                                  lambda ws, x: ws[x].has_obj is not None,
                                  self.hands)

                           
    def get_robot(self):
        return self.root[self.robot]

    def get_default_planning_problem(self, get_goal,
                                     is_done=None,
                                     actor=None,
                                     **kwargs):
        # Done or not
        if is_done is None:
            is_done = lambda q: False
        # Default actor
        if actor is None:
            actor = self.robot
        # Get interface and set parameters
        robot = self.iface.get_object(actor)
        pb_config = {
                'dof': robot.dof,
                'p_sample_goal': 0.2,
                'iterations': 100,
                'goal_iterations': 100,
                'verbose': 1, #self.verbose,
                'shortcut': True,
                'min_iterations': 10,
                'shortcut_iterations': 50,
                }
        pb_config.update(kwargs)
        is_valid = lambda q: not self.iface.check_collisions(robot, q, max_pairwise_distance=0.005)
        extend = lambda q1, q2: simple_extend(q1, q2, 0.2)
        return MotionPlanningProblem(sample_fn=robot.sample_uniform,
                                     goal_fn=get_goal,
                                     extend_fn=extend,
                                     is_valid_fn=is_valid,
                                     is_done_fn=is_done,
                                     config=pb_config,
                                     distance_fn=None)

class CartDomainDefinition(RobotDomainDefinition):
    """ Basic version """

    def __init__(self, iface, obj_config, ik_solver, *args, **kwargs):
        super(CartDomainDefinition, self).__init__(iface, obj_config,
                                                   *args,
                                                   **kwargs)
        self.ik_solver = ik_solver
        self.root[self.robot].ref.set_ik_solver(ik_solver)

        self.surfaces = ["tabletop", "left", "right", "far", "center"]
        self.drawers = []
        table = self.references["table"]
        #tabletop = table.get_surface("top")
        tabletop = table.get_surface("workspace")
        left = table.get_surface("left")
        right = table.get_surface("right")
        far = table.get_surface("far")
        center = table.get_surface("center")
        surface_refs = [tabletop, left, right, far, center]
        self._add_surfaces(self.surfaces, surface_refs)

        self.doors = []
        self.handles = self.drawers + self.doors

        drawer_surface_refs = [kitchen.get_surface(drawer) for drawer in self.drawers]
        joint_refs = [kitchen.get_joint_reference(surface.parent_frame)
                      for surface in drawer_surface_refs]
        for drawer, drawer_surface, joint_ref in zip(self.drawers, drawer_surface_refs, joint_refs):
            self.add_entity(drawer, DrawerState, joint_ref, drawer_surface)

        # Get the list of all objects we can move around
        self.manipulable_objs = [obj for obj in self.objects.keys() if obj not in
                ["table", "kitchen", self.robot] + self.hands]
        self.stackable_objs = [obj for obj in self.manipulable_objs if "block" in obj]

        DefineKitchenDomain(self)

class CartObjectsDomainDefinition(CartDomainDefinition):
    """For handling more variable sets of objects"""

    def __init__(self, iface, robot_control=None, ik_solver=None, obj_config=None):
        config = {"robot": {
                    "control": robot_control,
                    },
                "table": {},
                }
        # config.update(self.objs)
        if obj_config is not None:
            config.update(obj_config)
        super(CartObjectsDomainDefinition, self).__init__(iface, config,
                                                          ik_solver,
                                                           add_default_predicates=False)


class CartBlocksDomainDefinition(CartDomainDefinition):
    """ Define version of the domain with blocks on table """

    def _get_objs(self, robot_control):
        objs = {"robot": {"control": robot_control},
                "table": {},
                }
        for color in ["red", "green", "blue", "yellow"]:
            cfg = {
                    "obj_type": "block",
                    "obj_size": "median",
                    "color": color
                    }
            # name = "%s_block_%s" % (cfg["obj_size"], cfg["color"])
            name = "%s_block" % (cfg["color"])
            objs[name] = cfg
        return objs

    def __init__(self, iface=None, robot_control=None, hands=False,
                 ik_solver=None, assets_path=None, visualize=False,
                 padding=0.):
        if iface is None:
            iface = problems.franka_cart_blocks(assets_path, visualize, "d435",
                    padding=padding)
        super(CartBlocksDomainDefinition, self).__init__(iface,
                                                         self._get_objs(robot_control),
                                                         ik_solver,
                                                         self._get_hands(hands),
                                                         add_default_predicates=False)


class KitchenDomainDefinition(RobotDomainDefinition):
    """Create objects, operators, and concepts for the kitchen object
    manipulation domain."""

    def _get_objs(self, robot_control, extra_objs=None, add_blocks=False):
        objs = {"robot": {"control": robot_control},
                "table": {},
                "kitchen": {},
                }

        # Add in unique objects 
        if extra_objs is not None:
            objs.update(extra_objs)

        if add_blocks:
            for color in ["red", "green", "blue", "yellow"]:
                cfg = {
                        "obj_type": "block",
                        "obj_size": "median",
                        "color": color,
                        "bounds": [0.05, 0.05, 0.05],
                        }
                # name = "%s_block_%s" % (cfg["obj_size"], cfg["color"])
                name = "%s_block" % (cfg["color"])
                objs[name] = cfg

        return objs

    def __init__(self, iface, objs={}, robot_control=None, hands=False, ik_solver=None,
                 add_blocks=True):
        super(KitchenDomainDefinition, self).__init__(iface,
                                                      self._get_objs(robot_control, objs,
                                                          add_blocks=add_blocks),
                                                      self._get_hands(hands),
                                                      add_default_predicates=False)

        # Create interfaces for solving some problems
        self.ik_solver = ik_solver
        self.root[self.robot].ref.set_ik_solver(ik_solver)

        # Create surface objects
        # These tags are used in domain compilation
        self.surfaces = ["indigo", "hitman", "tabletop"]
        self.drawers = ["indigo_top_drawer", "indigo_bottom_drawer", "hitman_top_drawer"]

        # Get referebces to create the objects
        table = self.references["table"]
        kitchen = self.references["kitchen"]
        
        # Set up tabletop and other tables
        tabletop = table.get_surface("top")
        over_table = table.get_surface("over")
        hitman = kitchen.get_surface("hitman")
        indigo = kitchen.get_surface("indigo")
        surface_refs = [indigo, hitman, tabletop]
        self._add_surfaces(self.surfaces, surface_refs)

        # openable drawers and doors
        # TODO: fill this out when you've decided how to handle these
        self.doors = []
        self.handles = self.drawers + self.doors
        drawer_surface_refs = [kitchen.get_surface(drawer) for drawer in self.drawers]
        joint_refs = [kitchen.get_joint_reference(surface.parent_frame)
                      for surface in drawer_surface_refs]
        for drawer, drawer_surface, joint_ref in zip(self.drawers, drawer_surface_refs, joint_refs):
            self.add_entity(drawer, DrawerState, joint_ref, drawer_surface)

        # Get the list of all objects we can move around
        self.manipulable_objs = [obj for obj in self.objects.keys() if obj not in
                ["table", "kitchen", self.robot] + self.hands]
        self.stackable_objs = [obj for obj in self.manipulable_objs if "block" in obj]

        DefineKitchenDomain(self)
