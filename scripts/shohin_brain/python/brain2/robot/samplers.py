# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np

import brain2.utils.status as status
import brain2.utils.transformations as tra

# Import text colors
from brain2.utils.info import *

# Basic nonsense for holding objects
from brain2.motion_planners.abstract import GoalSampler
from brain2.task.action import ActionParams


def _get_robot_end(sampler, world_state, robot):
    """ Get the end of the robot's kinematic chain """
    if sampler.use_attached and robot.attached_to is not None:
        # We will be sampling goals for this object
        attached_pose = robot.attached_pose
        attached_obj_name = robot.attached_to
        attached_obj = world_state[attached_obj_name]

    else:
        attached_obj = None
        attached_pose = None
       
    ee_ref = robot.ref.ee_ref
    if attached_obj is not None:
        end_pose = attached_obj.pose
        inv_end_pose = tra.inverse_matrix(end_pose)
        end_ref = attached_obj.ref
    else:
        end_pose = robot.ee_pose
        inv_end_pose = robot.inv_ee_pose
        end_ref = robot.ref.ee_ref

    assert(end_pose is not None)
    return attached_obj, attached_pose, end_pose, inv_end_pose, end_ref, ee_ref


class DiscreteGoalSampler(GoalSampler):
    """ Sample grasps from a particular distribution. Or other poses. Who knows really. """

    def __init__(self, affordance, standoff=None, use_attached=True, attach=status.ATTACH,
            cost=None, threshold=float('Inf'),
            max_iter=100, condition=None):
        self.affordance = affordance
        self.standoff = standoff                # Compute standoff position
        self.attach = attach                    # Are we attaching to the object at this pose?
        self.cost = cost
        self.threshold = threshold              # Cost must be under threshold
        self.use_attached = use_attached        # Should we use object currently being held?
        self.max_iter = max_iter
        self.condition = condition              # Create a condition that we can use

    def __call__(self, world_state, actor, goal, batch_size=100, verbose=False):
        """
        Create a sampler for a particular world configuration. Will take the object's current pose
        and a list of grasps, and will generate that set of grasps.
        """

        obj = world_state[goal]
        robot = world_state[actor]

        # Extract necessary values
        obj_pose = np.copy(obj.pose) # object pose
        robot_ref = robot.ref
        config = robot.q
        
        # Get information on the end of the arm
        attached_obj, attached_pose, end_pose, inv_end_pose, end_ref, ee_ref = _get_robot_end(self, world_state, robot)

        if inv_end_pose is None:
            raise RuntimeError('did not have end pose set correctly; inv end pose missing')

        # TODO: set this up better
        # This gets a list of poses associated with whichever affordance we had wanted for this
        # object
        #print(world_state[goal].ref, goal, actor)
        #print("end pose =\n", end_pose, "inv end pose =\n", inv_end_pose)
        poses = world_state[goal].ref.affordances[self.affordance](batch_size)

        # Are we generating a condition-restricted sampler?
        uses_condition = False
        if self.condition is not None:
            uses_condition = True
            ws = world_state.fork()

        # Handle attachment properly
        ignored = [] if robot.attached_to is None else [attached_obj.ref]
        if self.attach == status.ATTACH:
            ignored.append(obj.ref)
            attach_to_obj = goal
        elif robot.attached_to is not None: # self.attach == status.PLACE:
            attach_to_obj = robot.attached_to
            attached_pose = robot.attached_pose
            attached_inv_pose = robot.attached_inv_pose
        else:
            attach_to_obj = None

        ee_ignored = [name for name in ignored] + [robot_ref]

        _use_standoff = self.standoff is not None

        # Search in order to find one that works
        # For each entry in the batch...
        # for _ in range(batch_size):
        np.random.shuffle(poses)
        #for pose, val in sorted(zip(poses, scores), key=lambda p: p[1]):
        it = 0
        for pose in poses:
            if it > batch_size: break
            it += 1

            # print(actor, goal, "affordance =", self.affordance, "cost =", val, "pos =", pose[:3,3])
            # internal function just for this
            if self.cost is not None:
                val = self.cost(end_pose, inv_end_pose, obj_pose, pose, ref=end_ref)

                # Make sure it's still legal
                if val is None:
                    continue

            goal_pose = obj_pose.dot(pose)
            # print(obj_pose[:3,3], goal_pose[:3, 3], val)

            # Check collisions here for the floating hand reference
            end_ref.set_pose_matrix(goal_pose)
            # TODO check collisions here

            if attach_to_obj is not None:
                goal_pose = goal_pose.dot(attached_pose)
                ee_ref.set_pose_matrix(goal_pose)
                # TODO check collisions here
            
            if not ee_ref.validate(None, suppressed_refs=ee_ignored):
                if verbose:
                    logwarn("end effector in collision: " + str(ee_ignored))
                continue

            if _use_standoff:
                _T = np.eye(4)
                _T[2,3] = -1*self.standoff
                standoff_pose = goal_pose.dot(_T)
                goal_pose = standoff_pose

            # Do inverse kinematics to make sure we can get here from where we were
            q_init_idx = np.random.randint(3)
            q_init = [config, robot_ref.home_q, np.random.random(robot_ref.dof)][q_init_idx]
            _q = robot_ref.ik_solver(robot_ref, goal_pose, q0=q_init)

            # If we couldn't solve the inverse kinematics then quit
            if _q is None:
                if verbose:
                    logwarn("Goal - IK failed")
                continue
            
            # Check the standoff pose and return that instead.
            if _use_standoff:
                # Compute standoff position
                _q = robot_ref.ik_solver(robot_ref, standoff_pose, q0=_q)

            if _q is None:
                if verbose:
                    logwarn("Standoff - IK failed")
                continue
            
            # Make sure this is expected to be collision-free
            if robot_ref.validate(_q, suppressed_refs=ignored):
                return ActionParams(actor=actor, goal=goal, q=_q,
                                    relative_pose=pose,
                                    attach=self.attach,
                                    attach_to=attach_to_obj,
                                    ee_pose=goal_pose,
                                    success=True)

        return ActionParams(actor=actor, goal=goal, success=False)


class LookAtSampler(GoalSampler):
    """
    Generate configurations that look at random points on a surface
    """

    def __init__(self, surface_name, look_area_name):
        self.surface_name = surface_name
        self.look_area_name = look_area_name

    def __call__(self, world_state, actor, goal=None, batch_size=100, verbose=False):
        robot = world_state[actor]
        surface = world_state[self.surface_name].surface
        look_area = world_state[self.look_area_name].surface
        robot_ref = robot.ref

        for _ in range(batch_size):
            
            # sample a position in the first area and then the second one
            pt1 = surface.sample_xyz()
            pt2 = look_area.sample_xyz()

            # Compute orientation between them (axis up)

            q_init_idx = np.random.randint(3)
            q_init = [config, robot_ref.home_q, np.random.random(robot_ref.dof)][q_init_idx]
            _q = robot_ref.ik_solver(robot_ref, standoff_pose, q0=q_init)
            if robot_ref.validate(_q, suppressed_refs=ignored):
                return ActionParams(actor=actor, goal=goal, q=_q,
                                    success=True)

        return ActionParams(actor=actor, goal=goal, success=False)


class SurfacePlacementSampler(GoalSampler):
    """ Finds a pose to place on a particular surface """
    def __init__(self, surface_name, standoff, cost=None, condition=None,
            flag=False):
        self.surface_name = surface_name
        self.standoff = 0.0                     # Compute standoff position
        self.cost = cost
        self.use_attached = True                # Always will use attached object
        self.condition = condition
        self.flag = flag                        # Mark to debug this sampler

    def __call__(self, world_state, actor, goal, batch_size=100, verbose=False):
        obj = world_state[goal]
        robot = world_state[actor]
        surface = world_state[self.surface_name]

        # Extract necessary values
        obj_pose = np.copy(obj.pose) # object pose
        robot_ref = robot.ref
        config = robot.q

        # Get information on the end of the arm
        attached_obj, attached_pose, end_pose, inv_end_pose, end_ref, ee_ref = _get_robot_end(self, world_state, robot)
        if attached_obj is None:
            raise RuntimeError('tried to place without an object')

        # Are we generating a condition-restricted sampler?
        uses_condition = False
        if self.condition is not None:
            uses_condition = True
            # We need to create a copy of the world state that we can use to evaluate all these
            # different conditions. This will make our lives a bit easier
            ws = world_state.fork()

        # Get the object we are placing
        attach_to_obj = robot.attached_to
        attached_pose = robot.attached_pose
        attached_inv_pose = robot.attached_inv_pose
        verbose = verbose or self.flag

        ee_pose = robot.ee_pose
        inv_ee_pose = robot.inv_ee_pose

        it = 0
        parent_ref = [surface.surface.obj_ref]
        while it < batch_size:
            it += 1

            # Sample a random surface pose
            pose = surface.surface.sample_pose(obj.pose, var_theta=1.)
            goal_pose = pose.dot(attached_pose)
            end_ref.set_pose_matrix(pose)
            ee_ref.set_pose_matrix(goal_pose)

            if self.cost is not None:
                #val = self.cost(end_pose, inv_end_pose, obj_pose, pose,
                val = self.cost(ee_pose, inv_ee_pose, pose, goal_pose,
                        ref=ee_ref, verbose=verbose)
                if self.flag:
                    print(it, val)
                    raw_input('---')

                # Make sure it's still legal
                if val is None:
                    continue

            if not end_ref.validate(verbose=self.flag, suppressed_refs=parent_ref):
                continue
            
            # Now we have a placement position. We just need to make sure it actually makes sense to
            # go there.
            # Figure out where the robot's arm is going to go with our given IK solver.
            q_init_idx = np.random.randint(3)
            q_init = [config, robot_ref.home_q, np.random.random(robot_ref.dof)][q_init_idx]
            # goal_pose = pose.dot(attached_pose)
            _q = robot_ref.ik_solver(robot_ref,
                                     goal_pose,
                                     q0=q_init)

            # If we couldn't solve the inverse kinematics then quit
            if _q is None:
                if verbose:
                    logwarn("IK failed")
                continue

            # Make sure this is expected to be collision-free
            # If it is, then we can return everything we need
            if robot_ref.validate(_q, suppressed_refs=[end_ref]):
                return ActionParams(actor=actor, goal=goal, q=_q,
                                    relative_pose=pose,
                                    attach_to=robot.attached_to,
                                    attach=status.PLACE,
                                    ee_pose=goal_pose,
                                    success=True)

        if self.flag: import pdb; pdb.set_trace()
        return ActionParams(actor=actor, goal=goal, success=False)
        

class LinearSampler(GoalSampler):
    """ Linear motion of a certain size. These motions will open drawers or cabinets. """

    def __init__(self, axis, offset, in_world_frame=False, attach=status.NO_CHANGE):
        self.axis = axis
        self.offset = offset
        self.attach = attach
        self.offset_pose = np.eye(4)
        self.offset_pose[self.axis, 3] = self.offset
        self.in_world_frame = in_world_frame

    def __call__(self, world_state, actor, goal, batch_size=100, debug=False):
        robot_ref = world_state[actor].ref
        ee_pose = world_state[actor].ee_pose
        if self.in_world_frame:
            goal_pose = np.copy(ee_pose)
            goal_pose[self.axis, 3] += self.offset
        else:
            goal_pose = ee_pose.dot(self.offset_pose)
        config = world_state[actor].q
        _q = robot_ref.ik_solver(robot_ref,
                                 goal_pose,
                                 q0=config)
        ee_ref = robot_ref.ee_ref
        ee_ref.set_pose_matrix(goal_pose)
        # import pdb; pdb.set_trace()
        return ActionParams(actor=actor, goal=goal, q=_q,
                            ee_pose=ee_pose,
                            #relative_pose=self.offset_pose,
                            attach=self.attach,
                            linear_dist=self.offset,
                            success=True,)

class AttachSampler(GoalSampler):
    """ use high-level plan information. connects two objects. """

    def __init__(self, attach=status.ATTACH, static=False, use_attached=True):
        self.attach = attach
        self.static = static
        self.use_attached = use_attached

    def __call__(self, world_state, actor, goal, batch_size=100, debug=False):
        actor_obj = world_state[actor]
        goal_obj = world_state[goal]
        obj_pose = np.copy(goal_obj.pose)

        if not self.static:
            if actor_obj.goal_pose is None:
                # We cannot do this
                _q = None
                goal_pose = actor_obj.ee_pose
            else:
                # Compute the actual goal position
                goal_pose = obj_pose.dot(actor_obj.goal_pose)

                # Get the end of the robot's current kinematic chain
                ends = _get_robot_end(self, world_state, actor_obj)
                (attached_obj, attached_pose, end_pose, inv_end_pose, end_ref, ee_ref) = ends

                # If it has an object handle that properly
                if actor_obj.attached_to is not None:
                    goal_pose = goal_pose.dot(attached_pose)
                    ee_ref.set_pose_matrix(goal_pose)
                    # TODO check collisions here

                # Do inverse kinematics
                config = actor_obj.q
                _q = actor_obj.ref.ik_solver(actor_obj.ref,
                                             goal_pose,
                                             q0=config)
        else:
            _q = actor_obj.q
            goal_pose = actor_obj.ee_pose

        # Return the motion to actually attach at the given pose
        return ActionParams(actor=actor, goal=goal,
                            attach=self.attach,
                            attach_to=goal,
                            relative_pose=actor_obj.goal_pose,
                            q=_q,
                            ee_pose=goal_pose,
                            success=_q is not None)


class OffsetSampler(LinearSampler):
    """ Offset applied from goal params. This is one of the two types of actions currently
    configured to be able to move drawers or doors that are a part of other geometry. """

    def __init__(self, axis, step, attach=status.NO_CHANGE):
        self.attach = attach
        self.axis = axis
        self.step = step
        self.offset_pose = np.eye(4)
        self.offset_pose[self.axis, 3] = self.step
        self.use_attached = False
    
    def __call__(self, world_state, actor, goal, batch_size=100, debug=False):
        # TODO ------
        actor_obj = world_state[actor]
        goal_pose = actor_obj.ee_pose.dot(self.offset_pose)
        end_ref = robot.ee_ref
        end_ref.set_pose(goal_pose)

        # Do inverse kinematics
        robot_ref = actor_obj.ref
        config = actor_obj.q
        _q = robot_ref.ik_solver(robot_ref,
                                 goal_pose,
                                 q0=config)
        
        return ActionParams(actor=actor, goal=goal, q=_q,
                            ee_pose=goal_pose,
                            attach=self.attach,
                            attach_to=goal,
                            linear_dist=self.step,
                            success=_q is not None)


class DrawerOffsetSampler(LinearSampler):
    """ Offset applied from goal params. This is one of the two types of actions currently
    configured to be able to move drawers or doors that are a part of other geometry. """

    def __init__(self, axis, open_drawer=True, attach=status.NO_CHANGE, ik_error_tol=5e-3):
        self.attach = attach
        self.axis = axis
        self.open_drawer = open_drawer
        self.offset_pose = np.eye(4)
        self.ik_error_tol = ik_error_tol
    
    def __call__(self, world_state, actor, goal, batch_size=100, debug=False):
        # TODO ------
        actor_obj = world_state[actor]    
        goal_obj = world_state[goal]
        step = goal_obj.q - goal_obj.ref.max if self.open_drawer else goal_obj.q - goal_obj.ref.min
        self.offset_pose[self.axis, 3] = step

        # Compute goal pose
        goal_pose = actor_obj.ee_pose.dot(self.offset_pose)
        end_ref = actor_obj.ref.ee_ref
        end_ref.set_pose_matrix(goal_pose)

        # Do inverse kinematics
        robot_ref = actor_obj.ref
        config = actor_obj.q
        _q = robot_ref.ik_solver(robot_ref,
                                 goal_pose,
                                 q0=config,
                                 tol=self.ik_error_tol)
        
        return ActionParams(actor=actor, goal=goal, q=_q,
                            ee_pose=goal_pose,
                            attach=self.attach,
                            attach_to=goal,
                            linear_dist=step,
                            success=_q is not None)
