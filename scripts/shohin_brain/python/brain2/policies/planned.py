# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import time

import brain2.utils.transformations as tra
import brain2.utils.status as status
from brain2.task.action import Policy
from brain2.utils.info import logerr, logwarn
from brain2.robot.trajectory import retime_trajectory

from brain2.motion_planners.rrt_connect import rrt_connect
from brain2.motion_planners.problem import MotionPlanningProblem
from brain2.utils.extend import simple_extend

# Import tools for grasping
from brain2.policies.grasping import get_relative_goal_discrete_sampler
from brain2.motion_planners.linear import LinearPlan, JointSpaceLinearPlan

# Conditions
from brain2.conditions.position import ObjPoseCondition


class AbstractPlanPolicy(Policy):

    FIND_GOAL = 0
    FIND_PLAN = 1
    EXECUTE = 2

    def __init__(self, step_size=0.02, delay=0):
        super(AbstractPlanPolicy, self).__init__()
        self.step_size = step_size
        self.reset()
        self._check = None
        self.stage = self.FIND_GOAL
        self.index = 0
        self.delay = delay

    def reset(self):
        self.plan = None
        self.stopped = False
        self.config = None
        self._finished = False
        self.stage = self.FIND_GOAL
        self.index = 0
        self.t = 0

    def enter(self, world_state, actor, goal):
        world_state[actor].set_goal(goal, None)
        self._replan(world_state, actor, goal)
        self.stage = self.FIND_GOAL
        self.index = 0
        self.t = world_state.time
        return status.SUCCESS

    def exit(self, world_state, actor, goal):
        self.plan = None
        self.stopped = False
        self.config = None
        self._finished = False
        self.stage = self.FIND_GOAL
        return status.SUCCESS

    def _replan(self, world_state, actor, goal):
        self._finished = False
        res = self._get_timed_plan(world_state, actor, goal)
        if res is not None:
            plan, timings = res
            timings = timings + world_state.time
            self.plan = plan, timings
            self.indices = np.arange(len(plan))
            if self.config is None:
                self.config = plan[-1]
            
            # Tell the rest of the system where we are going
            logwarn('got a new plan of length ' + str(len(plan)))
    
            # Visualize it if its long enough
            if len(plan) > 1:
                world_state[actor].ctrl.visualize_trajectory(world_state[actor],
                                                             plan, timings)
        else:
            # No plan found
            world_state[actor].set_goal(goal, None)
            self.plan = None
            logerr('computing linear motion plan failed!')

            # TODO REMOVE THIS 
            # raw_input('----')
        
        # have not safely stopped to start new trajectory
        self.stopped = False

    def step(self, world_state, actor, goal=None):
        
        # Take no action if we are waiting
        if self.delay > 0 and (world_state.time - self.delay) < self.t:
            return 0

        # world_state[actor].ctrl.arm.viz.send(world_state[actor].ee_pose)
        # Check plan, make sure it's still ok.
        if self._check and not self._check(world_state, goal):
            logerr("Goal object moved! Clearing plan.")
            self.stage = self.FIND_GOAL
            self.index = 0
            self.plan = None
        # Clear important information if we finished executing
        if self._finished:
            self.stage = self.FIND_GOAL
            self.index = 0
        # Execute the rest of the plan if it exists.
        if self.plan is None or self._finished:
            self._replan(world_state, actor, goal)
        else:
            # Get control interface 
            ctrl = world_state[actor].ctrl
            plan, timings = self.plan
            # print("Timings:", timings > world_state.time)
            # Time margin is to allow for timing delays and sensor noise
            # TODO we should not really need this
            if world_state.time > timings[-1]:
                idx = -1
                if world_state.time > timings[-1] + 1.:
                    self._finished = True
            else:
                idx = np.argmax(timings > world_state.time)
            # print(timings[idx], world_state.time)
            # Select and move
            T = ctrl.forward_kinematics(plan[idx])
            ctrl.go_local(T, self.config)

    def _get_motion_planning_problem(self, world_state, actor, goal, get_goal, **kwargs):
        robot = world_state[actor].ref
        pb_config = {
                'dof': robot.dof,
                'p_sample_goal': 0.2,
                'iterations': 100,
                'goal_iterations': 1,
                'verbose': 1, #self.verbose,
                'shortcut': True,
                'min_iterations': 10,
                'shortcut_iterations': 50,
                }
        is_done = lambda: False
        pb_config.update(kwargs)
        if goal is None:
            is_valid = lambda q: robot.validate(q, max_pairwise_distance=0.005)
        else:
            goal_ref = world_state[goal].ref  # Contains reference to sim obj
            is_valid = lambda q: (robot.validate(q, max_pairwise_distance=0.005)
                                  and goal_ref.is_visible())
        extend = lambda q1, q2: simple_extend(q1, q2, 0.2)
        return MotionPlanningProblem(sample_fn=robot.sample_uniform,
                                     goal_fn=get_goal,
                                     extend_fn=extend,
                                     is_valid_fn=is_valid,
                                     is_done_fn=is_done,
                                     config=pb_config,
                                     distance_fn=None)


class RelativeGoPlanned(AbstractPlanPolicy):
    def __init__(self, approach_offset=None, step_size=0.04,
            rotation_step_size=0.1, problem=None,
            goal_offsets=None, q_metric=None):
        super(RelativeGoPlanned, self).__init__(step_size)
    
        if approach_offset is not None:
            self.approach_offset = approach_offset
        else:
            self.approach_offset = np.eye(4)

        # Set up object grasping offsets
        if goal_offsets is None:
            # Only one offset and it's the identity
            self.goal_offsets = [np.eye(4)]
        else:
            self.goal_offsets = goal_offsets

        # Rotation steps
        self.rotation_step_size = rotation_step_size

        # Metric for scoring possible positions
        self.q_metric = q_metric
        self.problem = problem

        self.reset()

    def _get_timed_plan(self, world_state, actor, goal):
        """ generate a motion plan and timings. this will attempt to choose
        something that'll stay close to some preferred joint position. """

        #goal_pose = world_state[goal].pose.dot(self.pose_offset)
        #import time
        #while True:
        #    world_state[actor].ctrl.arm.viz.send(world_state[goal].pose)
        #    time.sleep(0.1)
        #    world_state[actor].ctrl.arm.viz.send(goal_pose)
        #    time.sleep(0.1)
        q0 = world_state[actor].q
        ee_pose = world_state[actor].ee_pose
        obj_pose = world_state[goal].pose
        roots = [obj_pose.dot(goal_offset) for goal_offset in self.goal_offsets]
        goals = [root.dot(self.approach_offset) for root in roots]
        actor_ref = world_state[actor].ref
        ik_solver = actor_ref.ik_solver
        options = []
        self._check = None
        if self.stage == self.FIND_GOAL:
            if self.q_metric is not None:
                for root, goal_pose, goal_offset in zip(roots, goals, self.goal_offsets):
                    # world_state[actor].ctrl.arm.viz.send(goal)
                    # raw_input("press enter")
                    q = ik_solver(actor_ref,
                                  goal_pose,
                                  q0=q0,
                                  pose_as_matrix=True)
                    # Check first position is reasonable. We also check to see if
                    # the object will stay visible.
                    world_state[actor].ctrl.arm.viz.send(goal_pose)
                    if (q is not None and world_state[actor].ref.validate(q) and
                        world_state[goal].ref.is_visible()):

                        qf = ik_solver(actor_ref,
                                root,
                                q0=q, # Search based on the previous estimate.
                                pose_as_matrix=True)
                    else:
                        qf = None
                    # Check final position is reasonable. If so, we will add this to
                    # our set of potential goals. For this one we do not need to
                    # make sure the object will still be visible.
                    world_state[actor].ctrl.arm.viz.send(root)
                    if (qf is not None and world_state[actor].ref.validate(qf, suppressed_objs=set([goal]))):
                        # Then this is a possible goal. Compute metric.
                        metric = self.q_metric(q0, q)
                        #metrics.append(np.linalg.norm(q - world_state[actor].q))
                        # print("pose =", goal_pose[:3, 3], "score =", metric)
                        options.append((metric, goal_pose, goal_offset, q))
            else:
                # "sorted" list
                metrics = range(len(goals))
                qs = [ik_solver(actor_ref,
                                     goal_pose,
                                     q0=q0,
                                     pose_as_matrix=True)
                      for goal_pose
                      in goals]
                options = [(m, g, go, q)
                           for (m, q, go, q)
                           in zip(metrics, goals, self.goal_offsets, qs)
                           if q is not None]

            # After computing goals, stop and on next cycle we'll find a plan
            self.options = options
            if len(self.options) > 0:
                self.stage = self.FIND_PLAN
                self.index = 0
            return None

        
        if len(self.options) < 1:
            self.stage = self.FIND_PLAN
            self.index = 0
            return None

        i = 0
        # Go over the sorted list
        # Generate plans in order, but don't take too long. We'll get there.
        tree = q0
        for score, goal_pose, root, q in sorted(self.options, key=lambda pair: pair[0]):
            print ("--->", i)
            if i < self.index:
                i += 1
                continue
            elif i > self.index:
                return None
            self.index += 1
            

            # print("!!!! pose =", goal_pose[:3, 3], "score =", score)
            world_state[actor].ctrl.arm.viz.send(goal_pose)

            if score > 1000:
                break

            #res = LinearPlan(world_state[actor], ee_pose, goal_pose,
            #                                  self.ik_solver,
            #                                  step_size=self.step_size,
            #                                  rotation_step_size=self.rotation_step_size)
            #print("!!!!!!!!!!")
            #print(goal_pose[:3, 3], q)
            path, tree = rrt_connect(tree,
                    self._get_motion_planning_problem(world_state,
                                                      actor,
                                                      goal,
                                                      lambda: q))
            if path is not None:
                res = retime_trajectory(world_state[actor], path)
            else:
                res = None
            if res is not None:
                world_state[actor].set_goal(goal, root)
                self._check = ObjPoseCondition(world_state[goal].pose,
                                               pos_tol=0.1,
                                               theta_tol=np.radians(180),
                                               relative=False,
                                               verbose=True)
                return res
        # If this was a total failure
        if self.index == len(self.options):
            self.index = 0
            self.stage = self.FIND_GOAL
        return None


class WaypointGoPlanned(AbstractPlanPolicy):
    """ Go to config, taking reasonable steps in cartesian space """
    def __init__(self, config=None, step_size=0.1, delay=0.):
        super(WaypointGoPlanned, self).__init__(step_size, delay)
        self.waypoint = config
        self.reset()

    def _get_timed_plan(self, world_state, actor, goal):
        """ generate a motion plan and timings """
        #return JointSpaceLinearPlan(world_state[actor],
        #                   world_state[actor].q,
        #                    self.config,
        #                    self.step_size)
        path, tree = rrt_connect(world_state[actor].q,
                                self._get_motion_planning_problem(world_state,
                                actor,
                                None,
                                lambda: self.config))
        if path is not None:
            return retime_trajectory(world_state[actor], path)
        else:
            return None

    def enter(self, world_state, actor, goal=None):
        if self.waypoint is None:
            self.config = world_state[actor].ref.home_q
        else:
            self.config = self.waypoint

        self._replan(world_state, actor, goal)
        self.t = world_state.time

        return status.SUCCESS


class LiftObject(AbstractPlanPolicy):
    """
    This policy will lift an object up off off a surface. It's just to break contact between the
    object and the surface.
    """
    def __init__(self, step_size=0.04, plan_length=0.2):
        self.step_size = step_size
        self.plan_length = plan_length

    def enter(self, world_state, actor, goal):
        return status.SUCCESS

    def exit(self, world_state, actor, goal):
        return status.SUCCESS

    def _get_timed_plan(self, world_state, actor, goal):
        ee_pose = world_state[actor].ee_pose
        goal_pose = np.copy(ee_pose)
        # Just move up in the world Z axis
        goal_pose[2,3] += self.plan_length
        actor_ref = world_state[actor].ref
        ik_solver = actor_ref.ik_solver
        # Create plan and return it
        return LinearPlan(world_state[actor],
                              ee_pose,
                              goal_pose,
                              ik_solver,
                              step_size=self.step_size,
                              suppressed_objs=set([goal]))


class BlockingGraspObject(Policy):
    """
    This is a blocking grasp policy. It'll move to take the object, close the
    gripper, and withdraw when done.

    ik_solver: what we use to get IK
    step_size: size of positions to sample
    retreat: should we back off / withdraw after grasping?
    """
    def __init__(self, step_size=0.04, retreat=True, ignore_objs=None):
        self.step_size = step_size
        self.retreat = retreat
        self.ignore_objs = ignore_objs

    def enter(self, world_state, actor, goal):
        return status.SUCCESS

    def exit(self, world_state, actor, goal):
        return status.SUCCESS

    def step(self, world_state, actor, goal):
        actor_state = world_state[actor]
        ee_pose = actor_state.ee_pose
        goal_pose = actor_state.get_goal(goal)
        if actor_state.goal_is_relative:
            goal_pose = world_state[goal].pose.dot(goal_pose)
        if goal_pose is None:
            raise RuntimeError('never planned for this! ' + str(goal))

        actor_ref = actor_state.ref
        ik_solver = actor_ref.ik_solver
    
        if self.ignore_objs is not None:
            suppress = set([goal, self.ignore_objs])
        else:
            suppress = set([goal])

        # Get a motion plan
        res1 = LinearPlan(actor_state,
                              ee_pose,
                              goal_pose,
                              ik_solver,
                              step_size=self.step_size,
                              suppressed_objs=suppress)
        if res1 is not None:
            plan1, t1 = res1
            res2 = retime_trajectory(actor_state, plan1[::-1])
        else:
            logerr('planning failed step 1')
            # raw_input('---')
            res2 = None

        # Make sure both are valid and then execute
        if res2 is not None:
            plan1, t1 = res1
            plan2, t2 = res2
            #print("plan1 len =", len(plan1))
            #print("plan2 len =", len(plan2))
            actor_state.ctrl.execute_joint_trajectory(plan1, t1, wait_at_end=False)
            actor_state.ctrl.close_gripper(wait=True)
            actor_state.ctrl.execute_joint_trajectory(plan2, t2, wait_at_end=False)
            actor_state.attach(goal)
        else:
            logwarn("planning failed for linear motion: " + str(actor) + ", " +
                    str(goal))
            # raw_input('---')

        actor_state.clear_goal()
        return status.SUCCESS
        

class BlockingPlannedMotion(Policy):
    """Planned motion that moves to a position based on heuristic (chooses
    the closest grasp only)."""
    
    def __init__(self, domain, lookup, metric, p_new_ik=0.5):
        """ This should call the reset function """
        super(BlockingPlannedMotion, self).__init__()
        self.domain = domain
        self.lookup = lookup
        self.metric = metric
        self.p_new_ik = p_new_ik
        self.ik_solutions = set()

    def reset(self):
        self.problem = None
        self.trajectory = None

        # Store the parameters for the initial obs goal etc
        self.start_pose = None
        self.start_config = None
        self.start_goal = None

        self.prev_ik_solution = None
        self.ik_solutions = []

    def enter(self, world_state, actor, goal):
        """Get the specific stuff we're planning about"""

        self.reset()
        robot = world_state[actor]
        goal_obj = world_state[goal]
        actor_ref = world_state[actor].ref
        ik_solver = actor_ref.ik_solver

        # Sample grasps/waypoints/whatever
        poses = self.lookup(world_state, goal)

        # Choose the best according to our metric
        grasp, vals = self.metric(robot.ee_pose,
                                         robot.inv_ee_pose,
                                         goal_obj.pose,
                                         poses)

        offset = np.eye(4)
        offset[2,3] = -1.
        appr = grasp.dot(offset)
        final_ee_pose = goal_obj.pose.dot(appr)

        # Find inverse kinematics -- sample goals
        self.start_goal = final_ee_pose
        _sample_goal = get_relative_goal_discrete_sampler(ik_solver,
                                                          robot.ref,
                                                          goal_obj.pose,
                                                          [appr],
                                                          config=robot.q,
                                                          metric=self.metric,
                                                          standoff=0,)
        # Do not need the "done" function
        _is_done = lambda q: False

        # Create problem and plan to get there
        if ik_solver is None:
            raise RuntimeError('ik solver not defined')
        ik_solver.reset()
        problem = self.domain.get_default_planning_problem(_sample_goal,
                                                           _is_done,
                                                           actor)
        self.trajectory, tree = rrt_connect(world_state[actor].q, problem)
        print("trajectory =", self.trajectory)


    def step(self, world_state, actor, goal):
        """ one execution of the algorithm """
        if self.trajectory is not None:
            robot = world_state[actor]
            print(" -- stepping:", self.trajectory)
            for i, q in enumerate(self.trajectory):
                robot.ref.set_joint_positions(q)
                time.sleep(0.1)
