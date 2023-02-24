# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import os
import pdb
import time

# Bullet-specific stuff
import brain2.bullet.problems as problems
from brain2.bullet.interface import BulletInterface
from brain2.policies.planned import BlockingPlannedMotion
from brain2.robot.domain import RobotDomainDefinition
import brain2.utils.axis as axis
from brain2.utils.pose import make_pose
from brain2.datasets.acronym import GraspDataset

# Other???
import torch
import brain2.learning.quaternion as quaternion

# Motion planning tools
from brain2.motion_planners.problem import MotionPlanningProblem
from brain2.motion_planners.rrt_connect import rrt_connect
from brain2.utils.extend import simple_extend

# Task model and task planning stuff
from brain2.task.action import Policy 
from brain2.conditions.approach import ApproachRegionCondition
from brain2.policies.planned import WaypointGoPlanned
from brain2.policies.planned import BlockingGraspObject
from brain2.policies.gripper import BlockingOpenGripper, BlockingCloseGripper
from brain2.robot.lookup_table import LookupTable
from brain2.robot.samplers import *
from brain2.robot.connectors import *
from brain2.robot.heuristics import *


def load_object(iface, assets_path, obj_name, obj_type, table=None):
    """Load an individual object"""
    problems.load_simple_object(iface, assets_path, obj_name, obj_type)
    obj = iface.get_object(obj_name)
    if table is not None:
        p, r = table.sample_surface_pose("top")
        obj.set_pose(p, r, wxyz=True)
    return obj


def create_env(assets_path, gui=False, add_ground_plane=False):
    """ Create bullet environment representing the whole world """

    # Create bullet backend for planning
    iface = BulletInterface(gui=gui, add_ground_plane=add_ground_plane)
    # Load the robots
    robot = problems.load_franka(iface, assets_path, "robot", camera="d415",
            padding=0.05)
    # Load the cart
    table = problems.load_simple_cart(iface, assets_path)
    robot.set_allowed_collisions(table, idx=0)
    robot.set_allowed_collisions(table, idx=-1)
    #robot.set_pose(np.array([0.15, 0.0, 0.7]),
    table.set_pose(np.array([0.35, 0, -0.35]), 
                    np.array([1, 0, 0, 0]))
    pos, quat = [0.509, 0.993, 1.542], [-0.002, 0.823, -0.567, -0.010]
    iface.set_camera((pos, quat), matrix=False)
    hand = iface.load_urdf(name="left",
                            pkg=assets_path,
                            fixed_base=True,
                            model="hand_marker.urdf")
    hand = iface.load_urdf(name="right",
                            pkg=assets_path,
                            fixed_base=True,
                            model="hand_marker.urdf")
    return iface

class GraspLookupTable(LookupTable):
    """ 
    """
    def __init__(self, max_dt=0.1, *args, **kwargs):
        super(GraspLookupTable, self).__init__(*args, **kwargs)
        self.objs = {}
        self.max_dt = max_dt

    def update_grasps_for_obj(self, obj, grasps, t=0., scores=None):
        if scores is None:
            scores = np.zeros(len(grasps))
        self.objs[obj] = (t, grasps, scores)


    def get_scores(self, ws, obj):
        return self.objs[obj][2]

    def __call__(self, ws, obj, actor=None):
        time_constraint = True
        if obj in self.objs and time_constraint:
            return self.objs[obj][1]
        else:
            return []

    def has(self, ws, obj):
        if not obj in self.objs:
            return False
        else:
            t, grasps = self.objs[obj]
            # TODO check time
            return True


class ReactiveApproachPolicy(Policy):
    """
    Reactive approach policy. Needs to be given a grasp planner.
    """

    presets = ["default", "graspnet", "once"]

    def __init__(self, grasps, pos_wt=1., rot_wt=0.5, step_size=0.1,
            offset=0.2, max_opts=10, hand="left", preset="default"):
        self.grasps = grasps
        self.ignore_objs = hand
        self.hand = hand
        self.metric = quaternion.WeightedPoseDistance(position_wt=pos_wt,
                                                      orientation_wt=rot_wt)
        self.step_size = step_size
        self.reset()
        self.offset_pose = np.eye(4)
        self.offset_pose[2, 3] = offset
        self.offset_size = -1*offset
        self.max_opts = max_opts
        self.max_track_step = 0.2

        # switch over presets
        if preset not in self.presets:
            raise RuntimeError("preset option not specified; options are "
                               + str(self.presets) + " and was given " +
                               str(preset))
        if preset == "default":
            self.height_penalty = 0.
            self.penalty_wt = 0.
            self.home_wt = 5.
            self.score_threshold = 0.5
            self.score_wt = 1.
            self.base_wt = 5.
            self.init_base_wt = self.base_wt
        elif preset == "graspnet":
            # This preset will keep choosing the closest pose to wherever it
            # currently is, after choosing the best based on score
            # Originally there was a "graspnet" one but it was unsafe
            self.height_penalty = 0.
            self.penalty_wt = 0.
            self.home_wt = 0.
            self.score_threshold = 1.
            self.score_wt = 1.
            self.base_wt = 1000.
            self.init_base_wt = 0.
        else:
            raise RuntimeError('preset not recognized and something went wrong'
                               'with checks. preset was: ' + str(preset))

        self.ee_pose = make_pose([0.4, -0.043, 0.507],
                                 [-0.006, 0.979, 0.028, 0.200])

        #- Translation: [-0.003, 0.078, 0.279]
        #- Rotation: in Quaternion [0.858, 0.189, 0.158, -0.451]
        #self.q_pref = torch.Tensor(np.array([[0., 0.08, 0.28, 0.858, 0.189,
        #    0.158, -0.451]]))
        # - Translation: [0.350, -0.043, 0.507]
        # - Rotation: in Quaternion [-0.006, 0.979, 0.028, 0.200]
        #    in RPY (radian) [3.084, 0.403, -3.142]
        #    in RPY (degree) [176.710, 23.092, -179.998]
        # STORED AS:
        # XYZ + QUATERNION
        self.q_pref = torch.Tensor(np.array([[0.350, -0.043, 0.507,
                                              -0.006, 0.979, 0.028, 0.200]]))

        self.adjust_goals = True
        self.generate_linear_plan = False

    def reset(self):
        # Tracks the trajectory we are currently following
        self.plan = None
        self.last_cfg = None
        self.q_goal = None

    def enter(self, ws, actor, goal):
        self.reset()
        robot_state = ws[actor]
        robot_ref = robot_state.ref
        self.last_cfg = robot_ref.home_q
        self.goal_pose = robot_state.ee_pose
        if self.adjust_goals:
            self.goal_pose[:3, 3] = np.zeros(3)
        return True

    def exit(self, ws, actor, goal):
        self.reset()
        return True

    def step(self, ws, actor, goal):
        robot_state = ws[actor]
        goal_state = ws[goal]
        hand_state = ws[self.hand]

        # get objects
        robot_ref = robot_state.ref
        end_ref = robot_ref.ee_ref
        ctrl = robot_state.ctrl

        # Find our reference pose
        # We will try to find a good pose close to this one as much as possible
        pose = self.goal_pose
        if self.adjust_goals:
            pose = np.copy(pose)
            pose[:3, 3] += goal_state.pose[:3, 3]
        if self.q_goal is None:
            q0 = self.q_pref
            # Track here 
            self._track(robot_state, goal_state, ctrl)
            _tracking = True
            base_wt = self.init_base_wt
        else:
            _tracking = False
            base_wt = self.base_wt
            q0 = self.q_goal

        opts = self.grasps(ws, goal)
        scores = self.grasps.get_scores(ws, goal)
        batch_size = len(opts)
        q0 = torch.Tensor(q0).repeat(batch_size, 1)

        # If this is empty, we're actually just done here
        if batch_size == 0:
            return 0

        # Otherwise let's say how many grasps we have
        print("IN APPROACH: #grasps =", len(opts))

        # This is just because I foolishly implemented the pose math in torch
        if not isinstance(opts, torch.Tensor):
            opts = torch.Tensor(opts)
        if not isinstance(scores, torch.Tensor):
            scores = torch.Tensor(scores)

        # high scores are GOOD
        # We subtract the threshold, and penalize anything that has a negative
        # score very heavily (because these are bad grasps)
        scores = torch.nn.functional.relu(-1 * (scores - self.score_threshold))

        # Apply offset to quaternions
        q1 = np.array([0, 0, self.offset_size, 0, 0, 0, 1])
        q1 = torch.Tensor(q1).repeat(batch_size, 1)
        opts = quaternion.compose_qq(opts, q1)

        penalty = torch.zeros(len(opts))
        z = opts[:,2] - hand_state.pose[2,3]
        penalty[z < 0] = self.height_penalty
        base_cost = self.metric(q0, opts)
        costs = ((base_cost * base_wt)
                + (self.home_wt * self.metric(self.q_pref, opts))
                + (self.penalty_wt * penalty)
                + (self.score_wt * scores))

        costs = costs.numpy()
        opts = opts.numpy().tolist()
        goal_cfg = None
        
        best_idx = np.argmin(costs)
        costs = costs.tolist()
        best_cost = costs[best_idx]
        best_opt = opts[best_idx]

        # This block is for what happens if you have a lot of examples and NEEd
        # to find one but need to do it quickly. Currently disabled
        if len(costs) > self.max_opts and False:
            idx = np.arange(len(costs))
            np.random.shuffle(idx)
            costs = [costs[i] for i in idx[:self.max_opts]]
            opts = [opts[i] for i in idx[:self.max_opts]]

        costs.append(best_cost)
        opts.append(best_opt)

        if self.ignore_objs is not None:
            suppress = set([goal, self.ignore_objs])
        else:
            suppress = set([goal])

        # Support the parallel IK solver
        ik_done = False
        if robot_ref.ik_solver.is_parallel():
            ik_done = True
            opts = [opt[:3], opt[3:] for opt in opts]
            iks = robot_ref.ik_solver.solve_batch(opts,
                    q0s=([robot_state.q] * len(opts)),
                    robot_ref=robot_ref,
                    pose_as_matrix=False)
            costs = [((np.linalg.norm(robot_state.q - q))
                    + (self.home_wt * np.linalg.norm(robot_ref.home_q - q)))
                    if q is not None else float('inf')
                    for q in iks]
            raise NotImplementedError('THIS IS UNSAFE')

        cfgs = []
        best_opt = None
        print("Loop over", len(opts), "options:")
        for i, (c, opt) in enumerate(sorted(zip(costs, opts), key=lambda p: p[0])):

            print(i, c)

            if i >= self.max_opts:
                goal_pose = None
                ik_cfg = None
                goal_cfg = None
                print(i, "reached max options")
                break

            # Examine each one in order 
            offset_goal = tra.quaternion_matrix(opt[3:])
            offset_goal[:3, 3] = opt[:3]
            goal_pose = offset_goal.dot(self.offset_pose)

            # Show where we are trying to move to
            end_ref.set_pose_matrix(offset_goal)
            if self.generate_linear_plan:
                plan = LinearPlan(robot_state,
                                  robot_state.ee_pose,
                                  offset_goal,
                                  robot_ref.ik_solver,
                                  step_size=self.step_size)
            else:
                if ik_done:
                    ik_cfg = iks[i]
                else:
                    #ik_cfg = robot_ref.ik_solver(robot_ref, offset_goal, q0=robot_ref.home_q)
                    ik_cfg = robot_ref.ik_solver(robot_ref, offset_goal, q0=robot_state.q)
                if ik_cfg is None or not robot_ref.validate(ik_cfg):
                    print(i, "inverse kinematics failed")
                    continue

                # Try to generate linear motion to this config without
                # collisions!
                # Timings are already given
                plan = JointSpaceLinearPlan(robot_state,
                                            robot_state.q,
                                            ik_cfg,
                                            self.step_size)

            if plan is None:
                print(i, "plan to offset failed")
                continue
            ik_cfg = plan[0][-1]
            cost = np.sum((ik_cfg - robot_ref.home_q) ** 2)

            linear_plan = LinearPlan(robot_state,
                                    offset_goal,
                                    goal_pose,
                                    robot_ref.ik_solver,
                                    step_size=self.step_size,
                                    suppressed_objs=suppress)
            if linear_plan is None:
                print(i, "plan to grasp failed")
                continue

            goal_cfg = ik_cfg
            plan, timings = plan
            timings = timings + ws.time
            plan = plan, timings
            best_opt = opt
            # cfgs.append((cost, opt, goal_cfg, goal_pose, plan))
            break

        # If all of our current options are bad, we will do nothing
        # But we can move home and reset our current goal that we are tracking
        if goal_cfg is None or plan is None:
            # revert to home 
            self.last_cfg = robot_ref.home_q
            self.goal_pose = robot_state.ee_pose
            if self.adjust_goals:
                self.goal_pose[:3, 3] = np.zeros(3)

            # Clear goals and go home
            robot_state.clear_goal()

            # Compute center of hand and track it as it moves
            if not _tracking:
                self._track(robot_state, goal_state, ctrl)

            return 0

        self.goal_pose = np.copy(goal_pose)
        if self.adjust_goals:
            self.goal_pose[:3, 3] -= goal_state.pose[:3, 3]
        self.plan = plan
        self.last_cfg = goal_cfg
        self.q_goal = best_opt

        # Record the goal pose so that we know what to do next
        robot_state.set_goal(goal, goal_pose, relative=False)
        print(">>>>", robot_state.goal_obj, "goal was set")
        end_ref.set_pose_matrix(goal_pose)
        robot_ref.set_joint_positions(goal_cfg)

        # Try to actually move
        if ctrl is not None:
            plan, timings = self.plan
            idx = min((len(plan)/2)+1, len(plan)-1)
            idx = -1
            ctrl.go_local(q=plan[idx], speed="slow")

    def _track(self, robot_state, goal_state, ctrl):
        # Let's get this reference thingy again
        robot_ref = robot_state.ref 

        # Compute center of hand and track it as it moves
        x, y, z = goal_state.pose[:3,3]
        x = max(0.4, min(0.55, x-0.15))
        y = max(min(y, 0.3), -0.3)
        z = min(0.7, max(z + 0.15, 0.1))
        new_track_xyz = np.array([x, y, z])
        track_xyz = robot_state.ee_pose[:3, 3]
        track_dir = new_track_xyz - track_xyz
        track_dist = np.linalg.norm(track_dir)
        if track_dist > self.max_track_step:
            track_xyz = track_xyz + (track_dir / track_dist * self.max_track_step)
        else:
            track_xyz = new_track_xyz

        if ctrl is not None:
            #ctrl.go_local(q=robot_ref.home_q, speed="slow")
            track_pose = np.copy(self.ee_pose)
            track_pose[:3, 3] = track_xyz
            track_q = robot_ref.ik_solver(robot_state, track_pose, q0=robot_state.q)
            if track_q is not None and robot_ref.validate(track_q):
                ctrl.go_local(T=track_pose, q=track_q, speed="slow")
                # ctrl.go_local(T=track_pose, speed="slow)
    

class HandoverDomainDefinition(RobotDomainDefinition):
    def __init__(self, assets_path, gui=False, ctrl=None, ctrl2=None, ik_solver=None,
                 test_mode=False, hand="left",
                 grasp_dir='/data/NVGraspDataset_clemens_complete', 
                 shapenet_dir='/data/ShapeNetCore.v2',
                 semantic_dir='/data/ShapeNetSemantic'):

        # Create bullet interface and get objects
        self.ik_solver = ik_solver
        iface = create_env(gui=gui, assets_path=assets_path)
        table = iface.get_object("table")
        if test_mode:
            grasps = GraspDataset(preload=False, 
                grasp_dir=grasp_dir,
                shapenet_dir=shapenet_dir,
                semantic_dir=semantic_dir)
            filename = "Mug_85d5e7be548357baee0fa456543b8166_0.013915745438755706.json"
            obj_grasps = grasps.get_grasps_from_file(filename)
            iface.load_obj("obj",
                    pkg=None,
                    model=obj_grasps.mesh_filename,
                    scale=obj_grasps.scale)
        else:
            iface.load_urdf(name="obj",
                            pkg=assets_path,
                            model="small_ball.urdf")
            obj_grasps = []
        self.obj_grasps = obj_grasps

        # Define objects to be included in the world state
        # Should probably be used to parse for loading in the future but
        # whatever
        objs = {
                "robot": {"control": ctrl},
                "table": {},
                "obj": {
                        "obj_class": "shapenet",
                        "grasp_data": obj_grasps,
                        },
                }
        self.robot = "robot"
        self.hands = ["left", "right"]
        self.manipulable_objs = ["obj"]
        self.handles = []

        # Create everything, define policies and predicates
        super(HandoverDomainDefinition, self).__init__(iface, objs, hands=self.hands)
        self.root[self.robot].ref.set_ik_solver(ik_solver)
        self.grasps = GraspLookupTable()
        self.add_operators(ik_solver, hand=hand)
        self.compile()

    def add_operators(domain, ik, hand="left"):
        actor = domain.robot
        hand_appr_dist = -0.15 # How far away we go to stand-off from the hand
        approach_region = ApproachRegionCondition(approach_distance=1.5*hand_appr_dist,
                                                  approach_direction=axis.Z,
                                                  verbose=True,
                                                  slope=25.,
                                                  pos_tol=1.5e-2,
                                                  max_pos_tol=3e-2,
                                                  theta_tol=np.radians(45))

        domain.add_relation("in_approach_region", approach_region, domain.robot, domain.manipulable_objs)
        def too_close(ws, x, hand):
            if not ws[x].observed or not ws[hand].observed: return False
            pt1 = ws[x].ee_pose
            pt2 = ws[hand].pose
            if pt1 is None or pt2 is None: return False
            # Check for a distance slightly smaller than approach distance
            dist = np.linalg.norm(pt1[:3, axis.POS] - pt2[:3, axis.POS])
            limit = (abs(hand_appr_dist) - 0.03)
            return dist < limit
        domain.add_relation("too_close_to_hand", too_close, domain.robot, domain.hands)



        # TODO: implement this
        is_free = lambda ws, x: True
        domain.add_property("is_free", is_free, domain.robot)

        drop_q = np.array([-0.5148817479066682, 0.5074429247002853,
            0.28036785539827846, -2.1822740914544294, 0.019065026521645986,
            3.13, 0.35])
        def at_drop(ws, x):
                x = ws[x]
                return None if x.q is False else np.all(np.abs(x.q[:7] - drop_q[:7]) < 0.1)
        domain.add_property("at_drop", at_drop, domain.robot)

        # Approach one step at a time
        approach_obj = ReactiveApproachPolicy(domain.grasps,
                                              offset=np.abs(hand_appr_dist),
                                              hand=hand)
        domain.add_operator("approach_obj", policy=approach_obj,
                        preconditions=[
                            ("in_approach_region(%s, {})" % actor, False),
                            ("observed({})", True),
                            ("has_anything(%s)" % domain.robot, False),
                            ("is_free(%s)" % domain.robot, True),
                            ("hand_over_table(%s)" % hand, True),
                            ("gripper_fully_closed(%s)" % actor, False),
                        ],
                        effects=[
                            ("in_approach_region(%s, {})" % actor, True),
                            ("is_free(%s)" % domain.robot, False),
                            ("has_goal(%s)" % domain.robot, True),
                            ("at_home(%s)" % actor, False),
                            ("observed({})", False),
                        ],
                        task_planning=True,
                        to_entities=domain.manipulable_objs+domain.handles,
                        subgoal_sampler=DiscreteGoalSampler("grasp",
                                                            standoff=0.1,
                                                            cost=BasicCost(),
                                                            attach=status.NO_CHANGE),
                        subgoal_connector=RRTConnector(),
                        )
        # Closed-loop grasping to current grasp pose
        grasp_obj = BlockingGraspObject(step_size=0.05, retreat=True,
                                        ignore_objs=hand)
        domain.add_operator("grasp_obj", policy=grasp_obj,
                            preconditions=[
                                ("in_approach_region(%s, {})" % actor, True),
                                ("has_anything(%s)" % actor, False),
                                ("has_goal(%s)" % domain.robot, True),
                                ("gripper_fully_closed(%s)" % actor, False),
                            ],
                            effects=[
                                ("has_anything(%s)" % actor, True),
                                ("has_obj(%s, {})" % (domain.robot), True),
                                ("in_approach_region(%s, {})" % actor, False),
                                ("is_free(%s)" % domain.robot, True),
                                ("has_goal(%s)" % domain.robot, False),
                                ("at_home(%s)" % actor, False),
                            ],
                            task_planning=True,
                            planning_cost=0,
                            to_entities=domain.manipulable_objs,)

        # Open gripper if it was empty.
        open_gripper = BlockingOpenGripper()
        domain.add_operator("open_gripper_grasp_failed", policy=open_gripper,
                        preconditions=[
                            ("has_anything(%s)" % actor, True),
                            ("gripper_fully_closed(%s)" % actor, True)],
                        effects=[
                            ("has_anything(%s)" % actor, False),
                            ("has_goal(%s)" % domain.robot, False),
                            ("gripper_fully_closed(%s)" % actor, False)],
                        task_planning=True,
                        )
        domain.add_operator("open_gripper", policy=open_gripper,
                        preconditions=[
                            ("gripper_fully_closed(%s)" % actor, True)],
                        effects=[
                            ("gripper_fully_closed(%s)" % actor, False),
                            ("has_goal(%s)" % domain.robot, False),
                            ],
                        task_planning=True,
                        )
 

        # Wait for the human to present a block with a particular hand.
        # Just go home. Can be interrupted.
        go_home = WaypointGoPlanned(step_size=0.25, delay=3.)
        domain.add_operator("go_home",
                            policy=go_home,
                            preconditions=[
                                ("gripper_fully_closed(%s)" % actor, False),
                                ],
                            effects=[
                                     ("at_home(%s)" % actor, True),
                                     ("has_goal(%s)" % domain.robot, False),
                                     ],
                            task_planning=True,)
        domain.add_operator("wait_for_object",
                            policy=go_home,
                            preconditions=[
                                ("observed({})", False),
                                ("gripper_fully_closed(%s)" % actor, False),
                                ],
                            effects=[("observed({})", True),
                                     ("at_home(%s)" % actor, True),
                                     ("has_goal(%s)" % domain.robot, False),
                                    ],
                            task_planning=True,
                            to_entities=domain.manipulable_objs)
        domain.add_operator("wait_for_human_with_object",
                            policy=go_home,
                            preconditions=[
                                    ("hand_over_table({})", False),
                                    ("gripper_fully_closed(%s)" % actor, False),
                                    ],
                            effects=[("hand_over_table({})", True),
                                     ("stable({})", True),
                                     ("observed({})", True),
                                     ("has_anything(%s)" % actor, False),
                                     ("at_home(%s)" % actor, True),
                                     ("has_goal(%s)" % domain.robot, False),
                                     ("hand_has_obj({})", True)],
                            task_planning=True,
                            to_entities=domain.hands)

        # Drop the object at the drop position.
        # First goes to goal position, and if it makes it there it will open up
        # the gripper and be done.
        go_to_drop = WaypointGoPlanned(config=drop_q, step_size=0.25)
        domain.add_operator("go_to_drop",
                            policy=go_to_drop,
                            preconditions=[
                                    ("has_anything(%s)" % actor, True),
                                    ("at_drop(%s)" % actor, False)],
                            effects=[("at_drop(%s)" % actor, True)],
                            task_planning=True,)
        drop_object = BlockingOpenGripper()
        domain.add_operator("drop_object",
                            policy=go_to_drop,
                            preconditions=[
                                  ("at_drop(%s)" % actor, True),
                                  ("has_anything(%s)" % actor, True)
                                  ],
                            effects=[
                                ("has_anything(%s)" % actor, False)
                                ],
                            task_planning=True,
                            planning_cost=0,)


def create_scene(iface, domain):
    # Get a modifiable copy of the world state
    world_state = domain.root.fork()
    # Create positions, and move the robots around so they aren't in collision
    # any more to start out with.
    q1 = np.array([0.012, -0.57, 0., -2.8, 0., 3., 0.74])
    # q2 = np.array([0., -0.2, 0, -0.1, 0, 0.5, 0.24])
    x = np.random.randn() * 0.1 + 0.6
    y = np.random.randn() * 0.1
    z = np.random.randn() * 0.1 + 1.1
    dx, dy, dz = np.random.rand(3) * 0.02 + 0.01
    dx += 0.04
    world_state["obj"].set_base_pose_quat([x, y, z], [0, 0, 0, 1])
    world_state["obj"].observed = True
    world_state["table"].set_base_pose_quat([0.35, 0, 0.35], [1, 0, 0, 0])
    world_state["table"].observed = True
    world_state["robot"].set_config(q1)
    world_state["robot"].set_base_pose_quat([0, 0, 0.7], [0, 0, 0, 1])
    world_state["robot"].observed = True
    world_state["left"].set_base_pose_quat([x+dx, y+dy, z+dz], [0, 0, 0, 1])
    world_state["left"].observed = True
    iface.update(world_state)
    domain.update_logical(world_state)
    return world_state


def sample_grasps(ws, obj_grasps, batch_size=50):
    pose = ws["obj"].pose
    poses = obj_grasps.grasps
    #print(poses.shape)
    q0 = np.zeros((1, 7))
    q0[0,:3] = pose[:3, 3]
    q0[0,3:] = tra.quaternion_from_matrix(pose)
    q0 = torch.Tensor(q0).repeat(batch_size, 1)
    batch = np.zeros((batch_size, 7))
    idx = np.arange(poses.shape[0])
    np.random.shuffle(idx)
    for i, j in zip(range(batch_size), idx):
        batch[i] = poses[j]
    q1 = torch.Tensor(batch)
    batch = quaternion.compose_qq(q0, q1)
    return batch

