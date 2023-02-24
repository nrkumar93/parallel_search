# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

import heapq
import numpy as np
import brain2.utils.info as info

# This is only necessary for execution
from brain2.task.execution import PlanExecutionPolicy

class PlanningNode(object):
    '''
    Create search nodes. These have children determined by operator.
    '''

    def __init__(self, key, world_state, parent, opname, op, op_args, depth):
        self.world_state = world_state
        self.key = key
        self.depth = depth
        self.parent = parent
        self.opname = opname
        self.op = op
        self.op_args = op_args

        # relaxed L -- never undo progress
        self.p_correct = 0
        self.p_relaxed = 0
        self.heuristic = 0
        self.goal_sat = None
        self.relaxed_goal_sat = None

        # Track the tree search
        self.expanded = False
        self.children = []

        # Store some information about how to get here
        self.params = None

    def apply_goal(self, domain, goal):
        L = self.world_state.logical_state
        R = self.world_state.relaxed_state
        self.relaxed_goal_sat = np.zeros(len(goal))
        self.goal_sat = np.zeros(len(goal))

        if self.parent is not None:
            assert np.sum(R) >= np.sum(self.parent.world_state.relaxed_state)
            #print(np.sum(R), np.sum(self.parent.world_state.relaxed_state))
        
        for i, (predicate, val) in enumerate(goal):
            predicate_idx = domain.predicate_idx[predicate]
            res = L[predicate_idx]
            self.goal_sat[i] = 1 if res == val else 0

            # For relaxed -- do not ever lose progress
            # We will be tracking both negative and positive expressions
            offset = 0 if val else L.shape[0]
            res = R[predicate_idx + offset]
            if val and res == val:
                self.relaxed_goal_sat[i] = 1

        self.p_relaxed = np.sum(self.relaxed_goal_sat)
        self.p_correct = np.sum(self.goal_sat)

        if self.parent is not None:
            assert(self.p_relaxed >= self.parent.p_relaxed)

    def __lt__(self, other):
        return self.depth < other.depth


def relaxed_heuristic(node):
    return np.sqrt(node.depth) - node.p_relaxed


def relaxed_only_heuristic(node):
    return -node.p_relaxed


def exact_heuristic(node):
    return np.sqrt(node.depth) - node.p_correct


def depth_heuristic(node):
    return node.depth


class TaskPlanner(object):
    """
    This contains nodes associated with a particular task planning problem. When
    called, it will reset itself and perform a greedy A* search out to max
    depth.
    """

    def __init__(self, domain, heuristic=None, use_relaxed_state=True, max_steps=50000, verbose=0,
                 relaxed_search=False, greedy=False):
        """
        Set up the planner. Takes in a heuristic function.
        """
        self.domain = domain
        self.verbose = verbose
        self.use_relaxed_state = use_relaxed_state or relaxed_search
        self.relaxed_search = relaxed_search
        self.greedy = greedy
        if heuristic is None:
            if self.use_relaxed_state:
                self.heuristic = relaxed_heuristic
            else:
                self.heuristic = exact_heuristic
        else:
            self.heuristic = heuristic

        self.max_steps = max_steps
        self.reset(None)

    def reset(self, root):
        self.root = root
        self.nodes = {}

    def hash_logical(self, logical_state):
        """ Generate unique hash for world states """
        val = 0
        for i, predicate in enumerate(logical_state):
            val += (int(predicate) << i)
        return val

    def get_compiled_planning_ops(self):
        """ Get a list of the highest-level operators we'll be using to build our graph of possible
        high-level actions for execution. """
        compiled_ops = {}
        for op in self.domain.planning_operators:
            for compiled_op in self.domain.operators[op].compiled_ops:
                compiled_ops[compiled_op] = self.domain.compiled_op[compiled_op]
        return compiled_ops

    def push_world_state(self, world_state, parent, opname, op, op_args, goal,
                         step=0):
        """
        Create node for this world state and add it to the priority heap. If
        this node already exists then forget about it.
        """
        depth = 0
        if parent is not None:
            depth = parent.depth + op.cost

        # compute hash for the logical state
        # we use logical state to lookup the actual (possibly continuous)
        # world state expectation
        if self.relaxed_search:
            key = self.hash_logical(world_state.relaxed_state)
        else:
            key = self.hash_logical(world_state.logical_state)
        if key not in self.nodes or depth < self.nodes[key].depth:
            node = PlanningNode(key, world_state, parent, opname, op, op_args, depth)
            # Compute goal satisfaction
            node.apply_goal(self.domain, goal)
            # Compute heuristic based on goal satisfaction
            hval = self.heuristic(node)
            self.nodes[key] = node
            heapq.heappush(self.unvisited, (hval, node))
            return True
        return False
        
    def backup(self, node):
        plan = []
        while node.parent is not None:
            plan.append(node.opname)
            node = node.parent
        plan.reverse()
        return plan

    def verify(self, root, goal_conditions, plan):
        self.domain.update_logical(root)
        self.reset(root)
        world_state = root
        for i, op in enumerate(plan):

            print("Checking goal...")
            if self.domain.check(world_state, goal_conditions, verbose=True):
                print("done early")
                return True

            print("===>", i, op)
            compiled_op = self.domain.compiled_op[op]
            (op, preconditions, run_conditions, effects, mutables,
             op_args) = compiled_op
            if self.domain.check(world_state, preconditions, verbose=True):
                world_state = op.apply(world_state, *op_args)
            else:
                print("SKIPPING")
                continue

        print("Checking goal...")
        if self.domain.check(world_state, goal_conditions):
            print("DONE AT END!")
            return True

        print("PLAN DOES NOT WORK")
        return False

    def solve(self, root, goal_conditions, reset=True):
        """
        Very simple search for now.
        Iterate over the available conditions. Create nodes corresponding to
        the different logical states.
        """
        
        # If this is a new search and not one we're continuing
        if reset is True:
            # Compute all predicates for the current world state
            root = root.fork()
            # TODO: should we be doing this?
            # self.domain.update_logical(root)
            root.update_relaxed() # Compute relaxation of logical state
            self.reset(root) # Reset search tree
            self.unvisited = []

            # Add world state to unvisited set
            self.push_world_state(root, None, "", None, [], goal_conditions)
        
        # Get set of actions (edges) we can explore
        compiled_ops = self.get_compiled_planning_ops()

        # main loop
        step = 0
        best_depth = float('inf')
        best_node = None
        while len(self.unvisited) > 0:
            step += 1

            # Get the next value with lowest heuristic
            (hval, node) = heapq.heappop(self.unvisited)
            world_state = node.world_state
            # print(hval, node.opname)
            
            # TODO: remove this. for debugging plan lengths.
            # print(len(self.unvisited), step, node.depth)

            # Ignore any nodes that are farther than the goal that we have found
            if node.depth >= best_depth:
                continue

            if self.domain.check(world_state, goal_conditions):
                if self.verbose:
                    print(">>>> Found goal at", node.depth)
                    if not self.greedy:
                        print(">>>> Will now attempt to improve path to goal.")
                if node.depth < best_depth:
                    best_node = node
                    best_depth = node.depth
                if self.greedy:
                    break
                continue

            # Iterate over all compiled operations
            # for opname, compiled_op in self.domain.compiled_op.items():
            # Track the numebr of things we're adding as we do this
            added = 0
            for (opname, (op, preconditions, run_conditions, effects, mutables, op_args)) in compiled_ops.items():
                if self.domain.check(world_state, preconditions,
                                     verbose=self.verbose > 2,
                                     relaxed=self.relaxed_search):

                    # apply each operator to this world state
                    pred_ws = op.apply(world_state, *op_args, relaxed=self.use_relaxed_state)

                    if pred_ws:
                        if self.push_world_state(pred_ws, node, opname, op, op_args, goal_conditions, step):
                            added += 1
                self.verbose = 0

            if step > self.max_steps:
                info.logerr("Aborting planning; reached max iterations of " + str(self.max_steps))
                break

        if best_node is not None:
            return self.backup(best_node)
        else:
            info.logerr("PLANNER: No solution found!")
            return None

    def get_policy(self, root, goal, *args, **kwargs):
        plan = self.solve(root, goal, *args, **kwargs)
        if plan is not None:
            info.inform("Found plan: " + str(plan))
            return PlanExecutionPolicy(self.domain, plan, goal)
        else:
            info.logerr("Planning failed to goal: " + str(goal))
            return None
