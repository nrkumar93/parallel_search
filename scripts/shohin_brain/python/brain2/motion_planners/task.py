# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import timeit

import brain2.utils.status as status
from brain2.task.action import Policy
from brain2.task.execution import PlanExecutionPolicy

from brain2.task.search import TaskPlanner, PlanningNode
from brain2.task.action import ActionParams
import brain2.task.search as search


# Pausing in python3
try: raw_input
except NameError: raw_input = input

class ContinuousPlanExecutionPolicy(PlanExecutionPolicy):
    """
    Wrapper for continuous task-and-motion plan execution.
    """
    def __init__(self, domain, plan, goal, update_logical_state=True):
        """
        Create plan execution policy. Will execute a specific sequence of
        operators until goal is met.
        """
        self.domain = domain
        self.plan = plan
        self.goal = goal
        self.reset()
        self.update_logical_state = update_logical_state

    def reset(self):
        """ Unlike the symbolic version, this policy really is always the same.
        We have particular actionds and motions that we've chosen to be the
        best here. If we need new ones we'll have to replan.
        """
        super(ContinuousPlanExecutionPolicy, self).reset()
        self.current_nodes = self.plan[0]
        self.current_subgoals = self.plan[1]
        self.current_motions = self.plan[2]

        # compute goals based on templates
        self.current_goal = self.goal
        self.current_plan = [node.opname for node in self.current_nodes]

        # Compute a plan
        if self.plan is None:
            raise RuntimeError('planning not yet supported')

        self.implicit_conditions = self.domain.get_implicit_conditions(
            self.current_plan, self.current_goal)

    def enter(self, world_state=None):
        if self.op is not None or self.op_args is not None:
            raise RuntimeError('did not properly exit plan')
        self.reset()
        return True

    def execute_open_loop(self, world_state, pause=True):
        """ Execute the policies by moving to the correct sequence of goal
        positions using the chosen motions. Do nothing else. """
        for i, (node, subgoal, motion) in enumerate(zip(self.current_nodes,
            self.current_subgoals, self.current_motions)):
            print("========", i, node.opname, "========")
            print("Motion:")
            print(motion)
            print("Subgoal:")
            print(subgoal)
            ctrl = world_state[subgoal.actor].ctrl
            if motion.trajectory is not None and len(motion.trajectory) > 0:
                qs, ts = motion.trajectory
                ctrl.execute_joint_trajectory(qs, ts)
            ctrl.go_local(T=subgoal.ee_pose, q=subgoal.q)
            if pause:
                raw_input("============================")
            if subgoal.attach == status.ATTACH:
                ctrl.close_gripper(wait=True)
            elif subgoal.attach == status.DETACH:
                ctrl.open_gripper(wait=True)
        print("DONE EXECUTION.")

    def step(self, world_state, evaluate=False):
        """
        Apply policy or generate plan. Three modes:
        - plan and find a goal
        - follow a given plan
        - just execute a policy to test it

        This loop runs continuously at 10hz, 30hz, etc.
        It computes the current continuous world state:
            world_state = self.observer.observe()
        Then it updates the associated logical state:
            domain.update_logical(world_state)

        If need be, we can also use our samplers again, or trigger replanning.
        """

        if self.current_plan is None:
            raise RuntimeError('Plan is missing!')
        if self.current_goal is None:
            raise RuntimeError('Goal is missing!')

        # Check goal status: run through all elements in plan, find the last
        # element in the plan whose conditions are met.
        # We're assuming that the plan has been properly parameterized by
        # the enter() function.
        res, self.op, self.op_args = self.domain.step(
            self.current_plan, self.current_goal, world_state,
            self.op, self.op_args, self.implicit_conditions,
            update_logical=self.update_logical_state,
            evaluate=evaluate)

        # Check return status
        if res == status.FINISHED:
            self.op = None
            self.op_args = None
            info.logwarn("Plan execution complete.")
            return status.FINISHED
        elif self.op is None or res < 0:
            info.logwarn("Could not continue execution: " + str(res))
            return status.FAILED
        else:
            return status.RUNNING


class TaskAndMotionPlanner(TaskPlanner):
    """
    This contains nodes associated with a particular task planning problem. When
    called, it will reset itself and perform a greedy A* search out to max
    depth.
    """

    def __init__(self, domain, heuristic=None, use_relaxed_state=True, max_steps=250000, verbose=0, tamp_config={},
                 pause_after_failures=False, preplan_steps=10000):
        """
        Set up the planner. Takes in a heuristic function.
        """
        #if heuristic is None:
        #    heuristic = search.depth_heuristic
        super(TaskAndMotionPlanner, self).__init__(domain, heuristic, use_relaxed_state, max_steps, verbose)
        self.max_goal_iter = self._read_config(tamp_config, "max_goal_iter", 10)

        # Relaxed planner
        self.relaxed_planner = TaskPlanner(domain,
                                           #heuristic=search.relaxed_only_heuristic,
                                           heuristic=search.relaxed_heuristic,
                                           max_steps=preplan_steps,
                                           relaxed_search=True,
                                           use_relaxed_state=True,
                                           verbose=False,
                                           greedy=True)

        # Use our reset function
        self.reset(None)

        # How many iterations to refine? This is how much we exploit a GOOD task plan before
        # searching for others.
        self._refine_iter = 1
        self._num_subgoal_attempts = 25
        self._h_wt = -5

        # For debugging
        self.pause_after_failures = pause_after_failures 

        # Future features
        self.always_expand_subgoals = False

    def _read_config(self, config, key, default_value):
        """ safe input """
        if key in config:
            self.__dict__[key] = config[key]
        else:
            self.__dict__[key] = default_value

    def reset(self, root):
        self.root = root
        self.nodes = {}

    def push_world_state(self, world_state, parent, opname, op, op_args, goal,
                         step=0, modifier=0):
        """
        Create node for this world state and add it to the priority heap. If
        this node already exists then forget about it.
        """
        depth = 0
        if parent is not None:
            # Some things don't make this much deeper - for example grasping
            # Because we already spent an action lining up
            depth = parent.depth + op.cost

        # compute hash for the logical state
        # we use logical state to lookup the actual (possibly continuous)
        # world state expectation
        key = self.hash_logical(world_state.logical_state)
        new_node = key not in self.nodes
        if new_node or depth < self.nodes[key].depth:
            node = PlanningNode(key, world_state, parent, opname, op, op_args, depth)
            # Compute goal satisfaction
            node.apply_goal(self.domain, goal)

            # Should we be computing subgoals here?
            if op is not None and self.always_expand_subgoals:
                print(">>>", opname, op, op_args)
                params = node.op.refine_subgoal(node.parent.world_state, node.world_state, *node.op_args,
                                                batch_size=self._num_subgoal_attempts)
                if params.success:
                    self.domain.apply_action(node.parent.world_state, node.world_state, params)
                    self.domain.update_logical(node.world_state)
                    node.params = params

            # Add created node to the parent
            if parent is not None:
                parent.children.append(node)

            # Compute heuristic based on goal satisfaction
            hval = self.heuristic(node) + modifier
            node.heuristic = hval
            self.nodes[key] = node
            if new_node:
                heapq.heappush(self.unvisited, (hval, node))
            return node
        return None

    def backup(self, node):
        plan = []
        while node.parent is not None:
            plan.append(node)
            node = node.parent
        plan.reverse()
        return plan

    def refine_subgoals(self, plan):
        """ Predict the results of each high level action to determine if it will be feasible and
        might actually result in us accomplishing our goals. """

        subgoal_plan = []
        if self.verbose > 0:
            print([node.opname for node in plan])

        if len(plan) > 0:
            self.domain.reset_world_state(plan[0].parent.world_state)

        # Start with each node in the plan
        for node in plan:
            if self.verbose > 1:
                print("\n==============================")
            if self.verbose > 0:
                info.inform("Finding subgoal for", node.opname)

            # Pull out the necessary sampler based on its original world state
            params = node.op.refine_subgoal(node.parent.world_state, node.world_state, *node.op_args,
                                            batch_size=self._num_subgoal_attempts)
            if self.verbose > 1:
                print(params)
            if not params.success:
                if self.verbose > 0:
                    print("SUBGOAL REFINEMENT FAILED AT", node, node.opname)
                if self.pause_after_failures: raw_input()
                return False, subgoal_plan
            else:
                # Advance the world state. This is done via an in-place update
                self.domain.apply_action(node.parent.world_state, node.world_state, params)
                if self.verbose > 1:
                    print("After refinement: attach =", node.world_state[self.domain.robot].attached_to)
                self.domain.update_logical(node.world_state)
                # And add to our plan
                # TODO: we could also check predicates and all here instead of just applying
                subgoal_plan.append(params)
        else:
            if self.verbose > 0:
                print("DONE FINDING SUBGOALS:", subgoal_plan)
            # At the end, if we were able to instantiate goals for the whole plan, return it and
            # we'll try to create motions connecting these positions
            return True, subgoal_plan

    def refine_connections(self, plan, subgoal_plan):
        """ Find trajectories connecting each of the different operators so that we can get some
        decent motion plans going here. """
        motion_plan = []

        # Clean up the world so that we can find a decent plan
        if len(plan) > 0:
            self.domain.reset_world_state(plan[0].parent.world_state)

        for node, params in zip(plan, subgoal_plan):
            params = node.op.refine_connection(node.parent.world_state, node.world_state,
                                               params, *node.op_args)
            if not params.success:
                print("FAILED MOTION PLANNING")
                return False, motion_plan
            else:
                motion_plan.append(params)
        else:
            print("DONE CONNECTING SUBGOALS:", motion_plan)
            return True, motion_plan

    def _descend(self, goal_conditions, node, compiled_ops, step, t0, max_time, relaxed_ops=set()):
        """
        Add new nodes to current tree.
        """
        
        # First, check timer to make sure we aren't over-budget
        t = timeit.default_timer() - t0
        if t > max_time:
            info.logwarn('Node expansion took too long: time = %f seconds' % t)
            return None
        
        # If we have arrived at the goal we are done here
        if self.domain.check(node.world_state, goal_conditions, verbose=True):
            print("Descent: reached a valid goal.")
            return node

        # self.verbose = 3
        # Otherwise, try to add new actions - or go down a new path
        if not node.expanded:
            # Iterate over all compiled operations
            # for opname, compiled_op in self.domain.compiled_op.items():
            # Track the numebr of things we're adding as we do this
            node.expanded = True
            added = 0
            best_heuristic = float('inf')
            best_node = None
            for (opname, (op, preconditions, run_conditions, effects, mutables, op_args)) in compiled_ops.items():
                # Get the parameters of the particular op and try things
                if self.verbose > 2:  print("APPLYING", opname, op_args)

                # Check conditions here and apply to the world state 
                if self.domain.check(node.world_state, preconditions, verbose=self.verbose > 2):
                    pred_ws = op.apply(node.world_state, *op_args, relaxed=True)
                    relaxed_h = self._h_wt if opname in relaxed_ops else 0

                    # If operation was successful, add it to the tree;
                    # otherwise, just move on 
                    if pred_ws is None:
                        if self.verbose > 1:
                            print("WARNING: op not added:", opname)
                        continue

                    if self.verbose > 0:
                        print("--", opname, "with args =", op_args)
                    new_node = self.push_world_state(pred_ws, node, opname, op, op_args, goal_conditions, step,
                                             modifier=relaxed_h)
                    if new_node is None: continue

                    print("Added", new_node.opname, new_node.p_correct,
                            new_node.p_relaxed, new_node.heuristic)
                    added += 1
                    if new_node.heuristic < best_heuristic:
                        best_node = new_node
                        best_heuristic = new_node.heuristic
                else:
                    if self.verbose > 2:
                        print(opname, "--> NOT POSSIBLE")
            if best_node is not None:
                print("Descend: what can we do after adding?", best_node.opname)
                return self._descend(goal_conditions, best_node, compiled_ops,
                                     step, t0, max_time,
                                     relaxed_ops=relaxed_ops)
            else: return None
        else:
            raise RuntimeError()

    def solve2(self, root, goal_conditions, reset=True, max_time=60., greedy=False):
        # If this is a new search and not one we're continuing
        if reset is True:
            # Compute all predicates for the current world state
            self.domain.update_logical(root)
            root.update_relaxed() # Compute relaxation of logical state

        # Reset internal structures
        self.reset(root) # Reset search tree
        self.unvisited = []

        # Get world state and operators
        if self.verbose > 0:  print("Planning to goal:", goal_conditions)
        root_node = self.push_world_state(root, None, "", None, [], goal_conditions)
        compiled_ops = self.get_compiled_planning_ops()
        t0 = timeit.default_timer()

        # Get a relaxed version of the symbolic plan - so we know which actions to use
        relaxed_plan = self.relaxed_planner.solve(root, goal_conditions)
        relaxed_ops = set()
        if relaxed_plan is not None:
            for op in relaxed_plan:
                relaxed_ops.add(op)
            #if self.verbose > 0:
            print("-> Preplanning found these useful ops:", relaxed_plan)
        else:
            print("-> Preplanning failed. This is gonna be hard.")

        # main loop
        step = 0
        possible_solutions = []
        best_solution = None
        best_depth = float('inf')
        best_node = None
        
        # Loop over all elements and try to find a plan that might work
        while step < self.max_steps:
            # Check the time
            t = timeit.default_timer() - t0
            if t > max_time:
                info.logwarn('Took too long: time = %f seconds' % t)
                break

            # Main logic here: find a node and expand it
            # We iteratively go down the tree, adding each one to the tree and
            # choosing the best child.
            # Needs to stop when we add a new node to the search tree
            node = self._descend(goal_conditions, root_node, compiled_ops,
                                 step, t0, max_time,
                                 relaxed_ops=relaxed_ops)
            print(node)

            # If we found a potential goal, then return everything
            if node is not None:
                # Now that we have a plan, try to instantiate it, starting at the root I guess.
                # First we try to check if the "goals" of each motion look reasonable
                plan = self.backup(node)
                print(">>>>> Try:", [p.opname for p in plan])
                for it in range(self._refine_iter):
                    res, subgoals = self.refine_subgoals(plan)
                    if not res:
                        opname = plan[len(subgoals)].opname
                        info.logwarn("Attempt %d: failed to find subgoal plan of len %d on %s" % (it+1, len(plan), opname))
                        continue
                    res, connections = self.refine_connections(plan, subgoals)
                    if res: break
                    else:
                        opname = plan[len(connections)].opname
                        info.logwarn("Attempt %d: failed to find motion plan of len %d on %s" % (it+1, len(plan), opname))
                    
                # If this succeeded... flesh out plans a bit more
                if not res:
                    info.logerr("Failed to find connected subgoal plan!")
                    # return None
                elif res:
                    # Return everything here
                    best_solution = plan, subgoals, connections
                    best_node = node
                    best_depth = node.depth
                    if greedy:
                        break

            # Check goal conditions for our new node 

            # Increment and continue
            step += 1
            if step > self.max_steps:
                info.logerr("Aborting planning; reached max iterations of " + str(self.max_steps))
                break



    def solve(self, root, goal_conditions, reset=True, max_time=60., greedy=False):
        """
        Very simple search for now.
        Iterate over the available conditions. Create nodes corresponding to
        the different logical states.
        """

        # If this is a new search and not one we're continuing
        if reset is True:
            # Compute all predicates for the current world state
            self.domain.update_logical(root)
            root.update_relaxed() # Compute relaxation of logical state
            # Reset internal structures
            self.reset(root) # Reset search tree
            self.unvisited = []

            # Add world state to unvisited set
            self.push_world_state(root, None, "", None, [], goal_conditions)

        if self.verbose > 0:
            print("Planning to goal:", goal_conditions)

        compiled_ops = self.get_compiled_planning_ops()

        # start timing
        t0 = timeit.default_timer()

        # Get a relaxed version of the symbolic plan - so we know which actions to use
        relaxed_plan = self.relaxed_planner.solve(root, goal_conditions)
        relaxed_ops = set()
        if relaxed_plan is not None:
            for op in relaxed_plan:
                relaxed_ops.add(op)
            #if self.verbose > 0:
            print("-> Preplanning found these useful ops:", relaxed_plan)
        else:
            print("-> Preplanning failed. This is gonna be hard.")

        # main loop
        step = 0
        possible_solutions = []
        best_solution = None
        best_depth = float('inf')
        best_node = None
        while len(self.unvisited) > 0:
            step += 1
            t = timeit.default_timer() - t0
            if t > max_time:
                info.logwarn('Took too long: time = %f seconds' % t)
                break
            if step > self.max_steps:
                info.logerr("Aborting planning; reached max iterations of " + str(self.max_steps))
                break

            # Get the next value with lowest heuristic
            (hval, node) = heapq.heappop(self.unvisited)
            # print(hval, node.opname)
            world_state = node.world_state

            # Ignore any nodes that are farther than the goal that we have found
            if node.depth >= best_depth:
                continue

            # Check prereqs. In the future we should make this work a bit better - expand the tree a
            # bit - and also make some of those tree expansions a bit less expensive.
            if self.domain.check(world_state, goal_conditions):
                plan = self.backup(node)

                # Now that we have a plan, try to instantiate it, starting at the root I guess.
                # First we try to check if the "goals" of each motion look reasonable
                print(">>>>> Try:", [p.opname for p in plan])
                for it in range(self._refine_iter):
                    res, subgoals = self.refine_subgoals(plan)
                    if not res:
                        opname = plan[len(subgoals)].opname
                        info.logwarn("Attempt %d: failed to find subgoal plan of len %d on %s" % (it+1, len(plan), opname))
                        continue
                    res, connections = self.refine_connections(plan, subgoals)
                    if res: break
                    else:
                        opname = plan[len(connections)].opname
                        info.logwarn("Attempt %d: failed to find motion plan of len %d on %s" % (it+1, len(plan), opname))
                    
                # If this succeeded... flesh out plans a bit more
                if not res:
                    info.logerr("Failed to find connected subgoal plan!")
                    # return None
                elif res:
                    # Return everything here
                    best_solution = plan, subgoals, connections
                    best_node = node
                    best_depth = node.depth
                    if greedy:
                        break

            # Iterate over all compiled operations
            # for opname, compiled_op in self.domain.compiled_op.items():
            # Track the numebr of things we're adding as we do this
            added = 0
            for (opname, (op, preconditions, run_conditions, effects, mutables, op_args)) in compiled_ops.items():
                # Get the parameters of the particular op and try things
                
                if self.verbose > 2:
                    print("APPLYING", opname, op_args)  # , preconditions)

                # Check conditions here and apply to this world state 
                if self.domain.check(world_state, preconditions, verbose=self.verbose > 2):
                    pred_ws = op.apply(world_state, *op_args, relaxed=True)
                    relaxed_h = self._h_wt if opname in relaxed_ops else 0

                    # If operation was successful, add it to the tree
                    if pred_ws:
                        if self.verbose > 0:
                            # "-->", pred_ws.logical_state)
                            print("--", opname, "with args =", op_args)
                        new_node = self.push_world_state(pred_ws, node, opname, op, op_args, goal_conditions, step,
                                                 modifier=relaxed_h)
                        if new_node is not None:
                            added += 1
                        else:
                            if self.verbose > 1:
                                print("WARNING: op not added:", opname)
                else:
                    if self.verbose > 2:
                        print(opname, "--> NOT POSSIBLE")

                self.verbose = 0

            if step > self.max_steps:
                info.logerr("Aborting planning; reached max iterations of " + str(self.max_steps))
                break

        if best_solution is None:
            info.logerr("PLANNER: No solution found!")
            return None
        else:
            return best_solution

    def get_policy(self, root, goal, *args, **kwargs):
        """
        Return a continuous task and motion plan execution policy.
        """
        plan = self.solve(root, goal, *args, **kwargs)
        nodes, goals, motions = plan
        if plan is not None:
            info.inform("Found plan: " + str([p.opname for p in nodes]))
            return ContinuousPlanExecutionPolicy(self.domain, plan, goal)
        else:
            info.logerr("Planning failed to goal: " + str(goal))
            return None
