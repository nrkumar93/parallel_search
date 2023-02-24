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
import types

from termcolor import colored

from .condition import Condition
from .action import Operator, SimplePolicy
from .world_state import WorldState

import brain2.utils.status as status
import brain2.utils.info as info


class DomainDefinition(object):

    """
    Holds conditions to define a world state.

    """

    success = 1
    running = 0
    failed = -1

    def __init__(self, verbose=0, *args, **kwargs):

        # Basic world state
        self.root = WorldState(self)
        self.verbose = verbose

        # Ways we can measure the world
        self.sensors = {}

        # Store entity indices
        self.entities = set()
        # Store all conditions here -- these are the fields in our logical
        # state.
        self.conditions = {}
        # Store available operators here -- these are the high-level symbolic
        # actions that we can perform.
        self.operators = {}
        # Holds compiled list of qualities and predicates we want to compute
        self.predicate_idx = {}
        self.compiled_op = {}

        # Record size of the hidden state
        self.compiled = False
        self.num_predicates = None
        self.num_entities = None
        self.linear = False
        self.last_op_data = None, None

        # Default operators to use for task planning
        self.planning_operators = []

    def get_predicate_indices(self):
        """
        Just an accessor to get the index of each predicate in logical state
        """
        return self.predicate_idx.items()

    def add_sensor(self, name, weight, SensorType, *args, **kwargs):
        """
        Add a sensor
        """
        self.sensors[name] = SensorType(weight, *args, **kwargs)

    def get_state(self, fork=False):
        """
        Get the current world state and update it.
        Should this fork the world state or just return the current observation?
        If we fork, history will be preserved but for long-running applications
        we would essentially have ourselves a memory leak.
        """
        if fork:
            return self.root.fork()
        else:
            return self.root

    def add_entity(self, name, dtype, *args, **kwargs):
        """
        Create entity and add it to our current world state.
        """
        if self.compiled:
            raise RuntimeError('world state has already been finalized')
        if name in self.entities:
            raise RuntimeError("name collision at " + str(name))
        elif not isinstance(name, str):
            raise TypeError("use strings as unique identifiers")

        entity = dtype(*args, **kwargs)
        self.entities.add(name)
        self.root.entities[name] = entity

        if self.verbose:
            print("Entities:", self.entities)

        # When returning the entity -- return the one associated with the
        # current world state. We can have multiple world states, and need to
        # be able to look up within these.
        return self.root.entities[name]

    def get_all_entities(self):
        return list(self.entities)

    def _fix_types(self, entity):
        """
        Set up entities as a list we can iterate over
        """
        if entity is None:
            return self.get_all_entities()
        elif not isinstance(entity, list):
            return [entity]
        else:
            return entity

    def add_relation(self, name, function, from_entity=None, to_entity=None):
        """
        Add a relationship to the definition of our logical world state.
        """
        if self.compiled:
            raise RuntimeError('world state has already been finalized')
        from_entity = self._fix_types(from_entity)
        to_entity = self._fix_types(to_entity)
        if name in self.conditions:
            raise RuntimeError(
                "Name collision! Cannot add duplicate condition %s" % name)
        else:
            self.conditions[name] = Condition(
                self, name, function, from_entity, to_entity)

    def add_property(self, name, function, from_entity=None):
        """
        Properties are true of only a single entity (e.g. the world)
        """
        if self.compiled:
            raise RuntimeError('world state has already been finalized')
        from_entity = self._fix_types(from_entity)
        if name in self.conditions:
            raise RuntimeError(
                "Name collision! Cannot add duplicate condition %s" % name)
        else:
            self.conditions[name] = Condition(
                self, name, function, from_entity, None)

    def format(self, term, actor, goal=""):
        '''
        Return name of an instance of this term (operator, predicate, whatever).

        TODO: support more complex queries.
        '''
        return term.format(goal)

    def format_list(self, clist, e1):
        cclist = []
        for cname in clist:
            ccname = cname.format(e1)
            cclist.append(ccname)
        return cclist

    def format_conditions(self, clist, e1):
        cclist = []
        for cname, val in clist:
            ccname = cname.format(e1)
            cclist.append((ccname, val))
        return cclist

    def get_mutable(self, opname, op, preconditions, run_conditions, effects,
                    mutable, op_args):
        """
        Get all conditions that can change within a given action and it's ok
        This is meant to be called from compile() only
        """
        if self.compiled:
            raise RuntimeError('not supposed to call get_mutable() on a domain'
                               ' that you already compiled.')

        plan_mutable = []
        try:
            plan = op.policy.plan
        except AttributeError:
            plan = None

        if plan is not None:
            # Loop over a plan, adding anything that changes from pre- to
            # effect within that plan.

            eff = []
            for opname in plan:
                print(opname, op.actor, op_args)
                c_opname = self.format(opname, op.actor, *op_args)
                vals = self.compiled_op[c_opname]
                (_, preconditions, run_conditions, effects, op_mutable,
                 c_op_args) = vals
                # print(preconditions)
                # for p in preconditions:
                #    if p not in run_conditions or p not in effects:
                #        plan_mutable.append(p)
                # print(run_conditions)
                # print(effects)
                # raw_input()
                ed = {}
                for k, v in effects:
                    ed[k] = v
                eff.append(ed)
            # print("----------------")
            # print("mutating conditions for ", opname)
            # print("----------------")
            mutating = set()
            for i, ed in enumerate(eff):
                if i + 1 == len(eff):
                    continue
                for j in range(i + 1, len(eff)):
                    for k, v in ed.items():
                        if k in eff[j] and eff[j][k] != v:
                            # this value mutates later
                            mutating.add(k)
            # print(mutating)
            # raw_input()
        return mutable + plan_mutable

    def add_operator(self, name,
                     preconditions,
                     effects,
                     policy=None,
                     operator_type=Operator,
                     run_conditions=None,
                     mutable=None,
                     actor=None, to_entities=None,
                     preemptable=True,
                     task_planning=False,
                     subgoal_sampler=None,
                     subgoal_connector=None,
                     planning_cost=1.,
                     *args, **kwargs):
        """
        Add an executable policy. Policies are all associated with:
        - preconditions: list of conditions which must hold for us to enter this state
        - run conditions: list of conditions that will hold while running.
                          Preconditions may no longer be met, but we will check
                          these instead of exiting if there are no higher
                          priority states to enter.
        - effects: functor to compute the next logical world state.
        - policy: this is called to actually execute the supposed action.
        """
        if isinstance(actor, list):
            raise RuntimeError('currently every operator only uses a single'
                               ' actor. Please specify separate operators.')
        elif actor is None:
            # No actor provided, use the default actor.
            actor = self.robot

        if self.compiled:
            raise RuntimeError('world state has already been finalized')

        if isinstance(policy, types.FunctionType):
            # This handles the case where we're just passing in a function --
            # nice and simple.
            policy = SimplePolicy(policy)

        # Register this policy
        if name in self.operators:
            raise RuntimeError(
                "Name collision! Cannot add duplicate operator %s" % name)
        else:
            if run_conditions is None:
                run_conditions = preconditions
            if mutable is None:
                mutable = []
            self.operators[name] = operator_type(
                self, name, preconditions, run_conditions, effects, mutable, policy,
                actor, to_entities, preemptable, subgoal_sampler, subgoal_connector,
                planning_cost,
                *args, **kwargs)

            if task_planning:
                self.planning_operators.append(name)

    def update_logical(self, obs):
        """
        Evaluate all predicates to get current world state
        """
        idx = 0
        for cname, c in self.conditions.items():
            idx = c.apply(obs, idx)
        return obs

    def get_implicit_conditions(self, plan, goal):
        """
        Get the set of "implicit" conditions on the different elements of a
        plan. These are things that are required by future actions, and are not
        affected by the current actions of the plan.
        """
        conditions = []
        requirements = {}
        for term, val in goal:
            requirements[term] = val

        # Accumulate the effects we've had so far -- doesn't make sense to
        # enforce properties from the initial condition since we don't have a
        # state to return to in order to enforce them.
        available = set()
        all_available = []
        print(plan)
        for i, opname in enumerate(plan):
            (op, op_preconditions, op_run_conditions, op_effects, op_mutable,
             op_args) = self.compiled_op[opname]
            for pname, pval in op_preconditions + op_effects:
                available.add(pname)
            all_available.append(copy.copy(available))
        all_available.reverse()

        # Compute the effects that we actually cared about to complete the plan
        for i, opname in enumerate(reversed(plan)):
            (op, op_preconditions, op_run_conditions, op_effects, op_mutable,
             op_args) = self.compiled_op[opname]

            # remove conditions that are allowed to change
            for pname, pval in op_effects + op_preconditions:
                # remove things that aren't in the available set
                # Check to see if this set is in requirements
                # If so, remove it
                if pname in requirements:
                    del requirements[pname]

            # remove conditions from the mutable list
            for pname in op_mutable:
                if pname in requirements:
                    del requirements[pname]

            # remove conditions that are unavailable (not specified by the plan)
            req_keys = [str(key) for key in requirements.keys()]
            for pname in req_keys:
                if pname not in all_available[i]:
                    del requirements[pname]

            print(i, requirements)

            # copy the requirements and add them to the set of implicit
            # conditions
            implicit_conditions = []
            for k, v in requirements.items():
                implicit_conditions.append((k, v))
            conditions.append(implicit_conditions)

            # Carry the requirements forward to the next one
            for term, val in op_preconditions:
                requirements[term] = val

        # Reverse and return
        conditions.reverse()
        print(conditions)
        return conditions

    def get_compiled_op(self, opname):
        """
        Get the specific conditions associated with a particular goal object.
        Returns:
        - the operator
        - instantiated preconditions
        - instatiated run conditions
        - instantiated effects
        - mutable conditions during this operator
        - tuple of arguments to pass to policies
        """
        if opname in self.compiled_op:
            (op, op_preconditions, op_run_conditions, op_effects, op_mutable,
             op_args) = self.compiled_op[opname]
            return op, op_preconditions, op_run_conditions, op_effects, op_mutable, op_args
        else:
            return None

    def step(self, plan, goal, obs, current_op=None, current_op_args=[],
             implicit_conditions=None, update_logical=True, evaluate=False, op_status=None):
        """
        Evaluate this plan according to the current domain information and an
        up to date state observation.
        """
        if update_logical:
            obs = self.update_logical(obs)
        if self.verbose:
            print("===== CHECKING SUBPLAN GOAL ======")
            goal_reached = self.check(obs, goal, verbose=True)
            print("Goal was:", goal)
        else:
            goal_reached = self.check(obs, goal)
        if goal_reached:
            if current_op is not None:
                current_op.exit(obs, *current_op_args)
            if self.verbose:
                print(colored("Finished task.", "green"))
            return status.FINISHED, current_op, current_op_args

        # Update the set of extra, implict conditions if it has not been
        # provided. This prevents premature execution of a particular state.
        if implicit_conditions is None:
            implicit_conditions = self.get_implicit_conditions(plan, goal)
        op_implicit_conditions = []

        # Move through all operators to find the final one that is currently
        # satisfied
        last_op, last_opname, last_op_args = None, None, None

        current_idx = None
        # Plan is a list of names of operators that we want to execute.
        for i, (opname, op_implicit_conditions) in enumerate(zip(plan,
                                                                 implicit_conditions)):

            # Get preconditions, run conditions, arguments for this specific
            # action.
            (op, op_preconditions, op_run_conditions, op_effects, op_mutable,
             op_args) = self.compiled_op[opname]

            # Check op conditions + implicit conditions.
            if op == current_op:
                res = self.check(obs, op_run_conditions
                                 + op_implicit_conditions,
                                 verbose=self.verbose > 0)
                current_idx = i
            else:
                res = self.check(obs, op_preconditions + op_implicit_conditions,
                                 verbose=self.verbose > 0)

            # for linear tests -- skip everything
            if current_op is not None and self.linear and current_idx is None:
                continue

            if self.verbose:
                print(">>>", colored(opname, 'green' if res else 'red'),
                      "with args", op_args, "=", res)
            if res:
                last_op = op
                last_opname = opname
                last_op_args = op_args

                # Some actions cannot be preempted until they are finished.
                # This flag should be set for such actions.
                if op == current_op and not op.preemptable:
                    break

            if self.linear and current_idx is not None and i > current_idx + 1:
                break

        if last_op is None:
            # raise RuntimeError(
            #    "This should never happen, your task is poorly defined.")
            info.logerr("No valid action possible!")
            return status.FAILED, current_op, current_op_args
        elif last_op is not current_op:

            # Case where we need to enter/exit
            if current_op is None or current_op.exit(obs, *current_op_args):
                # Make sure we properly cleanup the operator states. This is
                # particularly important depending on what shared resources are
                # involved.
                current_op = last_op
                # Allocates any shared resources for this new operator.
                if self.verbose:
                    info.logwarn("ENTERING " + str(last_opname))
                if not current_op.enter(obs, *last_op_args):
                    raise RuntimeError(
                        "could not enter execution state: " + str(last_opname))
            else:
                # NOTE: we might change this behavior if we need to drop into
                # a recovery or something. Right now I'm assuming that the low
                # level controllers will be somewhat sane and will act
                # reasonbly when killed/started from different states.
                raise RuntimeError("could not exit current execution state!")

        # Step the current operator. Operators are all associated with some
        # executable policy, which takes the usual world state + args.
        if self.verbose:
            info.logwarn("STEPPING " + str(last_opname)
                         + " with args = " + str(last_op_args))
        # This is where the policy is evaluated and commands are actually sent
        # to the robot!
        # Get executed leaf.
        self.last_op_data = (last_opname, last_op_args)

        if evaluate is False or current_op.is_subplan is True:
            op_status = current_op.step(obs, *last_op_args, evaluate=evaluate)
        else:
            op_status = None

        if op_status is None or op_status >= 0:
            plan_status = status.RUNNING
        else:
            plan_status = status.FAILED

        return plan_status, current_op, last_op_args

    def check(self, world_state, preconditions, verbose=False, relaxed=False):
        """
        Return value of predicate from the logical world state.
        """
        if not self.compiled:
            raise RuntimeError('cannot check predicates before compiling!')

        satisfied = True
        state = world_state.logical_state if not relaxed else world_state.relaxed_state
        for predicate_name, target_value in preconditions:
            idx = self.predicate_idx[predicate_name]
            if relaxed:
                offset = 0 if target_value else world_state.logical_state.shape[0]
                idx += offset
                target_value = True
            satisfied = (satisfied
                         and state[idx] == target_value)
            if verbose:
                print(" -", predicate_name, "= %s?" %
                      str(target_value), "==>", satisfied)
            if not satisfied:
                break
        return satisfied

    def compile(self):
        """
        This function needs to build the whole knowledge representation. It
        creates the hidden state representation (array filled with object
        relationships) and sets up the compiled version of ops + conditions.
        """

        self.num_entities = len(self.root.entities)
        self.num_predicates = 0
        self.predicate_idx = {}
        self.compiled_op = {}

        # Create the logical world state by instantiating all variations of the
        # predicate conditions.
        for cname, c in self.conditions.items():
            idxs_all = []
            for e1 in c.from_entities:
                if c.to_entities is None:
                    name = c.get_name(e1)
                    if name in self.predicate_idx:
                        raise RuntimeError(
                            'name collision when compiling: ' + str(name))
                    self.predicate_idx[name] = self.num_predicates
                    idxs_all.append(self.num_predicates)
                    self.num_predicates += 1
                else:
                    idxs_from = [] 
                    for e2 in c.to_entities:
                        if e1 == e2:
                            continue
                        else:
                            name = c.get_name(e1, e2)
                            if name in self.predicate_idx:
                                raise RuntimeError(
                                    'name collision when compiling: ' + str(name))
                            self.predicate_idx[name] = self.num_predicates
                            idxs_from.append(self.num_predicates)
                            idxs_all.append(self.num_predicates)
                            self.num_predicates += 1
                    else:
                        name = c.get_name(e1, "*")
            else:
                # Add generic version
                if c.to_entities is None:
                    name = c.get_name("*")
                else:
                    name = c.get_name("*", "*")

        # Create instances of all available operators (symbolic actions).
        for opname, op in self.operators.items():
            # Right now we are assuming only one actor. TODO: expand to
            # multiple actors.
            if op.to_entities is None:
                # Compiled operators consist of the op itself and list of
                # arguments. When we call it we will provide these arguments.
                templated_opname = opname + "()"
                self.compiled_op[templated_opname] = (
                    op, op.preconditions, op.run_conditions, op.effects,
                    op.mutable, ())
                self.operators[opname].add_compiled_op(templated_opname)
            else:
                for e1 in op.to_entities:
                    templated_opname = "{}({})".format(opname, e1)
                    preconditions = self.format_conditions(
                        op.preconditions, e1)
                    effects = self.format_conditions(op.effects, e1)
                    run_conditions = self.format_conditions(op.run_conditions,
                                                            e1)
                    mutable = self.format_list(op.mutable, e1)

                    # Compiled operators consist of the op itself and list of
                    # arguments. When we call it we will provide these
                    # arguments.
                    self.compiled_op[templated_opname] = (
                        op, preconditions, run_conditions, effects, mutable,
                        (e1,))
                    self.operators[opname].add_compiled_op(templated_opname)

        # Loop to fix up the conditions on this operator
        for opname, vals in self.compiled_op.items():
            (op, preconditions, run_conditions, effects, mutable, op_args) = vals
            new_mutable = self.get_mutable(opname, *vals)
            self.compiled_op[opname] = (op, preconditions, run_conditions,
                                        effects, new_mutable, op_args)

        # Compile specific operators and their constraints
        for opname, op in self.operators.items():
            # Now compile these
            # This creates hierachical planning structures
            pass

        # Create the compiled logical state and its 
        self.root.logical_state = np.zeros((self.num_predicates,))
        self.compiled = True
        return self

        if self.verbose > 1:
            print("=================================")
            print("PLANNING DOMAIN COMPILED.")
            print("Domain operators:")
            print(self.operators.keys())
            print("Domain compiled operators (after substitutions):")
            print(self.compiled_op.keys())
            print("All predicates:")
            print(self.predicate_idx.keys())
            print("All operators:")
            print(self.compiled_op.keys())
            print("=================================")

    def apply_action(self, world_state, config):
        """ This is specifically for continuous world configurations."""
        raise NotImplementedError('apply not implemented')
