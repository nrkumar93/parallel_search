# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

import brain2.utils.info as info
import brain2.utils.status as status

import threading


class Policy(object):
    """
    Policy class
    """

    def __init__(self):
        # self.domain = domain
        self.reset()

    def enter(self, world_state, actor, plandata=None):
        """
        Called when we start an action
        """
        return True

    def exit(self, world_state=None, actor=None, plandata=None):
        """
        Called at the end of an action
        """
        return True

    def reset(self):
        """This should reset any internal state that the policy holds."""
        pass

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, world_state, actor, *args, **kwargs):
        """
        this should return a status code. This should return immediately, and
        should be owned by a basic Operator.
        """
        raise NotImplementedError('should override this')

    def is_subplan(self):
        return False


class SimplePolicy(Policy):
    """
    This policy wraps a simple function.
    """

    def __init__(self, function):
        super(SimplePolicy, self).__init__()
        self.function = function
        self.step = function

    def __call__(self, *args, **kwargs):
        self.function(*args, **kwargs)


class Goal(object):
    """
    Represents the continuous end-state of a particular operator. It's used to
    sample different configurations of the world during task and motion
    planning, which are then added into a plan structure.
    """
    def __init__(self, domain):
        self.domain = domain

    def __call__(self, world_state, actor, goal):
        """
        Call should produce a list of acceptable goals.
        """
        raise NotImplementedError('Must override this')


class ActionParams(object):
    """
    Store info on where we are going next with the plan.
    """

    def __init__(self, actor, goal=None, attach=status.NO_CHANGE, pose=None, q=None, attach_to=None, relative_pose=None,
            depth=0, cost=0., ee_pose=None, success=True, linear_dist=None):
        self.actor = actor
        self.goal = goal
        self.attach = attach                # Are we connecting to something or what?
        self.attach_to = attach_to          # What are we connecting to?
        #self.action = action               # Name of the action/operator being executed
        self.pose = pose                    # base position goal
        self.ee_pose = ee_pose
        self.q = q                          # config space goal
        self.relative_pose = relative_pose  # relative offset (e.g grasp pose)
        self.depth = depth                  # depth (hierarchical trace)
        self.sub_actions = []
        self.cost = cost                    # Is there a cost here?
        self.success = success              # Did we find a valid subgoal or connection?
        self.linear_dist = linear_dist      # Used for opening drawers and whatnot


    def __str__(self):
        msg = "------------------- SUBGOAL ------------------\n"
        msg += "Actor = " + str(self.actor) + ", Goal = " + str(self.goal) + "\n"
        if not self.success:
            msg += ">>>> ACTION FAILED! <<<<<\n"
        if self.attach != status.NO_CHANGE:
            msg += "Attach state = " + str(self.attach) + " to = " + str(self.attach_to) + "\n"
        if self.pose is not None:
            msg += "move base to pose =" + str(self.pose[:3, 3]) + "\n"
        if self.ee_pose is not None:
            msg += "move ee to pose = " + str(self.ee_pose[:3, 3]) + "\n"
        if self.relative_pose is not None:
            msg += "move ee to relative pose = " + str(self.relative_pose[:3, 3]) + "\n"
        if self.q is not None:
            msg += "Arm configuration =" + str(self.q) + "\n"
        return msg


class ConnectionParams(object):
    """
    Store information on motions connecting various subgoals
    """
    def __init__(self, actor, goal, success, trajectory=[], tree=None):
        self.actor = actor              # The thing that's moving. Almost always the robot.
        self.goal = goal                # Goal of the motion, e.g. an object to grasp or a surface
        self.trajectory = trajectory    # Contains trajectory data for execution
        self.tree = tree                # Contains motion planning data for later refinement
        self.success = success          # Were we able to make this connection or did it just fail

    def __str__(self):
        msg = "------------------- CONNECTION ------------------\n"
        msg += "Actor = " + str(self.actor) + ", Goal = " + str(self.goal) + "\n"
        if not self.success:
            msg += ">>>> ACTION FAILED! <<<<<\n"
        if self.trajectory is not None and len(self.trajectory) > 0:
            msg += "Trajectory = " + str(self.trajectory) + "\n"
        else:
            msg += "<No trajectory.>\n"
        return msg


class Operator(object):

    """
    Combination of a list of preconditions, runnable (continuous) policy, and
    optional symbolic-state transition model.

    Operators own a policy, which should execute on the real robot.

    If using the basic operator, we expect a Policy object.
    If using the AsyncOperator (inherits from this), we expect an AsyncPolicy.
    """

    def __init__(self, domain, name, preconditions, run_conditions, effects, mutable,
                 policy, actor="robot", to_entities=None, preemptable=False,
                 subgoal_sampler=None,
                 subgoal_connector=None, cost=1):

        if actor is None:
            raise RuntimeError('must specify actor!')

        self.name = name
        self.domain = domain
        self.preconditions = preconditions
        self.run_conditions = run_conditions
        self.effects = effects
        self.mutable = mutable
        self.policy = policy
        self.actor = actor
        self.to_entities = to_entities
        self.preemptable = preemptable
        self.compiled_ops = []
        self.cost = cost

        # Samplers for goals and so on
        self.subgoal_sampler = subgoal_sampler
        self.subgoal_connector = subgoal_connector

        # Find out if this contains a subplan -- if it has the is_subplan()
        # function then it contains multiple actions.
        try:
            self.is_subplan = policy.is_subplan()
        except NotImplementedError:
            self.is_subplan = False

    def refine_subgoal(self, prev_world_state, world_state, goal=None, *args, **kwargs):
        """
        Update state in world_state to better reflect what will happen after the plan is executed
        optimally.
        :param prev_world_state: this is the parent world state before the action was executed
        :param world_state: this is the effect of the action,
        :param actor: what's acting
        :param goal: what it's acting relative to
        """
        if self.subgoal_sampler is None:
            return ActionParams(success=True, actor=self.actor, goal=goal)
        else:
            params = self.subgoal_sampler(prev_world_state, self.actor, goal, *args, **kwargs)

            # In-place update of the world state
            #self.domain.apply_action(prev_world_state, world_state, params)
            return params

    def refine_connection(self, prev_world_state, world_state, subgoal, goal=None, *args, **kwargs):
        """
        Update state in world_state to better reflect what will happen after the plan is executed
        optimally. Also computes the "connection" between the two -- trajectory, for example -- and
        ensures that the trajectory can be executed.

        :param prev_world_state: this is the parent world state before the action was executed
        :param world_state: this is the effect of the action,
        :param subgoal: this is what we computed when last refining the subgoal. It CAN be used to
                        specify a specific position we need to reach or ignored.
        :param actor: what's acting
        :param goal: what it's acting relative to
        """
        actor = subgoal.actor
        goal = subgoal.goal
        if self.subgoal_connector is None:
            return ConnectionParams(actor=actor, goal=goal, success=True)
        else:
            params = self.subgoal_connector(prev_world_state, self.actor, goal, subgoal, *args, **kwargs)

            # In-place update of the world state
            # TODO: do we actually need this or anything?
            # self.domain.apply_action(prev_world_state, world_state, params)
            return params

    def add_compiled_op(self, opname):
        """ Track compiled (instantiated) operators -- the ones that actually make up the edges in
        our task graph. """
        self.compiled_ops.append(opname)

    def enter(self, world_state, goal=None):
        """
        Called when we first start using this operator
        """
        return self.policy.enter(world_state, self.actor, goal)

    def exit(self, world_state, goal=None):
        """
        Called when we stop using this operator
        """
        return self.policy.exit(world_state, self.actor, goal)

    def step(self, world_state, goal=None, evaluate=False):
        """
        This computes an action to apply for this actor in this particular
        world. It takes the current world state and the entity we are acting
        on, since actor is fixed for each operation.
        """

        # World state is provided here -- it contains some extra information,
        # not just the current actors, which might be useful (e.g. for
        # collision avoidance).
        if evaluate:
            return self.policy.step(world_state, self.actor, goal, evaluate=evaluate)
        else:
            return self.policy.step(world_state, self.actor, goal)

    def apply(self, world_state, goal=None, relaxed=False):
        """
        This function should compute the logical world state resulting from
        this high-level action. This should let us sequence actions for task
        planning.
        """

        # This creates a new world state and adds it
        next_world_state = world_state.fork()

        # Compute logical effects and apply them
        for effect, value in self.effects:
            templated_effect = self.domain.format(effect, self.actor, goal)
            # TODO: better lookup here instead of querying via the domain. Or
            # maybe world_state objects need an alias to the domain as well.
            predicate_idx = self.domain.predicate_idx[templated_effect]
            next_world_state.logical_state[predicate_idx] = value

            if relaxed:
                offset = 0 if value else next_world_state.logical_state.shape[0]
                # print(effect, value, offset)
                next_world_state.relaxed_state[predicate_idx + offset] = True

        return next_world_state
