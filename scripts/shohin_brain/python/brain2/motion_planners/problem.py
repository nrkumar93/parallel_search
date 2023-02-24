# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from .config import default_config

class MotionPlanningProblem(object):
    """
    Stores all the necessary information about a sampling-based motion planning problem.

    The MotionPlanningProblem is not specific to a particular start configuration.

    Implementation note:
    This is not an abstract base class, mostly because I want to make it easy
    to provide function access to different collision checkers etc. without
    writing a new child class.
    """

    def __init__(self,
                 sample_fn, # sample intermediate waypoints
                 goal_fn, # where we want to go
                 extend_fn, # connect different waypoints
                 is_valid_fn, # check if point is valid
                 is_done_fn, # check if we are done
                 config=None, # various parameters for problem
                 distance_fn=None,
                 show_fn=None  # used for debugging
                 ):
        """
        Provide function pointers to the appropriate problem (simulation
        environment used for collision checking).
        """

        # Sample function should return a configuration q to explore next
        self.sample = sample_fn

        # Determine if we are done exploring or not.
        self.is_done = is_done_fn

        # Function to sample a goal. If there's only a configuration, then we
        # want to just return that.
        self.reset_sample_goal_fn(goal_fn)

        # Extend function connects two different configurations q1 and q2
        self.extend = extend_fn

        # Collision check -- determines if configuration q is valid to move to.
        self.is_valid = is_valid_fn

        # Show function - show an individual configuration for debugging
        self.show_config = show_fn

        if distance_fn is None:
            self.distance = self.default_distance
        else:
            self.distance = distance_fn

        if config is None:
            config = default_config
        # Set up configuration variables
        # Robot degrees of freedom
        self.dof = config['dof']
        # Probability of sampling from goal region
        self.p_sample_goal = config['p_sample_goal']
        # Number of iterations to run
        self.iterations = config['iterations']
        self.verbose = config['verbose']
        # number of iterations to check for valid goal positions before starting
        self.goal_iterations = config['goal_iterations']

        # Minimum number of cycles to run when randomly sampling
        self._read_config(config, 'min_iterations', 0)
        self._read_config(config, 'shortcut', False)
        self._read_config(config, 'shortcut_iterations', 50)
        self._read_config(config, 'neighborhood_radius', 1)
        self._read_config(config, 'tolerance', 1e-3)
        self._read_config(config, 'check_start_valid', True)
        self._read_config(config, 'check_goal_valid', True)
    
    def _read_config(self, config, key, default_value):
        """ safe input """
        if key in config:
            self.__dict__[key] = config[key]
        else:
            self.__dict__[key] = default_value

    def reset_sample_goal_fn(self, goal_fn):
        """
        The goal sampler function is most likely to change, since it'll depend
        on where you actually want the arm to go next.
        """
        if not callable(goal_fn):
            # In this case, we're assuming the goal is a single known
            # configuration that we have to get to.
            self.sample_goal = lambda: goal_fn

            if self.is_done is not None:
                raise RuntimeError('you should not provide an is_done() '
                                   'function with fixed configuration-space '
                                   'goal.')

            # Done is easy in this case: just check distance to goal
            self.is_done = lambda q: self.distance(q) < 1e-6
        else:
            self.sample_goal = goal_fn

    def sample_next_q(self, i=0):
        r = np.random.random()
        if r < self.p_sample_goal or i == 0:
            return self.sample_goal()
        else:
            return self.sample()

    def sample_next_valid_q(self, i=0):
        r = np.random.random()
        if r < self.p_sample_goal or i == 0:
            return self.sample_goal()
        else:
            return self.sample()

    def default_distance(self, q1, q2):
        #if not len(q2) == len(q1) or not len(q2) == self.dof:
        #    raise RuntimeError('configuration sizes do not match robot dof: '
        #                       + str(self.dof))
        return np.linalg.norm(q2 - q1)
