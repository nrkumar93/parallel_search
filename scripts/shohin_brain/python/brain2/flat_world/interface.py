# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FlatWorld(object):
    """Simple 2d test case for planning algorithms."""

    def __init__(self, radius=5, size=100, q_init=None, q_goal=None, obstacles=None,
                 visualize=False):
        self.size = size
        self.lower_bound = -1*self.size
        self.range = self.size - self.lower_bound
        self.radius = radius
        if obstacles is None:
            self.obstacles = []
        else:
            self.obstacles = obstacles

        if q_init is None:
            self.q_init = np.array([0., 0.,])
        else:
            self.q_init = q_init

        if q_goal is None:
            self.q_goal = np.array([0., 0.])
        else:
            self.q_goal = q_goal

        # Environment degrees of freedom
        self.dof = 2
        self.step_size = 2.5

    def set_start(self, x, y):
        self.q_init = np.array([x, y])

    def set_goal(self, x, y):
        self.q_goal = np.array([x, y])

    def add_obstacle(self, x1, y1, x2, y2):
        self.obstacles.append((x1, y1, x2, y2))

    def set_size(self, size, lower_bound=None):
        self.size = size
        if lower_bound is None:
            self.lower_bound = -1*self.size
        else:
            self.lower_bound = lower_bound
        self.range = self.size - self.lower_bound

    def is_valid(self, q):
        """Example valid function"""
        for i in range(2):
            if not q[i] <= self.size and q[i] >= self.lower_bound:
                return False
        x, y = q[0], q[1]
        for obs in self.obstacles:
            x1, x2, y1, y2 = obs
            if (x > x1 and x < x2 and y > y1 and y < y2):
                return False
        return True

    def sample(self):
        """Sample positions in the space."""
        return (np.random.random(2) * self.range) + self.lower_bound

    def goal(self):
        """Example goal sampler."""
        return np.copy(self.q_goal)

    def is_done(self, q):
        """Example done function (reached the goal)"""
        return np.sum((self.q_goal - q) ** 2) < 1e-5

    def show(self, trees, fig=1):
        plt.figure(fig)

        # Support multiple trees
        if not isinstance(trees, list):
            trees = [trees]

        # plot the tree(s)
        for tree in trees:
            x = np.array([n.q[0] for n in tree.nodes])
            y = np.array([n.q[1] for n in tree.nodes])
            plt.plot(x, y, 'b.')

        # Draw obstacles in red
        for obs in self.obstacles:
            x1, x2, y1, y2 = obs
            w, h = x2 - x1, y2 - y1
            rect = plt.Rectangle((x1, y1), w, h, color='r', fill=True)
            plt.gca().add_patch(rect)

        # Draw circles at start and goal
        start = patches.Circle(self.q_init, radius=int(0.1*self.size), color='c')
        plt.gca().add_patch(start)
        goal = patches.Circle(self.q_goal, radius=int(0.1*self.size), color='c')
        plt.gca().add_patch(goal)


        # plot the tree(s)
        for tree in trees:
            x = np.array([n.q[0] for n in tree.nodes])
            y = np.array([n.q[1] for n in tree.nodes])
            plt.plot(x, y, 'b.')

    def show_path(self, path, trees, fig=1, wait=True):
        self.show(trees, fig)
        if path is not None:
            x = np.array([n[0] for n in path])
            y = np.array([n[1] for n in path])
            plt.plot(x, y, 'g.')
        if wait:
            plt.show()

