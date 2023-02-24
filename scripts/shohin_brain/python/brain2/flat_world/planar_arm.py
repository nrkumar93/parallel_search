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
import matplotlib.lines as mlines

from brain2.flat_world.interface import FlatWorld

class PlanarArm(object):
    """ defines a simple n-link planar arm """
    def __init__(self, links=None):
        if links is None:
            links = [20, 20, 20]
        self.links = links
        self.dof = len(self.links)

    def sample_uniform(self):
        q = (np.random.random(self.dof) * 2 * np.pi) - np.pi
        print(q)
        return q

    def fwd_links(self, q):
        pt = np.array([0., 0.])
        theta = 0.
        pts = [pt]
        assert len(q) == self.dof
        for i, link in enumerate(self.links):
            theta += q[i]
            pt = pt + np.array([np.cos(theta), np.sin(theta)]) * link
            pts.append(pt)
        return pts

    def fwd(self, q):
        """ does forward kinematics to the end effector """
        return self.fwd_links(q)[-1]

    def is_valid(self, q, env):
        pts = self.fwd_links(q)
        for pt in pts:
            for i in range(2):
                if not pt[i] <= env.size and pt[i] >= env.lower_bound:
                    return False
            x, y = pt[0], pt[1]
            for obs in env.obstacles:
                x1, x2, y1, y2 = obs
                if (x > x1 and x < x2 and y > y1 and y < y2):
                    return False
        return True


class PlanarArmWorld(FlatWorld):
    """ Planar arm world. 2d environment for motion planning tests """

    def __init__(self, arm, size=100, q_init=None, q_goal=None, obstacles=None, visualize=False):
        super(PlanarArmWorld, self).__init__(5, size, q_init, q_goal, obstacles, visualize)
        self.arm = arm
        self.dof = self.arm.dof
        self.step_size = 0.1

    def sample(self):
        return self.arm.sample_uniform()

    def is_valid(self, q):
        return self.arm.is_valid(q, self)

    def set_start(self, *dims):
        self.q_init = np.array(list(dims))

    def set_goal(self, *dims):
        self.q_goal = np.array(list(dims))

    def show(self, trees, fig=1):
        plt.figure(fig)

        # Support multiple trees
        if not isinstance(trees, list):
            trees = [trees]

        # plot the tree(s)
        for tree in trees:
            pts = [self.arm.fwd(n.q) for n in tree.nodes]
            x = np.array([n[0] for n in pts])
            y = np.array([n[1] for n in pts])
            plt.plot(x, y, 'b.')

        # Draw obstacles in red
        for obs in self.obstacles:
            x1, x2, y1, y2 = obs
            w, h = x2 - x1, y2 - y1
            rect = plt.Rectangle((x1, y1), w, h, color='r', fill=True)
            plt.gca().add_patch(rect)

        # Draw circles at start and goal
        start = patches.Circle(self.arm.fwd(self.q_init), radius=int(0.1*self.size), color='c')
        plt.gca().add_patch(start)
        goal = patches.Circle(self.arm.fwd(self.q_goal), radius=int(0.1*self.size), color='c')
        plt.gca().add_patch(goal)

        # plot the tree(s)
        for tree in trees:
            configs = [self.arm.fwd_links(n.q) for n in tree.nodes]
            for pts in configs:
                x = np.array([n[0] for n in pts])
                y = np.array([n[1] for n in pts])
                plt.plot(x, y, 'b-o')

    def show_path(self, path, trees, fig=1, wait=True):
        self.show(trees, fig)
        if path is not None:
            configs = [self.arm.fwd_links(q) for q in path]
            for pts in configs:
                x = np.array([n[0] for n in pts])
                y = np.array([n[1] for n in pts])
                plt.plot(x, y, 'g-o')
        plt.show()
