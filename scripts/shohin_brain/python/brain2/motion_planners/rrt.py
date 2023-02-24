# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np


def get_configs(nodes):
    """Get array of configurations"""
    # TODO: make these using numpy tools
    # return np.array(list(map(lambda node: node.q, nodes)))
    return list(map(lambda node: node.q, nodes))


class Node(object):
    """
    Contains information regarding a single node in the tree.
    """

    def __init__(self, q, parent=None, cost=0):
        self.q = q
        self.parent = parent
        self.cost = cost

    def backtrace(self):
        """
        Get full trajectory ending in this point, going from the root of the
        tree all the way back up to the final leaf node.
        """
        trace = []
        node = self
        while node is not None:
            trace.append(node)
            # print("BACKTRACE COST", node.cost)
            node = node.parent
        trace.reverse()
        return trace

    def reconnect(self, new_parent, new_cost):
        self.parent = new_parent
        self.cost = new_cost


class Tree(object):
    """
    Contains a whole set of tree nodes. Presumably empty to begin with.
    """
    
    def __init__(self, problem=None, q0=None):
        self.q0 = q0
        self.nodes = [Node(q0)]
        self.goal_nodes = []
        self.problem = problem
        self.found_goal = False

    def insert(self, q_new, parent=None, cost=0):
        self.nodes.append(Node(q_new, parent, cost))
        n = self.nodes[-1]
        return n

    def add_goal(self, node):
        """ For basic RRTs"""
        self.goal_nodes.append(node)

    def is_done(self):
        """Have we reached any goals"""
        return len(self.goal_nodes) > 0

    def backtrace(self):
        return self.goal_nodes[-1].backtrace()

    def get_path(self):
        if self.is_done():
            path = get_configs(self.backtrace())
            if self.problem.shortcut:
                return self.random_shortcut(path)
            else:
                return path
        else:
            return None

    def random_shortcut(self, path, iterations=None):
        """Run random shortcutting algorithm. Included in the base tree class
        since it's pretty useful as a tool for many of these different
        algorithms. More interations will improve quality of the resulting path."""
        smoothed_path = path
        if iterations is None:
            iterations = self.problem.shortcut_iterations
        for _ in range(iterations):
            # choose two indices at random
            if len(smoothed_path) <= 2:
                # not enough points to bother smoothing
                return smoothed_path

            # Get random indices
            i = np.random.randint(len(smoothed_path))
            j = np.random.randint(len(smoothed_path))
            if abs(i - j) < 2:
                # these are the same index or are adjacent, get new ones
                continue
            if j < i:
                # flip the indices
                i, j = j, i
            
            # Connect two random indicesa
            shortcut = list(self.problem.extend(smoothed_path[i], smoothed_path[j]))

            # verify path is valid and actually a shortcut
            if len(shortcut) >= j - i:
                continue

            # check for collisions and validity
            if all(self.problem.is_valid(q) for q in shortcut):
                # reset the list
                smoothed_path = (smoothed_path[:(i + 1)] + shortcut
                                 + smoothed_path[(j + 1):])
            else:
                if self.problem.verbose:
                    print("invalid addition:", i, j)

        return smoothed_path

    def get_closest(self, q):
        """Naive implementation of getting closest point: loop over all nodes,
        choose the closest one, and return it. Really we should use some sort
        of spatial data structure.
        
        Note that right now this is by far the slowest part of the code."""

        min_node = None
        min_dist = float('inf')
        for node in self.nodes:
           distn = self.problem.distance(node.q, q)
           if distn < min_dist:
               min_node = node
               min_dist = distn
        return min_node, min_dist

        # Less FOR loop, maybe faster?
        # With some testing -- does not seem like it
        # TODO remove this code
        # dists = [self.problem.distance(q, node.q) for node in self.nodes]
        # return self.nodes[dists.index(min(dists))]

        # TODO this is the slowest idea yet
        # dists = [self.problem.distance(q, node.q) for node in self.nodes]
        # return self.nodes[np.argmin(dists)]

    def get_neighbors(self, q, radius):
        """Naive implementation. finds all neighbors within a certain distance
        of a new configuration q."""
        neighbors = []
        min_node = None
        min_cost = float('inf')
        for node2 in self.nodes:
            distn = self.problem.distance(q, node2.q)
            costn = distn + node2.cost
            if distn < radius:
                if costn < min_cost:
                    min_node = node2
                    min_cost = costn
                # Add to neighborhood
                neighbors.append((node2, distn))

        #return min_node, min_dist, neighbors
        return min_node, min_cost, neighbors


def rrt(q0, problem):
    """
    Run a simple RRT. Extend a tree a certain number of iterations.
    """

    # Create tree to store the whole search problem.
    if isinstance(q0, Tree):
        tree = q0
        # Reset planning problem if necessary
        tree.problem = problem
    else:
        tree = Tree(problem, q0)

    # Sanity check
    if not problem.is_valid(q0):
        if problem.verbose:
            print("Given start config", q0, "was invalid.")
        return None, tree

    for i in range(problem.iterations):
        qi = problem.sample_next_q(i)

        if qi is None:
            if problem.verbose:
                logwarn("Iteration %d: sampling failed", i)
            continue
        elif not problem.is_valid(qi):
            if problem.verbose:
                logwarn("Iteration %d: sampled invalid goal", i)
            continue

        # TODO: this is definitely just the slowest part of the procedure.
        prev_node, _ = tree.get_closest(qi)

        if problem.verbose:
            print(i, qi, "<----", prev_node.q)

        # Inner loop: extend tree towards next configuration. If you provide 
        # the right sort of extend() function, it'll keep extending until you
        # reach qi.
        for q_next in problem.extend(prev_node.q, qi):
            if problem.is_valid(q_next):
                prev_node = tree.insert(q_next, prev_node)
                # Add goal node
                if problem.is_done(q_next):
                    tree.add_goal(prev_node)

                if tree.is_done():
                    return tree.get_path(), tree
            else:
                break

    return None, tree
