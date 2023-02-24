# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



import numpy as np

from .rrt import Tree, get_configs
from brain2.utils.info import logwarn, logerr

class BiTree(Tree):
    """ Bi-directional tree structure variant for RRT-connect """

    def __init__(self, *args, **kwargs):
        super(BiTree, self).__init__(*args, **kwargs)
        self.other_tree = None

    def connect(self, tree):
        """Reference to a second tree"""
        self.other_tree = tree
        self.other_tree.other_tree = self

    def add_goal(self, node1, node2):
        """ for RRT-connect """
        self.goal_nodes.append((node1, node2))

    def get_path(self):
        node1, node2 = self.goal_nodes[-1]
        path = get_configs(node1.backtrace() + node2.backtrace()[::-1])
        if self.problem.shortcut:
            return self.random_shortcut(path,
                    iterations=self.problem.shortcut_iterations)
        else:
            return path


def rrt_connect(q0, problem):
    """
    Run a simple RRT. Extend a tree a certain number of iterations.
    """

    # Create tree to store the whole search problem.
    if isinstance(q0, Tree):
        tree1 = q0
    else:

        # Create data structures for the two trees
        # First, the start tree
        tree1 = BiTree(problem, q0)

    # Second, add a goal tree
    # For now, just sample final goal and assume that's it
    for i in range(problem.goal_iterations):
        q_goal = problem.sample_goal()
        if q_goal is not None and problem.check_goal_valid:
            if not problem.is_valid(q_goal):
                print("invalid goal", q_goal)
                q_goal = None
            else:
                break

    if q_goal is None:
        if problem.verbose:
            logerr("No valid goals found!")
        return None, tree1
    tree2 = BiTree(problem, q_goal)
    tree1.connect(tree2)

    # Sanity check
    if not problem.is_valid(q0) and problem.check_start_valid: 
        if problem.verbose:
            logerr("Invalid start for planning problem! Start = " + str(q0))
        return None, tree1

    # First we can try to just connect directly with the goal
    # We only want to do this if this is a newly-made tree
    prev_node1 = tree1.nodes[0]
    for q_next in problem.extend(q0, q_goal):
        if problem.is_valid(q_next):
            prev_node1 = tree1.insert(q_next, prev_node1)
        else:
            break
    else:
        # No problem connecting them, so we're done here
        tree1.found_goal = True
        tree1.add_goal(prev_node1, tree2.nodes[0])
        return tree1.get_path(), tree1

    # This is the actual tree from the source.
    tree0 = tree1

    # Main loop: try to grow the tree this many times.
    for i in range(problem.iterations):
        # We do not sample goals here -- just intermediate points
        # Get a new configuration
        qi = problem.sample()
        #problem.show_config(qi)

        if qi is None:
            if problem.verbose > 1:
                logwarn("Iteration %d: sampling failed", i)
            continue

        # TODO: this is definitely just the slowest part of the procedure.
        prev_node1, _ = tree1.get_closest(qi)
        #problem.show_config(prev_node1.q)
            
        # Inner loop: extend tree towards next configuration. If you provide 
        # the right sort of extend() function, it'll keep extending until you
        # reach qi.
        # This one grows the "first" tree
        for q_next in problem.extend(prev_node1.q, qi):
            if problem.is_valid(q_next):
                prev_node1 = tree1.insert(q_next, prev_node1)
                break
            else:
                break
        
        # TODO: this is definitely just the slowest part of the procedure.
        # Find closest point we reached to the next tree
        prev_node2, _ = tree2.get_closest(prev_node1.q)
        #problem.show_config(prev_node2.q)

        # Greedily try to connect the two trees
        for q_next in problem.extend(prev_node2.q, prev_node1.q):
            if problem.is_valid(q_next):
                prev_node2 = tree2.insert(q_next, prev_node2)
                #problem.show_config(q_next) # TODO
            else:
                break
        else:
            # If we successfully connected the two trees (i.e. we did not 
            # break in the for-loop above)...
            # We are now done. Get the two trees and return.
            tree1.add_goal(prev_node1, prev_node2)
            tree1.found_goal = True

        if tree0.found_goal and i > problem.min_iterations:
            return tree0.get_path(), tree0

        # Otherwise flip them around
        tmp = tree1
        tree1 = tree2
        tree2 = tmp

    return None, tree0
