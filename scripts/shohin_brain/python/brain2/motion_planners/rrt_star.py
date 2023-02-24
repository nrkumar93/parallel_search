# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
from brain2.motion_planners.rrt import Node, Tree, get_configs
from collections import deque

def rrt_star(q0, problem):
    """
    Asymptotically optimal RRT.
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
        prev_node, distance = tree.get_closest(qi)

        if problem.verbose:
            print(i, qi, "<----", prev_node.q)

        # Inner loop: extend tree towards next configuration. If you provide 
        # the right sort of extend() function, it'll keep extending until you
        # reach qi.
        rewire_nodes = deque()
        for q_next in problem.extend(prev_node.q, qi):
            if not problem.is_valid(q_next): break
            # Insertion
            #distance = problem.distance(prev_node.q, q_next)
            #cost = prev_node.cost + distance
            #prev_node = tree.insert(q_next, prev_node, cost)

            # Rewire tree
            best_node, best_cost, neighbors = tree.get_neighbors(q_next, problem.neighborhood_radius)
            for node, cost in neighbors:
                #print("new parent cost =", prev_node.cost, "old =",
                #        node.parent.cost if node.parent is not None else 0,
                #        "dist =", cost)
                new_cost = prev_node.cost + cost
                if new_cost < node.cost:
                    rewire_nodes.append((prev_node, node))

            prev_node = tree.insert(q_next, best_node, best_cost)

            # Add goal node
            if problem.is_done(q_next):
                tree.add_goal(prev_node)
                break

            # Only once
            break

        # Rewiring step
        # Loop over rewiring nodes
        while len(rewire_nodes) > 0:
            prev_node, interim_goal = rewire_nodes.popleft()
            for q_next in problem.extend(prev_node.q, interim_goal.q):
                if not problem.is_valid(q_next): break
                #cost = prev_node.cost + distance
                distance = problem.distance(prev_node.q, q_next)
                distance_to_end = problem.distance(interim_goal.q, q_next)
                cost = prev_node.cost + distance

                # Add a new node
                if distance_to_end > problem.tolerance:
                    prev_node = tree.insert(q_next, prev_node, cost)

                # Just keep rewiring
                # _,  _, neighbors = tree.get_neighbors(prev_node, problem.neighborhood_radius)
                #for node, cost in neighbors:
                #    new_cost = prev_node.cost + cost
                #    if new_cost < node.cost:
                #        rewire_nodes.append((prev_node, node))
            else:
                # Add the nodeA
                interim_goal.reconnect(prev_node, prev_node.cost + distance)


    if tree.is_done():
        return tree.get_path(), tree
    else:
        return None, tree
