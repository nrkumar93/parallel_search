# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import print_function

import copy
import numpy as np
from brain2.robot.kitchen import _checkCollisions
from brain2.robot.kitchen import _placeReachable

from brain2.utils.sampling import choose2, choose3, choose4
from brain2.language.parsing import tokenize
from brain2.language.parsing import tokenize_aggressive
from brain2.language.parsing import remove_filler


def _placeAllOnTable(iface, robot, surface, objs, ik_solver=None, max_tries=100):
    """ Create a scene that's known to be valid. Make sure everything makes sense and is somewhat
    reasonable. Don't create impossible scenes. """
    valid = False
    tries = 0
    while not valid:
        tries += 1
        if tries > max_tries:
            input('Press enter to terminate')
            raise RuntimeError('Could not create environment!')
        i = 0
        placed_objs = []
        for obj_name in objs:
            obj = iface.get_object(obj_name)
            # Don't check ignored objects
            placed = False
            while not placed and i < 100:
                # Place objects and make sure they're close enough if that's important
                placed = _placeReachable(surface, robot, obj, z=0.)
                print('- placed', obj_name, '=', placed)
                if placed:
                    # Check to see if blocks are in collision or generally just too close to each
                    # other. If so, then we need to re-sample their positions.
                    placed = _checkCollisions(iface, robot, placed_objs + [obj_name])
                    print('- placed', obj_name, 'collision check =', placed)
                if placed:
                    placed_objs.append(obj_name)
                    break
                i += 1
        valid = placed
    return valid

def _placeAll(iface, robot, surface, objs, ik_solver=None, max_tries=100,
        stack=False,
        stack_height=0.05):
    """ Create stacks. place objects on the table if they are not already there. stacks should be
    valid, and we should make sure things are still reasonable and make good sense.
    
    This should handle any real domain-specific stuff, like creating towers/pyramids."""
    valid = _placeAllOnTable(iface, robot, surface, objs, ik_solver, max_tries)
    if not valid: return False
    if stack:
        # randomly stack some of the objects
        num_bases = np.random.randint(1, len(objs) + 1)
        bases = np.random.choice(objs, num_bases, replace=False)
        stacks = copy.copy(bases)
        for obj in objs:
            if obj in bases:
                continue
            i = np.random.randint(len(bases))
            # add this to the i-th stack.
            obj_ref = iface.get_object(obj)
            base_ref = iface.get_object(stacks[i])
            pose = base_ref.get_pose(matrix=True).copy()
            pose[2, 3] += stack_height
            stacks[i] = obj
            obj_ref.set_pose_matrix(pose)
    return True

def GetLeonardoProblems():
    return list(_leonardo_problems.keys())

def TaskLeonardo(iface, ik_solver, problem=None, seed=None, begin_stacked=True):
    """
    Block stacking and rearrangement with language
    """
    all_objs = []
    colors = ["red", "green", "blue", "yellow"]
    blocks = [color + "_block" for color in colors]

    # Set up the camera position
    # pos, quat =  [0.492, 0.160, 1.128], [-0.002, 0.823, -0.567, 0.010]
    # TODO: this should really be set by the system itself.
    #  Computed via:
    #   rosrun tf tf_echo measured/base_link depth_camera_link
    #pos, quat = [0.509, 0.993, 0.542], [-0.002, 0.823, -0.567, -0.010]
    pos, quat = [0.428, 0.988, 0.634], [-0.012, 0.798, -0.601, 0.022]
    iface.set_camera((pos, quat), matrix=False)

    if seed is not None:
        np.random.seed(seed)

    table = iface.get_object("table")
    #tabletop = table.get_surface("top")
    #tabletop = table.get_surface("tower")
    tabletop = table.get_surface("workspace")
    robot = iface.get_object("robot")
    
    args = iface, robot, tabletop, blocks, ik_solver
    _placeAll(iface, robot, tabletop, blocks, ik_solver, stack=begin_stacked)
    if problem is None or len(problem) == 0:
        r = np.random.randint(90)
        for lower, upper, function in _leonardo_problems.values():
            if r >= lower and r <= upper:
                subcode = r - lower
                return r, function(subcode, *args)
        else:
            raise RuntimeError('failed to reach generation case for r = ' + str(r))
    else:
        l, u, function = _leonardo_problems[problem]
        r = np.random.randint(l, u)
        subcode = r - l
        return r, function(subcode, *args)

def _get_tower_2(block1, block2):
    """ tower has block1 on top, block2 on table """
    block1, block2 = _get_block(block1), _get_block(block2)
    opts = ["make a tower with %s on top of %s" % (block1, block2),
            "put %s on top of %s and %s on the ground" % (block1, block2, block2),
            "stack %s on top of %s" % (block1, block2),
            "make a stack with %s then %s" % (block2, block1),
            "put %s on the bottom and %s above it" % (block2, block1),
            "make a stack with %s on top of %s" % (block1, block2)]
    r = np.random.randint(len(opts))
    return opts[r]

def _get_tower_3(block1, block2, block3):
    """ tower has block1 on top, block2 on block3 on table """
    block1, block2 = _get_block(block1), _get_block(block2)
    block3 = _get_block(block3)
    opts = ["make a tower with %s on top of %s and %s" % (block1, block2, block3),
            "make a stack with %s on top of %s and %s on the ground" % (block1, block2, block3),
            "stack %s and then %s on top of %s" % (block2, block1, block3),
            "make a stack with %s %s then %s" % (block3, block2, block1),
            "put %s on the bottom and %s then %s above it" % (block3, block2, block1),
            "make a stack with %s on top of %s and %s" % (block1, block2, block3),
            "make a tower that goes top to bottom %s %s %s" % (block1, block2, block3),
            "make a tower that goes bottom to top %s %s %s" % (block3, block2, block1),]
    r = np.random.randint(len(opts))
    return opts[r]

def _get_two_towers_2(block1, block2, block3, block4):
    return _get_tower_2(block1, block2) + " and " + _get_tower_2(block3, block4)

def _get_1_not_on_table(block1):
    block = _get_block(block1)
    table = _get_surface("tabletop")
    opts = _get_not_on()
    opt = opts[np.random.randint(len(opts))]
    return opt % (block, table)

def _get_2_not_on_table(block1, block2):
    r = np.random.randint(2)
    if r == 0:
        term1, term2 = _get_1_not_on_table(block1), _get_1_not_on_table(block2)
        return term1 + " and " + term2
    else:
        block1 = _get_block(block1)
        block2 = _get_block(block2)
        opts = ["dont have either %s or %s on the table" % (block1, block2),
                "make it so neither %s nor %s is on the table" % (block1, block2),
                "make neither of %s and %s be on the table" % (block2, block1),
                "dont let %s or %s stay on the table" % (block2, block1)]
        return opts[np.random.randint(len(opts))]

def neg_1(sc, iface, robot, tabletop, blocks, ik_solver):
    if sc > 7: a = len(blocks) - 1
    else: a = np.random.randint(len(blocks)-1)
    block1 = blocks[a]
    goals = [("has_anything(robot)", False)]
    goals.append(("on_surface(%s, tabletop)" % block1, False))
    return blocks, goals, _get_1_not_on_table(block1)

def _get_1_dir(block1, loc):
    block1 = _get_block(block1)
    opts = ["put %s to the %s side of the table" % (block1, loc),
            "set %s on the %s of the table" % (block1, loc),
            "set %s to the %s of the table" % (block1, loc),
            "have %s on the %s" % (block1, loc),
            "put %s over to the %s" % (block1, loc),
            ]
    return opts[np.random.randint(len(opts))]

def sample_loc():
    loc = np.random.choice(["left", "right", "far", "center"])
    return loc

def dir_and_one(sc, iface, robot, tabletop, blocks, ik_solver):
    a, b, c = choose3(4, sc > 7)
    block1, block2, block3 = blocks[a], blocks[b], blocks[c]
    loc = sample_loc()
    r = np.random.randint(3)
    goals = [("has_anything(robot)", False)]
    if r == 0:
        str1 = _get_1_not_on_table(block2)
        str0 = _get_1_dir(block1, loc)
        goals.append(("on_surface(%s, %s)" % (block1, loc), True))
        goals.append(("on_surface(%s, tabletop)" % block2, False))
    elif r == 1:
        str0 = _get_1_dir(block2, loc)
        goals.append(("on_surface(%s, %s)" % (block2, loc), True))
        goals.append(("on_surface(%s, tabletop)" % block2, True))
        goals.append(("stacked(%s, %s)" % (block2, block1), True))
        str1 = _get_tower_2(block1, block2)
    elif r == 2:
        str0 = _get_1_dir(block3, loc)
        goals.append(("on_surface(%s, %s)" % (block3, loc), True))
        goals.append(("on_surface(%s, tabletop)" % block3, True))
        goals.append(("stacked(%s, %s)" % (block2, block1), True))
        goals.append(("stacked(%s, %s)" % (block3, block2), True))
        str1 = _get_tower_3(block1, block2, block3)
    choices = [
                "%s and also %s",
                "%s but %s",
                "%s and also make sure that you %s",
                "%s ... oh and %s",
                "%s , %s",
                "%s and %s", "%s and %s",]
    choice = np.random.choice(choices)
    r = np.random.randint(2)
    if r == 0:
        cmd = choice % (str0, str1)
    else:
        cmd = choice % (str1, str0)
    return blocks, goals, cmd

def block_dir(sc, iface, robot, tabletop, blocks, ik_solver):
    if sc > 7: a = len(blocks) - 1
    else: a = np.random.randint(len(blocks)-1)
    block1 = blocks[a]
    loc = sample_loc()
    goals = [("has_anything(robot)", False)]
    goals.append(("on_surface(%s, %s)" % (block1, loc), True))
    return blocks, goals, _get_1_dir(block1, loc)

def neg_2(sc, iface, robot, tabletop, blocks, ik_solver):
    a, b = choose2(4, sc > 7)
    block1 = blocks[a]
    block2 = blocks[b]
    goals = [("has_anything(robot)", False)]
    goals.append(("on_surface(%s, tabletop)" % block1, False))
    goals.append(("on_surface(%s, tabletop)" % block2, False))
    return blocks, goals, _get_2_not_on_table(block1, block2)

def neg_1_tower_2(sc, iface, robot, tabletop, blocks, ik_solver):
    a, b, c = choose3(4, sc > 7)
    block1, block2, block3 = blocks[a], blocks[b], blocks[c]
    goals = [("has_anything(robot)", False)]
    goals.append(("on_surface(%s, tabletop)" % block2, True))
    goals.append(("stacked(%s, %s)" % (block2, block1), True))
    goals.append(("on_surface(%s, tabletop)" % block3, False))
    return blocks, goals, _get_1_not_and_tower_2(block1, block2, block3)

def _get_1_not_and_tower_2(block1, block2, block3):
    t1 = _get_1_not_on_table(block3)
    t2 = _get_tower_2(block1, block2)
    r = np.random.randint(4)
    opts = {0: t1 + " and " + t2,
            1: t1 + " but " + t2,
            2: t2 + " and " + t1,
            3: t2 + " but " + t1,}
    return opts[r]


def one_tower_2(sc, iface, robot, tabletop, blocks, ik_solver):
    a, b = choose2(4, sc > 7)
    block1 = blocks[a]
    block2 = blocks[b]
    goals = [("has_anything(robot)", False)]
    goals.append(("on_surface(%s, tabletop)" % block2, True))
    goals.append(("stacked(%s, %s)" % (block2, block1), True))
    return blocks, goals, _get_tower_2(block1, block2)


def one_tower_3(sc, iface, robot, tabletop, blocks, ik_solver):
    a, b, c = choose3(4, sc > 7)
    block1 = blocks[a]
    block2 = blocks[b]
    block3 = blocks[c]
    goals = [("has_anything(robot)", False)]
    goals.append(("on_surface(%s, tabletop)" % block3, True))
    goals.append(("stacked(%s, %s)" % (block2, block1), True))
    goals.append(("stacked(%s, %s)" % (block3, block2), True))
    return blocks, goals, _get_tower_3(block1, block2, block3)


def two_towers_2(sc, iface, robot, tabletop, blocks, ik_solver):
    a, b, c, d = choose4(4, sc > 7)
    block1, block2, block3, block4 = blocks[a], blocks[b], blocks[c], blocks[d]
    goals = [("has_anything(robot)", False)]
    goals.append(("on_surface(%s, tabletop)" % block2, True))
    goals.append(("stacked(%s, %s)" % (block2, block1), True))
    goals.append(("on_surface(%s, tabletop)" % block4, True))
    goals.append(("stacked(%s, %s)" % (block4, block3), True))
    return blocks, goals, _get_two_towers_2(block1, block2, block3, block4)


def _get_stacked():
    opts = ["put %s on %s",
            "put %s on top of %s",
            "stack %s on %s",
            "stack %s on top of %s",
            "make %s on %s"]
    return opts


def _get_not_stacked():
    opts = ["make sure %s is not on %",
            "don't have %s on %s",
            "do not put %s on top of %s",
            "get %s off of %s",
            "remove %s from the top of %s",
            "take %s off of %s",
            "do not have %s on %s"]
    return opts


def _get_on():
    opts = ["put %s on %s",
            "place %s on %s",
            "make %s be on %s"]
    return opts


def _get_not_on():
    opts = ["don't put %s on %s",
            "don't place %s on %s",
            "do not place %s on %s",
            "do not have %s be on %s",
            "make %s not be on %s",
            "do not have %s sitting on %s",
            "don't have %s sitting on %s"]
    return opts


def _get_block(arg):
    if arg is None:
        raise RuntimeError('arg was missing and not a block')
    else:
        color = arg.split('_')[0]
        i = np.random.choice([None, "block", "cube", "square"])
        if i is not None:
            return color + " " + i
        else:
            return color


def _get_surface(arg):
    if arg == "tabletop":
        opts = ["table", "top", "the table", "tabletop",
                "table top", "the surface", "the top of the table"]
        idx = np.random.randint(len(opts))
        return opts[idx]
    elif arg == "center":
        opts = ["center", "the center", "the middle", "middle",
                "table top", "mid-table"]
        idx = np.random.randint(len(opts))
        return opts[idx]
    elif arg == "right":
        opts = ["the right", "the right side of the table", "right side wrt robot",
                "the right side with respect to the robot", "right from robot pov",
                "right", "right side"]
        idx = np.random.randint(len(opts))
        return opts[idx]
    elif arg == "left":
        opts = ["the left", "the left side of the table", "left side wrt robot",
                "the left side with respect to the robot", "left from robot pov",
                "left", "left side"]
        idx = np.random.randint(len(opts))
        return opts[idx]
    elif arg == "far":
        opts = ["far", "far way", "the far side of the table", "back of the table", "the back",
                "the back side", "remote", "away from the robot"]
        idx = np.random.randint(len(opts))
        return opts[idx]
    else:
        raise RuntimeError('arg not supported for surface: ' + str(arg))


def predicate_to_string(pred, arg1, arg2, value, verbose=False):
    if pred == "stacked":
        assert arg2 is not None
        # NOTE: flips these from what they should be; order was wrong before
        arg2, arg1 = _get_block(arg1), _get_block(arg2)
        opts = _get_stacked() if value else _get_not_stacked()
    elif pred == "on_surface":
        assert arg2 is not None
        arg1, arg2 = _get_block(arg1), _get_surface(arg2)
        opts = _get_on() if value else _get_not_on()
    elif pred == "has_anything":
        if verbose:
            print("!!! IGNORING HAS_ANYTHING")
        return None
    elif pred == "left_of":
        arg1, arg2 = _get_block(arg1), _get_block(arg2)
        opts = _get_left() if value else _get_not_left()
    elif pred == "right_of":
        arg1, arg2 = _get_block(arg1), _get_block(arg2)
        opts = _get_right() if value else _get_not_right()
    elif pred == "in_front_of":
        arg1, arg2 = _get_block(arg1), _get_block(arg2)
        opts = _get_front() if value else _get_not_front()
    elif pred == "behind":
        arg1, arg2 = _get_block(arg1), _get_block(arg2)
        opts = _get_behind() if value else _get_not_behind()
    else:
        raise RuntimeError('not understood: ' + str(pred))
    idx = np.random.randint(len(opts))
    choice = opts[idx]
    return choice % (arg1, arg2)


def goal_to_language(goal, num, verbose=False):
    """ Generate a goal and turn it into language """
    parsed = []
    for predicate, value in goal:
        tokens = tokenize(predicate)
        print(tokens)
        if len(tokens) == 2:
            pred, arg1, arg2 = tokens[0], tokens[1], None
        elif len(tokens) == 3:
            pred, arg1, arg2 = tokens[0], tokens[1], tokens[2]
        else:
            raise RuntimeError('failed to parse predicate: ' + str(predicate))
        parsed.append((pred, arg1, arg2, value))

    cmds = []
    for _ in range(num):
        np.random.shuffle(parsed)
        num_terms = len(parsed)
        lang = ""
        lang_terms = []
        for i, (pred, arg1, arg2, value) in enumerate(parsed):
            term = predicate_to_string(pred, arg1, arg2, value)
            if term is None:
                if verbose:
                    print("!!! SKIPPING:", pred, arg1, arg2, value)
                continue
            else:
                #lang += term
                lang_terms.append(term)
        num_terms = len(lang_terms)
        for i, term in enumerate(lang_terms):
            lang += term
            if i < num_terms - 1:
                lang += " and "
        cmds.append(lang)
    return cmds


def action_to_string(verb, with_obj, to_obj):
    if verb == "grasp":
        w, t = None, _get_block(to_obj)
        opts = ["grab %s", "grasp %s", "take %s"]
    elif verb == "approach":
        opts = ["reach for %s", "approach %s", "line gripper up with %s",
                "get ready to take %s", "align with %s"]
        w, t = None, _get_block(to_obj)
    elif verb == "stack":
        w, t = _get_block(with_obj), _get_block(to_obj)
        opts = ["stack %s on top of %s", "put %s on %s", "place %s on %s", "stack %s on %s"]
    elif verb == "align":
        w, t = _get_block(with_obj), _get_block(to_obj)
        opts = ["line %s up with %s", "move %s over top of %s", "align %s with %s",
                "move %s over %s"]
    elif verb == "place":
        t, w = _get_surface(with_obj), _get_block(to_obj)
        opts = ["put %s on %s", "move %s onto %s", "drop %s on %s",
                "place %s on %s"]
    elif verb == "lift":
        w, t = None, _get_block(to_obj)
        opts = ["lift %s", "pick up %s", "raise %s"]
    elif verb == "release":
        w, t = None, _get_block(to_obj)
        opts = ["release %s", "let go of %s", "open gripper with %s"]
    else:
        print("FAILED ON:", verb, with_obj, to_obj)
        raise RuntimeError('did not understand verb: ' + str(verb))

    i = np.random.randint(len(opts))
    action = opts[i]
    if w is None:
        return action % t
    else:
        return action % (w, t)


def _get_connector():
    connectors = ["then", "and then", "and", "and after that"]
    i = np.random.randint(len(connectors))
    return " " + connectors[i] + " "


def plan_to_language(plan, num):
    """ Get the list of actions and turn it into a plan description """
    parsed = []
    for action in plan:
        toks = remove_filler(tokenize_aggressive(action))
        parsed.append(toks)
    
    cmds = []
    for _ in range(num):
        cmd = ""
        num_actions = len(parsed)
        for i, action in enumerate(parsed):
            # Skippable actions
            if action[0] in ["approach", "align"]:
                i = np.random.randint(3)
                if i < 2:
                    continue
            verb = action[0]
            if len(action) == 3:
                to_obj = action[2]
                with_obj = action[1]
            elif len(action) == 4:
                # Case for "grasp from x, y"
                operator = action[1]
                assert operator == "from"
                to_obj = action[-1]
                # This is not important, should be treated as just grasping
                # Maybe should set to None?
                with_obj = action[-2]
            elif len(action) == 2:
                to_obj = action[1]
                with_obj = None
            cmd += action_to_string(verb, with_obj, to_obj)
            if i < num_actions - 1:
                cmd += _get_connector()
        cmds.append(cmd)
    return cmds

# =============================================================================
# =============================================================================
# Define the full set of problems with their relative probabilities.
# =============================================================================
# =============================================================================
_leonardo_problems = {"one_tower_2": (0, 10, one_tower_2),
                      "two_towers_2": (10, 20, two_towers_2),
                      "neg_1": (20, 30, neg_1),
                      "neg_2": (30, 40, neg_2),
                      "neg_1_tower_2": (40, 50, neg_1_tower_2),
                      "one_tower_3": (50, 60, one_tower_3),
                      "block_dir": (60, 70, block_dir),
                      "dir_and_one": (70, 90, dir_and_one),
                      #"neg_block_dir": (70, 80, neg_block_dir),
                      #"two_block_dir": (80, 90, two_block_dir),
                      }
