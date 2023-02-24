# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

"""
Tools for parsing predicates and stuff
"""

def tokenize(symbol):
    """ Strip a predicate name or action """
    tokens1 = symbol.split('(')
    tokens2 = []
    for token in tokens1:
        tokens2.append(token.split(','))
    tokens3 = []
    for tokens in tokens2:
        for token in tokens:
            tokens3.append(token.strip().strip(')'))
    return tokens3


def tokenize_aggressive(symbol):
    """ Strip a predicate name or action """
    tokens1 = symbol.split('(')
    tokens2 = []
    for token in tokens1:
        tokens2.append(token.split(','))
    tokens3 = []
    for tokens in tokens2:
        for token in tokens:
            tokens3.append(token.split('_'))
    tokens4 = []
    for tokens in tokens3:
        for token in tokens:
            tokens4.append(token.strip().strip(')'))
    return tokens4

def remove_filler(words):
    return [word for word in words if word not in ["obj", "block", "with", "on"]]
