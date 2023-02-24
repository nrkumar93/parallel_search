# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import re
import string
import torch
import numpy as np
import json
import os

def tokenize(ann, ignore_punctuation=','):
    """ Tokenize an annnotation into a sequence of sentences """
    tks = re.split('[,-.;" (){}]', ann)
    tks = [tk.strip().translate(str.maketrans('', '',\
        string.punctuation.replace(ignore_punctuation, ''))) for tk in tks]
    tks = [tk.lower().replace("“","").replace("”","") for tk in tks if len(tk) > 0]
    return tks


class Vocabulary(object):
    """
    Create a vocabulary object so that we can convert everything into integers
    This is used to tokenize as well
    """
    def __init__(self, vocab_file=None, verbose=False):
        self.verbose = verbose
        self.done = False
        if vocab_file is not None and os.path.isfile(vocab_file):
            # Load from vocabulary file
            self.load_vocabulary(vocab_file)
        else:
            # will be building a new vocab
            self.vocab = {"<NONE>": 0, "<START>": 1, "<END>": 2}
            self.vocabi = ["<NONE>", "<START>", "<END>"]
        self.next_id = len(self.vocabi)

    def add_token(self, token):
        if token not in self.vocab and len(token) > 0:
            self.vocab[token] = self.next_id
            self.next_id += 1
            self.vocabi.append(token)

    def add_sentence(self, sentence):
        # print(sentence)
        toks = tokenize(sentence)
        if len(toks) < 2:
            toks.append('done')
        for tok in toks:
            if len(tok) == 0:
                continue
            tok = tok.lower()
            self.add_token(tok)

    def embed_action(self, action):
        """ Specifically parse only an action string. """
        tokenized = tokenize(action)
        length = len(tokenized)
        if length == 2:
            return self.from_word(tokenized[0]), self.from_word(tokenized[1])
        elif length == 1:
            return self.from_word(tokenized[0]), 0
        else:
            raise RuntimeError('unsupported number of terms in action: ' + str(length))

    def embed_sentence(self, sentence, length=32):
        """ This computes an integer embedding for a RAW sentence from the data set.
        No padding, no special tokens."""
        iname = torch.zeros(length, dtype=torch.int64)
        tokenized_sentence = tokenize(sentence)
        if len(tokenized_sentence) < 2:
            tokenized_sentence.append('done')
        # add start and end token
        tokenized_sentence = ['<START>'] + tokenized_sentence + ['<END>']
        for i, w in enumerate(tokenized_sentence):
            if i >= length: # truncate sentences longer than length
                iname[-1] = 2
                return iname
            iname[i] = self.from_word(w)
        return iname

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, x):
        return self.vocab[x]

    def has(self, x):
        return x in self.vocab

    def convert(self, words):
        """ give us tensor embedding for word """
        return torch.as_tensor([self.vocab[w] for w in words]).cuda()

    def from_word(self, word):
        UNK = 0
        if word not in self.vocab:
            return UNK
        return self.vocab[word]

    def to_word(self, token):
        return self.vocabi[token]

    def from_idx(self, idx):
        for key, value in self.vocab.items():
            if value == idx:
                return key
        return('<NONE>')

    def save_vocabulary(self, save_file):
        '''
        Dump vocabularu as json for viz.
        '''
        with open(save_file, 'w') as f:
            json.dump(self.vocab, f, indent=4)

    def load_vocabulary(self, load_file):
        '''
        Load vocabularu as json.
        '''
        with open(load_file, 'r') as f:
            vocab = json.load(f)
        self.vocab = vocab
        self.vocabi = list(vocab.keys())
        for i, v in enumerate(self.vocabi):
            assert(i == self.vocab[v])
        self.next_id = len(self.vocabi)

    def add_goal_states(self, goal, states):
        '''
        Add sym_goal states to sub_goals.
        '''
        sent = []
        tokens = [e.replace(',','')+')' for e in goal.split('),') if e]
        for (subgoal, state) in zip(tokens, states.tolist()):
            sent.append(subgoal)
            sent.append(str(state))
            sent.append(',')
        return (' '.join(sent[:-1]))

    def add_plan_states(self, plan):
        '''
        Clean plan symbols to retain commas.
        '''
        sent = plan.replace(',', ' , ')
        # also return separate verb and objects
        if len(sent) > 1:
            verbs = ' '.join([x.strip().split('(')[0] for x in sent.split(',')])
            objects = ' '.join([x.strip().split('(')[1][:-1] for x in sent.split(',')])
        else:
            verbs = ''
            objects = ''
        return sent, verbs, objects
