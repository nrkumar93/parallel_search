# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import random

from brain2.learning.models import EncoderCNN
from brain2.learning.models import EncoderRNN


class PlanningDomainModel(nn.Module):
    """ Learned planner model - needs to be able to predict which future actions are
    possible/impossible given current observation of the world state.

    Question:
    - how do we represent these actions? just as a vector?
    - how do we represent the transition function? same way?

    Planner operates over masks.
    """

    def __init__(self, encoder, decoder, transition, image_encoder, action_dim, hidden_dim, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition
        self.device = device
        self.image_encoder = image_encoder
        self.action_dim = action_dim

        # Encode verb and object here
        d2 = hidden_dim
        d3 = int(hidden_dim * 0.5)
        self.verb_encoder = nn.Sequential(
            nn.Embedding(action_dim, d2),
            nn.Linear(d2, d3),
            nn.ReLU())
        self.obj_encoder = nn.Sequential(
            nn.Embedding(action_dim, d2),
            nn.Linear(d2, d3),
            nn.ReLU())
        d4 = int(hidden_dim * 2)
        d5 = hidden_dim
        d6 = int(hidden_dim * 0.5)
        # This model takes an encoded action (verb + object) and predicts if
        # it's executable from the given hidden state.
        self.query_encoder = nn.Sequential(
            nn.Linear(d4, d5),
            nn.ReLU(),
            nn.Linear(d5, d6),
            nn.ReLU(),
            nn.Linear(d6, 1))
        # This network operates on hidden state outputs from the forward-prediction
        # model described above
        self.value_predictor = nn.Sequential(
            nn.Linear(d4, d5),
            nn.ReLU(),
            nn.Linear(d5, d6),
            nn.ReLU(),
            nn.Linear(d6, 1),
            nn.Sigmoid())
        # This predicts the verb from a particular hidden state
        # 
        self.verb_predictor = nn.Sequential(
            nn.Linear(d4, d5),
            nn.ReLU(),
            nn.Linear(d5, d6),
            nn.ReLU(),
            nn.Linear(d6, action_dim))
        # And this predicts the object embedding
        # Eventually might need to have different dimensionality
        d4b = d4 + action_dim
        self.obj_predictor = nn.Sequential(
            nn.Linear(d4b, d5),
            nn.ReLU(),
            nn.Linear(d5, d6),
            nn.ReLU(),
            nn.Linear(d6, action_dim))

    def query_actions(self, goal, hiddens, verbs, objects, verbs_1hot=None):
        """ Returns if actions are possible from these hidden states. Note
        that this is pretty mch entirely for TRAINING, not for testing."""
        batch_size, length, hdim = hiddens.shape
        vdim = verbs.shape[-1]
        odim = objects.shape[-1]
        hiddens = hiddens.view(-1, hdim)
        verbs = verbs.view(-1)
        objects = objects.view(-1)
        v = self.verb_encoder(verbs)
        o = self.obj_encoder(objects)
        x = torch.cat([hiddens, v, o], dim=-1)
        res = self.query_encoder(x)

        goal = goal[:,None].repeat(1,length,1).view(-1, hdim)
        goal_and_state = torch.cat([goal, hiddens], dim=-1)

        # Value + action predictions
        val = self.value_predictor(goal_and_state).view(-1, length)
        verb = self.verb_predictor(goal_and_state).view(-1, length, self.action_dim)
        if verbs_1hot is not None:
            verbs_1hot = verbs_1hot.view(-1, self.action_dim)
            goal_and_state_and_verb = torch.cat([goal, hiddens, verbs_1hot], dim=-1)
        else:
            goal_and_state_and_verb = torch.cat([goal, hiddens, verb], dim=-1)
        obj = self.obj_predictor(goal_and_state_and_verb).view(-1, length, self.action_dim)
        return (torch.sigmoid(res.view(-1, length)),
                val, verb, obj)

    def predict_actions(self, goal, hiddens):
        """ Predict the next actions, conditioned on goal representation(s)
        predicted themselves from language + the hidden state we are currently
        located in. """
        _, length, hdim = hiddens.shape
        hiddens = hiddens.view(-1, hdim)
        verb = self.verb_predictor(hiddens)
        obj = self.obj_predictor(hiddens)
        verb = verb.view(-1, length)
        obj = obj.view(-1, length)
        return verb, obj

    def eval_state(self, state, goal):
        """ TEST CODE. Get the best verb + object. """
        goal_and_state = torch.cat([goal, state], dim=-1)
        val = self.value_predictor(goal_and_state)
        verb = self.verb_predictor(goal_and_state)
        best_verb = torch.argmax(verb, dim=-1)
        _verbs = torch.zeros_like(verb).to(self.device)
        _verbs.scatter(1, best_verb[:,None], 1)
        goal_and_state_and_verb = torch.cat([goal, state, _verbs], dim=-1)
        obj = self.obj_predictor(goal_and_state_and_verb)
        best_obj = torch.argmax(obj, dim=-1)
        return val, best_verb, best_obj

    def is_done(self, state, goal):
        """ TEST CODE. Get the best verb + object. """
        goal_and_state = torch.cat([goal, state], dim=-1)
        val = self.value_predictor(goal_and_state)
        return val

    def _sample_from_discrete(self, dist):
        """ generate samples from this distribution """
        m = torch.distributions.Categorical(probs)
        v = m.sample()
        best_verb = torch.argmax(v, dim=-1)
        _verbs = torch.zeros_like(verb).to(self.device)
        _verbs.scatter(1, best_verb[:,None], 1)
        return _verbs

    def sample_action(self, state, goal):
        """ sample a whole batch of actions from this state. """
        goal_and_state = torch.cat([goal, state], dim=-1)
        verb = self.verb_predictor(goal_and_state)
        verb = self._sample_from_discrete(verb)
        goal_and_state_and_verb = torch.cat([goal, state, verb], dim=-1)
        obj = self.obj_predictor(goal_and_state_and_verb)
        obj = self._sample_from_discrete(obj)
        return verb, obj

    def sample_rollout(self, state, goal, max_depth=48):
        for i in range(max_depth):
            verb, obj = self.sample_action(state, goal)
            state = self.predict(state, verb, obj)

    def predict(self, state, verb, obj):
        """ Roll forward hidden state """
        v = self.verb_encoder(verb)[None]
        o = self.obj_encoder(obj)[None]
        _, hidden = self.transition(state[None], v, o)
        return hidden[0]

    def forward(self, src, imgs,
                trg=None, verbs=None, objs=None,
                teacher_forcing_ratio = 0.0,
                mode='train', max_len=128,
                end_token_index=2):
        """ forward. similar to seq2seq but with the addition of some action information. """
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        if verbs is not None:
            cmd_len = verbs.shape[1]
        else:
            cmd_len = max_len
        trg_vocab_size = self.decoder.output_dim
        hidden_size = self.decoder.hidden_dim
        
        # Encode images
        imgs_embeddings = self.image_encoder(imgs)
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, h0 = self.encoder(src, imgs_embeddings)

        #first input to the decoder is the <sos> tokens
        inp = trg[0,:]

        if mode == 'test':
            trg_len = max_len
            # <START> token for first time step
            inp = torch.ones((batch_size), dtype=torch.long).to(self.device)

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hiddens = torch.zeros(cmd_len, batch_size, hidden_size).to(self.device)

        # This loop is the same as before - it predicts the "goal" term for the planner,
        hidden = h0  # Initialize the beginning of the loop with language + img
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(inp, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # in test mode, stop at <END> token
            if mode == 'test' and top1.item() == end_token_index:
                break
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            inp = trg[t] if teacher_force else top1

        if verbs is not None:
            # This loop predicts the set of hidden states we'll move through to
            # get there.
            hidden = imgs_embeddings[None]  # Initialize based on current state
            for i in range(0, cmd_len):
                # Get the hidden state BEFORE the transition occurs
                hiddens[i] = hidden
                v = self.verb_encoder(verbs[:,i])[None]
                o = self.obj_encoder(objs[:,i])[None]
                _, hidden = self.transition(hidden, v, o)
        else:
            # Just return the initial hidden state for use in planning
            hiddens[0] = hidden

        return outputs, h0, hiddens
