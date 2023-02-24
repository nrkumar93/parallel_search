# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Contains code for training model"""

import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import os, math, time
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
import json
import matplotlib.pyplot as plt
import numpy as np

from brain2.learning.h5f_loader import get_rgb_transforms, blocks_seg_classes
from brain2.learning.models import (EncoderRNN, DecoderRNN, Attention, Seq2Seq,
                                    EncoderCNN, TransitionRNN)
from brain2.learning.mask_model import DecoderMaskRNN, MaskSeq2Seq, MaskSelectSeq2Seq
from brain2.learning.unet import UNet, UNetEncoder
from brain2.learning.models import init_weights, count_parameters
from brain2.learning.planner import PlanningDomainModel
from brain2.learning.image import resize_image
import brain2.learning.split as split
from brain2.language.vocab import Vocabulary
import brain2.utils.image as img

#torch.backends.cudnn.enabled = False

logger = logging.getLogger()
old_level = logger.level

lang_types = ['goal2goal', 'plan2plan', 'human2plan', 'human2goal', 'goallang2plan', 'goalsym2plan']

def add_lang_to_sym_args(parser):
    parser.add_argument('--mode', type=str, default='train', help='Options- train or test.')
    parser.add_argument('--data', default='/data/brain2/leonardo_resized')
    parser.add_argument('--resume', action='store_true', help='load weights and resume training')
    parser.add_argument('--lang_type', type=str, default='human2plan',
                        choices=lang_types,
                        help='Type of language "translation" to perform - choices are ' + str(lang_types))
    parser.add_argument('--exp_type', default='lang2sym')
    parser.add_argument('--model_path', default='/data/brain2/model')
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--max_plan_length', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50, help="number of training epochs to do")
    parser.add_argument('--workers', type=int, default=4, help="number of worker threads for data loader")
    parser.add_argument('--print_freq', type=int, default=10, help="how often to print info during an epoch (0 to skip and only print at end of epoch)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_input', type=int, default=0, help='have image input to enc (0/1)')
    parser.add_argument('--bidirectional', help='Bidirectional GRU', action='store_true')
    parser.add_argument('--generate_splits', type=int, default=1, help="regenerate data splits")
    parser.add_argument('--model', choices=['seq2seq', 'planner', 'maskpred', 'maskselect'], default='seq2seq', help="model class to train")
    return parser


def parse_args():
    parser = argparse.ArgumentParser('train language-planning model')
    parser = add_lang_to_sym_args(parser)
    return parser.parse_args()


class LangToSym(nn.Module):
    """
    Class for converting between language and images and symbolic instructions that can be executed
    on the robot.
    """
    learning_rate = 0.001
    enc_emb_dim = 256
    dec_emb_dim = 256
    enc_hid_dim = 256
    dec_hid_dim = 256
    enc_dropout = 0.2
    dec_dropout = 0.2

    # Weights for different loss terms
    transition_wt = 0.1
    value_wt = 1.
    action_wt = 0.01
    verb_wt = 0.3
    object_wt = 0.3
    mask_wt = 0.4

    def __init__(self, args, enc_vocab=None, dec_vocab=None, dec_verb_vocab=None, dec_object_vocab=None, lang_sym_keys=None, transform=None, num_seg_classes=None):
        super(LangToSym, self).__init__()
        self.args = args

        # Get the encoder vocabulary - input language
        self.enc_vocab = enc_vocab
        # Get the decoder vocabulary
        self.dec_vocab = dec_vocab
        self.dec_verb_vocab = dec_verb_vocab
        self.dec_object_vocab = dec_object_vocab

        if lang_sym_keys is None:
            lang_sym_keys = get_lang_sym_keys(args.lang_type)


        self.lang_sym_keys = lang_sym_keys
        self.mse = torch.nn.MSELoss(reduction="none")
        self.xent = torch.nn.CrossEntropyLoss(reduction="none")
        self.transform = transform

        # Saving outputs
        input_dim = len(self.enc_vocab)
        output_dim = len(self.dec_vocab)

        self.num_seg_classes = num_seg_classes

        # Save it, if it was passed in, to make it easy to load this model later
        enc_vocab.save_vocabulary(os.path.join(args.model_path, 'enc_vocab.json'))
        dec_vocab.save_vocabulary(os.path.join(args.model_path, 'dec_vocab.json'))
        if dec_verb_vocab is not None:
            dec_verb_vocab.save_vocabulary(os.path.join(args.model_path, 'dec_verb_vocab.json'))
        if dec_object_vocab is not None:
            dec_object_vocab.save_vocabulary(os.path.join(args.model_path, 'dec_object_vocab.json'))

        input_dim = len(enc_vocab)
        output_dim = len(dec_vocab)
        output_verb_dim = len(dec_verb_vocab)
        output_object_dim = len(dec_object_vocab)

        # build models
        enc = EncoderRNN(input_dim, self.enc_emb_dim, self.enc_hid_dim,
                         self.dec_hid_dim, self.enc_dropout,
                         self.args.bidirectional)
        attn = Attention(self.enc_hid_dim, self.dec_hid_dim,
                         self.args.bidirectional)
        if args.model == 'maskpred':
            img_unet = UNet(n_channels=3, n_classes=output_object_dim)
            unet_latent_dim = 128 # unet latent dim is fixed for now
            dec = DecoderMaskRNN(output_verb_dim, output_object_dim, self.dec_emb_dim,
                                 self.enc_hid_dim, self.dec_hid_dim, self.dec_dropout,
                                 attn, unet_latent_dim, self.args.bidirectional)
        elif args.model == 'maskselect':
            img_unet_enc = UNetEncoder(n_channels=4)
            # input classes 4 as RGB(3)+mask(1)
            unet_latent_dim = 512 # unet encoder latent dim is fixed for now
            dec = DecoderMaskRNN(output_verb_dim, output_object_dim, self.dec_emb_dim,
                                 self.enc_hid_dim, self.dec_hid_dim, self.dec_dropout,
                                 attn, unet_latent_dim, self.args.bidirectional)
        else:
            img_enc = EncoderCNN(self.enc_hid_dim)
            dec = DecoderRNN(output_dim, self.dec_emb_dim, self.enc_hid_dim,
                             self.dec_hid_dim, self.dec_dropout, attn,
                             self.args.bidirectional)

        # Holds a tensor for us to use
        self.onehot = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_model = False
        self.mask_select_model = False
        if self.args.model == "seq2seq":
            self.model = Seq2Seq(enc, dec, img_enc, self.device,
                                 self.args.img_input).to(self.device)
            self.query_actions = False
            self.mask_model = False
        elif self.args.model == "planner":
            if not args.img_input:
                raise RuntimeError('planner models all expect image input')
            T = TransitionRNN(output_dim, self.dec_emb_dim, self.enc_hid_dim,
                              self.dec_hid_dim)
            self.model = PlanningDomainModel(enc, dec, T, img_enc,
                    output_dim, self.enc_hid_dim, self.device).to(self.device)
            self.query_actions = True
            self.mask_model = False
        elif self.args.model == 'maskpred':
            if not args.img_input:
                raise RuntimeError('mask models all expect image input')
            self.model = MaskSeq2Seq(enc, dec, img_unet, 224, self.device).to(self.device)
            self.query_actions = False
            self.mask_model = True
        elif self.args.model == 'maskselect':
            if not args.img_input:
                raise RuntimeError('mask models all expect image input')
            self.model = MaskSelectSeq2Seq(enc, dec, img_unet_enc, 224, self.num_seg_classes, self.device).to(self.device)
            self.query_actions = False
            self.mask_model = False
            self.mask_select_model = True
            # softmax for mask selection
            self.softmax = nn.Softmax(2)
        else:
            raise RuntimeError('Model type ' + str(self.args.model) + ' not recognized or'
                               ' supported.')

    def _get_action_params(self, batch, key):
        """ Compute the labels for each of the different actions """
        assert key in ["good", "bad", "obs"]
        verb_key = 'sym_verb_%s' % key
        to_key = 'sym_to_obj_%s' % key
        done_key = 'done' # % key

        # Convert them
        to_obj = batch[to_key].to(self.device)
        verb = batch[verb_key].to(self.device)
        if key in ["good", "obs"]:
            lbl = torch.ones_like(verb).type(torch.FloatTensor).to(self.device)
        else:
            lbl = torch.zeros_like(verb).type(torch.FloatTensor).to(self.device)
        done = batch[done_key].to(self.device)
        # If we take a wrong action, then the distance to goal will go up
        #if key == "good" or key == "bad":
        #    #done = done + 1
        #    #done = done / done.shape[-1]
        #    done = torch.ones_like(done)
        #else:
        #    done = done / done.shape[-1]
        #done = torch.ones_like(done)
        done = torch.zeros_like(done)
        mask = (verb > 0).type(torch.FloatTensor).to(self.device)
        #print("verb", verb[0])
        #print("obj", to_obj[0])
        #print("mask", mask[0])
        #print("done", done[0])
        return lbl, verb, to_obj, mask, done

    def _get_action_loss(self, pred, target, mask):
        _, length, dim = pred.shape
        pred = pred.view(-1, dim)
        mask = mask.view(-1)
        target = target.view(-1)
        return torch.sum(self.xent(pred, target) * mask) / torch.sum(mask)

    def _get_planner_losses(self, batch, goal, h, obs_params):
        """ Compute the loss terms specifically for (a) preconditions or
        executability, and (b) progress metrics for task planning.
        
        Params:
        - batch: loaded batch of samples from data loader
        - h: hidden states corresponding to "observed" sequence
        """
        obs_lbl, obs_verb, obs_obj, obs_mask, obs_done = obs_params

        # ---------------
        #if self.onehot is None:
        batch_size, length = obs_verb.shape
        dim = self.model.action_dim
        self.onehot = torch.FloatTensor(batch_size, length, dim).to(self.device)
        self.onehot.zero_()
        self.onehot.scatter_(2, obs_verb[:,:,None], 1)
        # ---------------

        ok, val, verb, obj = self.model.query_actions(goal, h, obs_verb, obs_obj, verbs_1hot=self.onehot)

        # Compute transition loss
        obs_loss = F.binary_cross_entropy(ok, obs_lbl, reduction='none')
        transition_loss = torch.sum(obs_loss * obs_mask) / torch.sum(obs_mask)

        # Compute value loss
        final_mask = batch['done_mask'].to(self.device)
        final_mask_f = final_mask.type(torch.FloatTensor).to(self.device)
        done_err = self.mse(val, final_mask_f) #obs_done)
        value_loss = torch.sum(done_err * obs_mask) / torch.sum(obs_mask)

        # Over-weight the final state
        value_loss += torch.sum(done_err * final_mask) / torch.sum(final_mask)
        #print(done_err[0] * obs_mask[0])
        #print(val[0] * final_mask[0])

        # Action loss - predict verb + object token
        _verb_loss = self._get_action_loss(verb, obs_verb, obs_mask)
        _obj_loss = self._get_action_loss(obj, obs_obj, obs_mask)
        action_loss = _verb_loss + _obj_loss

        # Loop over extra actions that weren't observed
        for key in ["good", "bad"]:

            lbl, verb, obj, mask, done = self._get_action_params(batch, key)

            # Compute transition loss on good/bad actions
            # ok, val, verb, obj = self.model.query_actions(goal, h, verb, obj)
            ok, val, verb, obj = self.model.query_actions(goal, h, verb, obj, verbs_1hot=self.onehot)
            _loss = F.binary_cross_entropy(ok, lbl, reduction='none')
            _loss = torch.sum(mask * _loss) / torch.sum(mask)
            transition_loss = transition_loss + _loss

            # Compute value loss for possible/impossible states
            _loss = torch.sum(self.mse(val, done) * mask) / torch.sum(mask)
            value_loss = value_loss + _loss
        return transition_loss, value_loss, action_loss

    # training
    def do_train_epoch(self, dataloader, optimizer, criterion, mask_criterion, clip=1.0):
        """ Code for running a single training loop on the given data loader. """
        
        self.train()
        epoch_loss = 0
        epoch_transition_loss = 0
        epoch_value_loss = 0
        epoch_action_loss = 0
        epoch_verb_loss = 0
        epoch_object_loss = 0
        epoch_mask_loss = 0

        for i, batch in enumerate(tqdm(dataloader, ncols=60)):
            
            # move both inputs and labels to gpu if available
            images = batch['rgb'] # [batch, length, channel, height, width]
            source_sent = batch[self.lang_sym_keys[0]].to(self.device)
            target_sent = batch[self.lang_sym_keys[1]].to(self.device)

            # forward pass
            if self.query_actions:
                # Get the observed set of actions and associated predictions over time
                obs_params = self._get_action_params(batch, "obs")
                obs_lbl, obs_verb, obs_obj, obs_mask, obs_done = obs_params
                images = images[:, 0, :, :, :].to(self.device) # pick the first image
                # Run the predictions given "good" observed actions
                outputs, goal, h = self.model(source_sent.T, images, target_sent.T,
                                              verbs=obs_verb, objs=obs_obj,
                                              max_len=args.max_length)
                h = h.transpose(0,1).contiguous()
                _losses = self._get_planner_losses(batch, goal, h, obs_params)
                transition_loss, value_loss, action_loss = _losses
                verb_loss, object_loss, mask_loss = 0., 0., 0.

                #print("overall loss =", transition_loss, "value =", value_loss)
            elif self.mask_model:
                images = images.to(self.device)
                target_verb = batch['sym_plan_verbs'].to(self.device)
                target_object = batch['sym_plan_objects'].to(self.device)
                target_mask = batch['segmentation'].to(self.device)

                output_verbs, output_objects, output_mask = \
                    self.model(source_sent.T, target_verb.T, target_object.T,
                               images, max_plan_len=args.max_plan_length)
                transition_loss = 0.
                value_loss = 0.
                action_loss = 0.
                verb_loss = criterion(output_verbs.view(-1, output_verbs.shape[-1]), target_verb.T.flatten())
                object_loss = criterion(output_objects.view(-1, output_objects.shape[-1]), target_object.T.flatten())
                mask_loss = mask_criterion(output_mask.transpose(0, 1), target_mask.squeeze(2))
                # empty unused gpu memory
                torch.cuda.empty_cache()
            elif self.mask_select_model:
                images = images.to(self.device)
                target_verb = batch['sym_plan_verbs'].to(self.device)
                target_object = batch['sym_plan_objects'].to(self.device)
                # ground truth mask becomes an input in this case
                input_mask = batch['segmentation'].to(self.device)
                target_mask_class = batch['segmentation_classes'].to(self.device)

                output_verbs, output_objects, output_mask_class = \
                    self.model(source_sent.T, target_verb.T, target_object.T,
                               images, input_mask, max_plan_len=args.max_plan_length)
                transition_loss = 0.
                value_loss = 0.
                action_loss = 0.
                verb_loss = criterion(output_verbs.view(-1, output_verbs.shape[-1]), target_verb.T.flatten())
                object_loss = criterion(output_objects.view(-1, output_objects.shape[-1]), target_object.T.flatten())
                mask_loss = criterion(output_mask_class.view(-1, output_mask_class.shape[-1]), target_mask_class.squeeze(2).squeeze(2).flatten())
                # empty unused gpu memory
                torch.cuda.empty_cache()
            else:
                images = images[:, 0, :, :, :].to(self.device) # pick the first image
                outputs = self.model(source_sent.T, target_sent.T, images, max_len=args.max_length)
                transition_loss = 0.
                value_loss = 0.
                action_loss = 0.
                verb_loss, object_loss, mask_loss = 0., 0., 0.

            if self.mask_model or self.mask_select_model:
                loss = verb_loss * self.verb_wt
                loss += object_loss * self.object_wt
                loss += mask_loss * self.mask_wt
            else:
                output_dim = outputs.shape[-1]
                outputs = outputs.view(-1, output_dim)
                # compute loss
                loss = criterion(outputs, target_sent.T.flatten())
                if self.query_actions:
                    loss += transition_loss * self.transition_wt
                    loss += value_loss * self.value_wt
                    loss += action_loss * self.action_wt

            # zero all the gradients of the tensors optimizer will update
            optimizer.zero_grad()

            # backward pass + update parameters
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_transition_loss += float(transition_loss)
            epoch_value_loss += float(value_loss)
            epoch_action_loss += float(action_loss)
            epoch_verb_loss += float(verb_loss)
            epoch_object_loss += float(object_loss)
            epoch_mask_loss += float(mask_loss)

            # statistics
            if args.print_freq > 0 and i % args.print_freq == 0:
                if self.query_actions:
                    print(f'Step: {i}/{len(dataloader)}  Loss: {loss.item():.5f} T: {transition_loss.item():.5f}')
                else:
                    print(f'Step: {i}/{len(dataloader)}  Loss: {loss.item():.5f}')
        losses = [epoch_loss, epoch_transition_loss, epoch_value_loss, epoch_action_loss, epoch_verb_loss, epoch_object_loss, epoch_mask_loss]
        return [l / len(dataloader) for l in losses]

    # validation
    def do_val_epoch(self, dataloader, criterion, mask_criterion):
        
        self.eval()
        
        epoch_loss = 0
        epoch_transition_loss = 0
        epoch_value_loss = 0
        epoch_action_loss = 0
        epoch_verb_loss = 0
        epoch_object_loss = 0
        epoch_mask_loss = 0
        
        with torch.no_grad():
        
            for i, batch in enumerate(tqdm(dataloader, ncols=60)):

                images = batch['rgb']
                source_sent = batch[self.lang_sym_keys[0]].to(self.device)
                target_sent = batch[self.lang_sym_keys[1]].to(self.device)

                # forward pass
                if self.query_actions:
                    # Get the observed set of actions and associated predictions over time
                    obs_params = self._get_action_params(batch, "obs")
                    obs_lbl, obs_verb, obs_obj, obs_mask, obs_done = obs_params
                    images = images[:, 0, :, :, :].to(self.device) # pick the first image
                    # Run the predictions given "good" observed actions
                    outputs, goal, h = self.model(source_sent.T, images, target_sent.T,
                                                  verbs=obs_verb, objs=obs_obj,
                                                  max_len=args.max_length)
                    h = h.transpose(0,1).contiguous()
                    _losses = self._get_planner_losses(batch, goal, h, obs_params)
                    transition_loss, value_loss, action_loss = _losses
                    verb_loss, object_loss, mask_loss = 0., 0., 0.
                elif self.mask_model:
                    images = images.to(self.device)
                    target_verb = batch['sym_plan_verbs'].to(self.device)
                    target_object = batch['sym_plan_objects'].to(self.device)
                    target_mask = batch['segmentation'].to(self.device)

                    output_verbs, output_objects, output_mask = \
                        self.model(source_sent.T, target_verb.T, target_object.T,
                                   images, max_plan_len=args.max_plan_length)
                    transition_loss = 0.
                    value_loss = 0.
                    action_loss = 0.
                    verb_loss = criterion(output_verbs.view(-1, output_verbs.shape[-1]), target_verb.T.flatten())
                    object_loss = criterion(output_objects.view(-1, output_objects.shape[-1]), target_object.T.flatten())
                    mask_loss = mask_criterion(output_mask.transpose(0, 1), target_mask.squeeze(2))

                    # empty unused gpu memory
                    torch.cuda.empty_cache()
                elif self.mask_select_model:
                    images = images.to(self.device)
                    target_verb = batch['sym_plan_verbs'].to(self.device)
                    target_object = batch['sym_plan_objects'].to(self.device)
                    # ground truth mask becomes an input in this case
                    input_mask = batch['segmentation'].to(self.device)
                    target_mask_class = batch['segmentation_classes'].to(self.device)

                    output_verbs, output_objects, output_mask_class = \
                        self.model(source_sent.T, target_verb.T, target_object.T,
                                   images, input_mask, max_plan_len=args.max_plan_length)
                    transition_loss = 0.
                    value_loss = 0.
                    action_loss = 0.
                    verb_loss = criterion(output_verbs.view(-1, output_verbs.shape[-1]), target_verb.T.flatten())
                    object_loss = criterion(output_objects.view(-1, output_objects.shape[-1]), target_object.T.flatten())
                    mask_loss = criterion(output_mask_class.view(-1, output_mask_class.shape[-1]), target_mask_class.squeeze(2).squeeze(2).flatten())
                    # empty unused gpu memory
                    torch.cuda.empty_cache()
                else:
                    images = images[:, 0, :, :, :].to(self.device) # pick the first image
                    outputs = self.model(source_sent.T, target_sent.T, images, 0) #turn off teacher forcing
                    transition_loss, value_loss, action_loss = 0., 0., 0.
                    verb_loss, object_loss, mask_loss = 0., 0., 0.

                if self.mask_model or self.mask_select_model:
                    loss = verb_loss * self.verb_wt
                    loss += object_loss * self.object_wt
                    loss += mask_loss * self.mask_wt
                else:

                    output_dim = outputs.shape[-1]
                    outputs = outputs.view(-1, output_dim)

                    loss = criterion(outputs,  target_sent.T.flatten())
                    loss += (self.transition_wt * transition_loss) + (self.value_wt * value_loss)
                    loss += action_loss * self.action_wt

                epoch_loss += loss.item()
                epoch_transition_loss += float(transition_loss)
                epoch_value_loss += float(value_loss)
                epoch_action_loss += float(action_loss)
                epoch_verb_loss += float(verb_loss)
                epoch_object_loss += float(object_loss)
                epoch_mask_loss += float(mask_loss)

        losses = [epoch_loss, epoch_transition_loss, epoch_value_loss, epoch_action_loss, epoch_verb_loss, epoch_object_loss, epoch_mask_loss]
        return [l / len(dataloader) for l in losses]

    def run_training(self, train_dataloader, val_dataloader):

        self.model.apply(init_weights)
        print(f'The model has {count_parameters(self.model):,} trainable parameters')

        # loss function and optimizer
        optimizer = optim.AdamW(model.parameters(), self.learning_rate)
        pad_idx = dec_vocab['<NONE>']
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='mean')
        mask_criterion = nn.BCEWithLogitsLoss()

        # Epoch loop
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            
            start_time = time.time()
            train_losses = self.do_train_epoch(train_dataloader, optimizer, criterion, mask_criterion)
            valid_losses = self.do_val_epoch(val_dataloader, criterion, mask_criterion)

            # Extract losses now that we are returning more information
            train_loss, train_transition_loss, train_value_loss, train_action_loss, train_verb_loss, train_object_loss, train_mask_loss = train_losses
            val_loss, val_transition_loss, val_value_loss, val_action_loss, val_verb_loss, val_object_loss, val_mask_loss = valid_losses

            end_time = time.time()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, 'lang2sym_bestmodel.pt'))
            
            print(f'Epoch: {epoch+1:02} | Time: {(end_time - start_time):.2f}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t\ttransition loss = {train_transition_loss:.3f}')
            print(f'\t\tvalue loss = {train_value_loss:.3f}')
            print(f'\t\taction loss = {train_action_loss:.3f}')
            print(f'\t\tverb loss = {train_verb_loss:.3f}')
            print(f'\t\tobject loss = {train_object_loss:.3f}')
            print(f'\t\tmask loss = {train_mask_loss:.3f}')
            try: math.exp(val_loss)
            except: val_loss = float('inf')
            print(f'\t Val Loss: {val_loss:.3f} |  Val PPL: {math.exp(val_loss):7.3f}')
            print(f'\t\ttransition loss = {val_transition_loss:.3f}')
            print(f'\t\tvalue loss = {val_value_loss:.3f}')
            print(f'\t\taction loss = {val_action_loss:.3f}')
            print(f'\t\tverb loss = {val_verb_loss:.3f}')
            print(f'\t\tobject loss = {val_object_loss:.3f}')
            print(f'\t\tmask loss = {val_mask_loss:.3f}')


    def output_to_words(self, outputs):
        pred_token = outputs.squeeze().argmax(-1).tolist()
        pred_words = [self.dec_vocab.from_idx(x) for x in pred_token]

        # collect sentences and truncate at end of sentence token
        try:
            end_idx = pred_words.index('<END>')
        except:
            end_idx = self.args.max_length
        # return predictions as a sentence
        return ' '.join(pred_words[1:end_idx])

    def input_to_words(self, inputs):
        pred_token = inputs
        if not isinstance(pred_token, list):
            pred_token = inputs.tolist()
        print(pred_token)
        pred_words = [self.enc_vocab.from_idx(x[0]) for x in pred_token]
        return ' '.join(pred_words)

    def run_test(self, test_dataloader, verbose=True):
        """
        Run test code. Predict symbols and see if they match.
        """

        self.model.load_state_dict(torch.load(os.path.join(args.model_path, 'lang2sym_bestmodel.pt')))

        self.model.eval()
        pred_sent = []
        gt_sent = []
        inp_sent = []
        filenames = []
        pred_verbs = []
        pred_objects = []
        gt_verbs = []
        gt_objects = []

        for i, batch in enumerate(tqdm(test_dataloader, ncols=60)):
            
            # move both inputs and labels to gpu if available
            images = batch['rgb']
            source_sent = batch[self.lang_sym_keys[0]].to(self.device)
            target_sent = batch[self.lang_sym_keys[1]].to(self.device)
            filename = batch['filename']

            # forward pass
            if self.query_actions:
                obs_lbl, obs_verb, obs_obj, obs_mask, obs_done = self._get_action_params(batch, "obs")
                images = images[:, 0, :, :, :].to(self.device) # pick the first image
                outputs, goal, h = self.model(source_sent.T, images, source_sent.T, # target_sent.T,
                                              verbs=obs_verb, objs=obs_obj,
                                              teacher_forcing_ratio=0., mode='test',
                                              max_len=args.max_length)

                # Check to see if all actions were labeled executable
                h = h.transpose(0,1).contiguous()
                state = h[:,0]
                if verbose:
                    print('-----------')
                for j in range(self.args.max_plan_length):
                    done, verb, obj = self.model.eval_state(state, goal)
                    done = float(done) > 0.5
                    pred_v, pred_o = int(verb[0]), int(obj[0])
                    gt_v, gt_o = int(obs_verb[0,j]), int(obs_obj[0, j])
                    if verbose:
                        print(j, done,
                              'VERB:', pred_v, self.dec_vocab.to_word(pred_v),
                              'vs.', gt_v, self.dec_vocab.to_word(gt_v),
                              'OBJ:',  pred_o, self.dec_vocab.to_word(pred_o),
                              'vs.', gt_o, self.dec_vocab.to_word(gt_o))
                    if done:
                        break
                    else:
                        state = self.model.predict(state, verb, obj)
            elif self.mask_model:
                images = images.to(self.device)
                target_verb = batch['sym_plan_verbs'].to(self.device)
                target_object = batch['sym_plan_objects'].to(self.device)
                target_mask = batch['segmentation'].to(self.device)

                output_verbs, output_objects, output_mask = \
                    self.model(source_sent.T, target_verb.T, target_object.T,
                               images, max_plan_len=args.max_plan_length,
                               teacher_forcing_ratio=0., mode='test')
                pred_verbs_tokens = output_verbs.argmax(2).tolist()
                pred_objects_tokens = output_objects.argmax(2).tolist()
                pred_verb_words = [self.dec_verb_vocab.from_idx(x[0]) for x in pred_verbs_tokens]
                pred_object_words = [self.dec_object_vocab.from_idx(x[0]) for x in pred_objects_tokens]

                gt_verb_words = [self.dec_verb_vocab.from_idx(x) for x in target_verb.tolist()[0]]
                gt_object_words = [self.dec_object_vocab.from_idx(x) for x in target_object.tolist()[0]]
                inp_words = [self.enc_vocab.from_idx(x) for x in source_sent.tolist()[0]]

                # collect sentences and truncate at end of sentence token
                try:
                    end_idx = min(pred_verb_words.index('<END>'), pred_object_words.index('<END>'))
                except:
                    end_idx = self.args.max_plan_length

                pred_verbs.append(pred_verb_words[1:end_idx])
                pred_objects.append(pred_object_words[1:end_idx])
                gt_verbs.append(gt_verb_words[1:gt_verb_words.index('<END>')])
                gt_objects.append(gt_object_words[1:gt_object_words.index('<END>')])
                inp_sent.append(inp_words[1:inp_words.index('<END>')])
                filenames.append(filename)

                if verbose:
                    # save images, gt and predicted masks
                    if not os.path.exists(os.path.join(args.model_path, 'test_maskpred')):
                        os.makedirs(os.path.join(args.model_path, 'test_maskpred'))
                    fig, ax = plt.subplots(nrows=3, ncols=end_idx, figsize=(15, 15))
                    logger.setLevel(100)
                    for col in range(end_idx-1):
                        ax[0, col].imshow(np.moveaxis(images.cpu().detach().numpy()[0, col, :, :, :], [0, 1], [-1, 0]))
                        ax[0, col].axis('off')
                        ax[1, col].imshow(target_mask.cpu().detach().numpy()[0, col, 0, :, :].astype('float32'), cmap=plt.get_cmap('gray'))
                        ax[1, col].axis('off')
                        ax[2, col].imshow(output_mask.cpu().detach().numpy()[col, 0, :, :].astype('float32'), cmap=plt.get_cmap('gray'))
                        ax[2, col].axis('off')
                    ax[0, end_idx-1].set_axis_off()
                    ax[1, end_idx-1].set_axis_off()
                    ax[2, end_idx-1].set_axis_off()
                    fig.subplots_adjust(wspace=0, hspace=0)
                    plt.savefig(os.path.join(args.model_path, 'test_maskpred', str(i)+'.png'))
                    plt.clf()
                    logger.setLevel(old_level)
            elif self.mask_select_model:
                images = images.to(self.device)
                target_verb = batch['sym_plan_verbs'].to(self.device)
                target_object = batch['sym_plan_objects'].to(self.device)
                # ground truth mask becomes an input in this case
                input_mask = batch['segmentation'].to(self.device)
                # TODO correctly obtain correct class of mask
                target_mask_class = batch['segmentation_classes'].to(self.device)

                output_verbs, output_objects, output_mask_class = \
                    self.model(source_sent.T, target_verb.T, target_object.T,
                               images, input_mask, max_plan_len=args.max_plan_length)
                pred_verbs_tokens = output_verbs.argmax(2).tolist()
                pred_objects_tokens = output_objects.argmax(2).tolist()
                pred_verb_words = [self.dec_verb_vocab.from_idx(x[0]) for x in pred_verbs_tokens]
                pred_object_words = [self.dec_object_vocab.from_idx(x[0]) for x in pred_objects_tokens]

                gt_verb_words = [self.dec_verb_vocab.from_idx(x) for x in target_verb.tolist()[0]]
                gt_object_words = [self.dec_object_vocab.from_idx(x) for x in target_object.tolist()[0]]
                inp_words = [self.enc_vocab.from_idx(x) for x in source_sent.tolist()[0]]

                # collect sentences and truncate at end of sentence token
                try:
                    end_idx = min(pred_verb_words.index('<END>'), pred_object_words.index('<END>'))
                except:
                    end_idx = self.args.max_plan_length

                pred_verbs.append(pred_verb_words[1:end_idx])
                pred_objects.append(pred_object_words[1:end_idx])
                gt_verbs.append(gt_verb_words[1:gt_verb_words.index('<END>')])
                gt_objects.append(gt_object_words[1:gt_object_words.index('<END>')])
                inp_sent.append(inp_words[1:inp_words.index('<END>')])
                filenames.append(filename)
                if verbose:
                    # save images, gt and selected masks
                    if not os.path.exists(os.path.join(args.model_path, 'test_maskselect')):
                        os.makedirs(os.path.join(args.model_path, 'test_maskselect'))
                    fig, ax = plt.subplots(nrows=3, ncols=end_idx, figsize=(15, 15))
                    logger.setLevel(100)
                    fig.suptitle('filename: '+ filename[0]+ '\n'+
                        'input_sent: '+ ' '.join(inp_words[1:inp_words.index('<END>')])+ '\n'+
                        'gt_verbs: '+ ' '.join(gt_verb_words[1:gt_verb_words.index('<END>')])+ '\n'
                        'gt_objects: '+ ' '.join(gt_object_words[1:gt_object_words.index('<END>')])+ '\n'+
                        'pred_verbs: '+ ' '.join(pred_verb_words[1:end_idx])+ '\n'+
                        'pred_objects: '+ ' '.join(pred_object_words[1:end_idx]))

                    # apply softmax and obtain selected mask classes
                    selected_masks = self.softmax(output_mask_class).argmax(2)

                    for col in range(end_idx-1):
                        ax[0, col].imshow(np.moveaxis(images.cpu().detach().numpy()[0, col, :, :, :], [0, 1], [-1, 0]))
                        ax[0, col].axis('off')
                        ax[1, col].imshow(input_mask.cpu().detach().numpy()[0, col, 0, :, :], cmap=plt.get_cmap('gray'))
                        ax[1, col].axis('off')

                        selected_class = selected_masks[col]
                        full_mask = input_mask.cpu().detach().numpy()[0, col, 0, :, :].astype('float32')
                        mask_id = [x for x in blocks_seg_classes.keys() if blocks_seg_classes[x] == selected_class]
                        selected_mask = ((full_mask*255.0) == mask_id[0]).astype('float32')

                        ax[2, col].imshow(selected_mask, cmap=plt.get_cmap('gray'))
                        ax[2, col].axis('off')
                    ax[0, end_idx-1].set_axis_off()
                    ax[1, end_idx-1].set_axis_off()
                    ax[2, end_idx-1].set_axis_off()
                    fig.subplots_adjust(wspace=0, hspace=0)
                    plt.savefig(os.path.join(args.model_path, 'test_maskselect', str(i)+'.png'))
                    plt.clf()
                    logger.setLevel(old_level)
            else:
                images = images[:, 0, :, :, :].to(self.device) # pick the first image
                outputs = self.model(source_sent.T, target_sent.T, images,
                                     teacher_forcing_ratio=1.0, mode='test',
                                     max_len=args.max_length)

            if not (self.mask_model or self.mask_select_model):
                pred_token = outputs.argmax(2).tolist()
                
                pred_words = [self.dec_vocab.from_idx(x[0]) for x in pred_token]
                gt_words = [self.dec_vocab.from_idx(x) for x in target_sent.tolist()[0]]
                inp_words = [self.enc_vocab.from_idx(x) for x in source_sent.tolist()[0]]

                # collect sentences and truncate at end of sentence token
                try:
                    end_idx = pred_words.index('<END>')
                except:
                    end_idx = self.args.max_length

                pred_sent.append(pred_words[1:end_idx])
                gt_sent.append(gt_words[1:gt_words.index('<END>')])
                inp_sent.append(inp_words[1:inp_words.index('<END>')])
                filenames.append(filename)
        

        # compute language metric and save language data
        if not (self.mask_model or self.mask_select_model):
            # compute Bleu metric on predictions
            print("BLEU score:", bleu_score(pred_sent, [[x] for x in gt_sent]))

            # dump input, gt and predictions to json
            all_data = [{'input_sent':' '.join(inp_sent[x]),
                         'gt_sent':' '.join(gt_sent[x]),
                         'pred_sent':' '.join(pred_sent[x]),
                         'filename': filenames[x]}
                        for x in range(len(inp_sent))]
            with open(os.path.join(args.model_path, 'test_predictions_bestmodel.json'), 'w') as f:
                json.dump(all_data, f, indent=4)
        else:
            print("Verbs BLEU score:", bleu_score(pred_verbs, [[x] for x in gt_verbs]))
            print("Objects BLEU score:", bleu_score(pred_objects, [[x] for x in gt_objects]))
            # dump input, gt and predictions to json
            all_data = [{'input_sent':' '.join(inp_sent[x]),
                         'gt_verbs':' '.join(gt_verbs[x]),
                         'gt_objects':' '.join(gt_objects[x]),
                         'pred_verbs':' '.join(pred_verbs[x]),
                         'pred_objects':' '.join(pred_objects[x]),
                         'filename': filenames[x]}
                        for x in range(len(inp_sent))]
            with open(os.path.join(args.model_path, 'test_predictions_bestmodel.json'), 'w') as f:
                json.dump(all_data, f, indent=4)

    def set_transform(self, transform):
        self.transform = transform

    def infer(self, rgb, sent, debug=True):
        """ For test time - take in image and sentence; embed sentence and get data. """
        if len(rgb.shape) != 3:
            raise RuntimeError('rgb images were wrong shape; need to be W x H x C')
        print("PREDICT FOR SENTENCE:", sent)
        sent = self.enc_vocab.embed_sentence(sent, length=self.args.max_length)
        sent = sent.unsqueeze(1)
        print("<-- SENTENCE:", self.input_to_words(sent))
        print("Image size:", rgb.shape)
        rgb = resize_image(rgb)
        print("(Corrected) image size:", rgb.shape)
        rgb = self.transform(rgb[:,:,:3]).unsqueeze(0)
        if debug:
            rgb2 = np.rollaxis(rgb[0].cpu().numpy(), 0, 3)
            plt.figure()
            plt.imshow(rgb2)
            plt.show()

        # move both inputs and labels to gpu if available
        rgb = rgb.to(self.device)
        source_sent = sent.to(self.device)
        # target sent not used in test mode so passing input as dummy
        target_sent = sent.to(self.device)

        # forward pass
        if self.query_actions:
            outputs, goal, h = self.model(source_sent, rgb, target_sent,
                                          teacher_forcing_ratio=0.,
                                          mode='test',
                                          max_len=self.args.max_length)
        else:
            outputs = self.model(source_sent, target_sent, rgb,
                                      teacher_forcing_ratio=0., mode='test',
                                      max_len=self.args.max_length)
        res = self.output_to_words(outputs)
        print("OUTPUT:", res)
        input('<<< ENTER >>>')
        return res


def get_lang_sym_keys(lang_type):
    if lang_type == 'goal2goal':
        lang_sym_keys = ('lang_goal', 'sym_goal')
    elif lang_type == 'plan2plan':
        lang_sym_keys = ('lang_plan', 'sym_plan')
    elif lang_type == 'human2goal':
        lang_sym_keys = ('lang_description', 'sym_goal')
    elif lang_type == 'human2plan':
        lang_sym_keys = ('lang_description', 'sym_plan')
    elif lang_type == 'goallang2plan':
        lang_sym_keys = ('lang_goal', 'sym_plan')
    elif lang_type == 'goalsym2plan':
        lang_sym_keys = ('sym_goal', 'sym_plan')
    else:
        raise ValueError('lang_type not recognized.')
    return lang_sym_keys


def get_lang2sym(args):
    """ Get a lang to sym testing/training object """

    enc_vocab = Vocabulary(os.path.join(args.model_path, 'enc_vocab.json'))
    dec_vocab = Vocabulary(os.path.join(args.model_path, 'dec_vocab.json'))
    dec_verb_vocab = Vocabulary(os.path.join(args.model_path, 'dec_verb_vocab.json'))
    dec_object_vocab = Vocabulary(os.path.join(args.model_path, 'dec_object_vocab.json'))
    lang_sym_keys = get_lang_sym_keys(args.lang_type)
    evaluator = LangToSym(args, enc_vocab, dec_vocab, dec_verb_vocab, dec_object_vocab, lang_sym_keys)
    evaluator.eval()
    evaluator.model.load_state_dict(torch.load(os.path.join(args.model_path, 'lang2sym_bestmodel.pt')))
    transform = get_rgb_transforms('test')
    evaluator.set_transform(transform)
    return evaluator


# only supports seq2seq for now
def infer_sample(args, rgb, sent):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_vocab = Vocabulary(os.path.join(args.model_path, 'enc_vocab.json'))
    dec_vocab = Vocabulary(os.path.join(args.model_path, 'dec_vocab.json'))
    dec_verb_vocab = Vocabulary(os.path.join(args.model_path, 'dec_verb_vocab.json'))
    dec_object_vocab = Vocabulary(os.path.join(args.model_path, 'dec_object_vocab.json'))

    # language type
    lang_sym_keys = get_lang_sym_keys(args.lang_type)

    evaluator = LangToSym(args, enc_vocab, dec_vocab, dec_verb_vocab, dec_object_vocab, lang_sym_keys)

    evaluator.model.load_state_dict(torch.load(os.path.join(args.model_path, 'lang2sym_bestmodel.pt')))
    evaluator.model.eval()

    # pre-process image and language to be same as that used in training
    transform = get_rgb_transforms('test')
    rgb = transform(rgb[:,:,:3]).unsqueeze(0)
    sent = enc_vocab.embed_sentence(sent, length=args.max_length)
    sent = sent.unsqueeze(1)

    # move both inputs and labels to gpu if available
    rgb = rgb.to(device)
    source_sent = sent.to(device)
    # target sent not used in test mode so passing input as dummy
    target_sent = sent.to(device)

    # forward pass
    outputs = evaluator.model(source_sent, target_sent, rgb,
                              teacher_forcing_ratio=1.0, mode='test',
                              max_len=args.max_length)

    pred_token = outputs.argmax(2).tolist()
    pred_words = [dec_vocab.from_idx(x[0]) for x in pred_token]

    # collect sentences and truncate at end of sentence token
    try:
        end_idx = pred_words.index('<END>')
    except:
        end_idx = args.max_length
    # return predictions as a sentence
    return ' '.join(pred_words[1:end_idx])


if __name__ == "__main__":

    args = parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # language type
    lang_sym_keys = get_lang_sym_keys(args.lang_type)

    if args.model == 'maskpred' or args.model == 'maskselect':
        assert lang_sym_keys[1] == 'sym_plan', "For mask, prediction should be plan symbols."

    # TODO- Current test mode only support batch size 1.
    if args.mode == 'test':
        args.batch_size = 1

    if args.generate_splits:
        # Create splits from h5 data
        print("Regenerating data splits...")
        split.split_h5_by_code(args.data, verbose=False)

    # prepare dataloaders
    print("Creating data loaders...")

    enc_vocab, dec_vocab, dec_verb_vocab, dec_object_vocab, train_dataloader, val_dataloader, test_dataloader, train_dataset, valid_dataset, test_dataset = \
        split.create_data_loaders(
            args.data,
            enc_vocab_path=os.path.join(args.model_path, 'enc_vocab.json'),
            dec_vocab_path=os.path.join(args.model_path, 'dec_vocab.json'),
            exp_type=args.exp_type,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.workers,
            max_plan_length=args.max_plan_length,
            model_type=args.model,
            dec_verb_vocab_path=os.path.join(args.model_path, 'dec_verb_vocab.json'),
            dec_object_vocab_path=os.path.join(args.model_path, 'dec_object_vocab.json'))

    model = LangToSym(args, enc_vocab, dec_vocab, dec_verb_vocab, dec_object_vocab, lang_sym_keys, num_seg_classes=train_dataset.num_seg_classes)
    if args.mode == 'train':
        if args.resume:
            model.model.load_state_dict(torch.load(os.path.join(args.model_path, 'lang2sym_bestmodel.pt')))
        model.run_training(train_dataloader, val_dataloader)
    if args.mode == 'test':
        model.run_test(test_dataloader)
