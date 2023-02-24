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


class DecoderMaskRNN(nn.Module):
    """ Decoder RNN that interprets the input to get a sequence of commands to
    follow. This uses attention back to language + prev hidden state."""
    def __init__(self, output_dim_verb, output_dim_object, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, unet_dim, bidirectional=False):
        super().__init__()

        self.output_dim_verb = output_dim_verb
        self.output_dim_object = output_dim_object
        self.attention = attention
        self.bidirectional = bidirectional
        self.hidden_dim = dec_hid_dim
        
        self.embedding_verb = nn.Embedding(output_dim_verb, emb_dim)
        self.embedding_object = nn.Embedding(output_dim_object, emb_dim)

        self.embedding_unet = nn.Linear(unet_dim, emb_dim)
        
        if self.bidirectional is True:
            self.rnn = nn.GRU((enc_hid_dim * 2) + (emb_dim * 3), dec_hid_dim)
            self.fc_out_verb = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim_verb)
            self.fc_out_object = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim_object)
        else:
            self.rnn = nn.GRU(enc_hid_dim + (emb_dim * 3), dec_hid_dim)
            self.fc_out_verb = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim_verb)
            self.fc_out_object = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim_object)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_verb, input_object, hidden, encoder_outputs, latent_mask):
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        input_verb = input_verb.unsqueeze(0)
        input_object = input_object.unsqueeze(0)
        
        #input = [1, batch size]
        embedded_verb = self.dropout(self.embedding_verb(input_verb))
        embedded_object = self.dropout(self.embedding_object(input_object))

        image_embedding = self.dropout(self.embedding_unet(latent_mask.squeeze(3).squeeze(2))).unsqueeze(0)
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
          
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded_verb, embedded_object, weighted, image_embedding), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output.shape == hidden.shape)
        
        embedded_verb = embedded_verb.squeeze(0)
        embedded_object = embedded_object.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction_verb = self.fc_out_verb(torch.cat((output, weighted, embedded_verb), dim = 1))
        prediction_object = self.fc_out_object(torch.cat((output, weighted, embedded_object), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction_verb, prediction_object, hidden.squeeze(0)


class MaskSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, image_unet, img_dim, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.image_unet = image_unet
        self.img_dim = img_dim

        self.att_activation = nn.Sigmoid()

    def forward(self, src, trg_verb, trg_object, imgs, teacher_forcing_ratio=0.0, mode='train', max_plan_len=16, end_token_index=2):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        # imgs = [batch, length, channel, height, width]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]

        # assert length of target object and verbs match
        assert trg_verb.shape[0] == trg_object.shape[0]

        trg_len = trg_verb.shape[0]
        trg_verb_vocab_size = self.decoder.output_dim_verb
        trg_object_vocab_size = self.decoder.output_dim_object
        
        #tensor to store decoder outputs
        outputs_verbs = torch.zeros(trg_len, batch_size, trg_verb_vocab_size).to(self.device)
        outputs_objects = torch.zeros(trg_len, batch_size, trg_object_vocab_size).to(self.device)
        output_masks = torch.zeros(trg_len, batch_size, self.img_dim, self.img_dim).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, None)

        #first input to the decoder is the <sos> tokens
        input_verb = trg_verb[0, :]
        input_object = trg_object[0, :]
        input_img = imgs[:, 0, :, :, :]

        # TODO- during inference, a new input image should be input after each time step

        if mode == 'test':
            trg_len = max_plan_len
            # <START> token for first time step
            input_verb = torch.ones((batch_size), dtype=torch.long).to(self.device)
            input_object = torch.ones((batch_size), dtype=torch.long).to(self.device)

        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
    
            # run unet inference
            img_logits, latent_img = self.image_unet(input_img)

            output_verb, output_object, hidden = self.decoder(input_verb, input_object, hidden, encoder_outputs, latent_img)

            import pdb; pdb.set_trace()
            # multiplicative attention between unet logits and object prediction
            pred_mask = torch.sum(torch.mul(output_object.unsqueeze(2).unsqueeze(2), img_logits), axis=1)
            output_masks[t, :] = self.att_activation(pred_mask)
            
            #place predictions in a tensor holding predictions for each token
            outputs_verbs[t] = output_verb
            outputs_objects[t] = output_object
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1_verb = output_verb.argmax(1)
            top1_object = output_object.argmax(1)

            # in test mode, stop at <END> token
            if mode == 'test' and (top1_object.item() == end_token_index or top1_verb.item() == end_token_index):
                break
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input_verb = trg_verb[t] if not teacher_force else top1_verb
            input_object = trg_object[t] if not teacher_force else top1_object
            # TODO- during test, a new input image should be input after each time step
            input_img = imgs[:, t , :, :, :]

        # return three outputs for three losses
        return outputs_verbs, outputs_objects, output_masks



class MaskSelectSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, image_unet_enc, img_dim, num_seg_classes, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.image_unet_enc = image_unet_enc
        self.img_dim = img_dim
        self.num_seg_classes = num_seg_classes

        # first layer- hidden_dim+unet dim --> 256
        self.embedding_mask_selection_first = nn.Linear(self.decoder.hidden_dim + 512, 256)
        # second layer- 256 --> number of segmentation classes
        self.embedding_mask_selection_second = nn.Linear(256, self.num_seg_classes)

    def forward(self, src, trg_verb, trg_object, imgs, masks, teacher_forcing_ratio=0.0, mode='train', max_plan_len=16, end_token_index=2):

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        # imgs = [batch, length, channel, height, width]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]

        # assert length of target object and verbs match
        assert trg_verb.shape[0] == trg_object.shape[0]

        trg_len = trg_verb.shape[0]
        trg_verb_vocab_size = self.decoder.output_dim_verb
        trg_object_vocab_size = self.decoder.output_dim_object
        #tensor to store decoder outputs
        outputs_verbs = torch.zeros(trg_len, batch_size, trg_verb_vocab_size).to(self.device)
        outputs_objects = torch.zeros(trg_len, batch_size, trg_object_vocab_size).to(self.device)
        output_mask_classes = torch.zeros(trg_len, batch_size, self.num_seg_classes).to(self.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, None)

        #first input to the decoder is the <sos> tokens
        input_verb = trg_verb[0, :]
        input_object = trg_object[0, :]
        input_img = imgs[:, 0, :, :, :]
        input_mask = masks[:, 0, :, :, :]
        unet_input = torch.cat((input_img, input_mask), dim = 1)

        # TODO- during inference, a new input image should be input after each time step

        if mode == 'test':
            trg_len = max_plan_len
            # <START> token for first time step
            input_verb = torch.ones((batch_size), dtype=torch.long).to(self.device)
            input_object = torch.ones((batch_size), dtype=torch.long).to(self.device)

        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state

            # run unet inference
            latent_img = self.image_unet_enc(unet_input)

            output_verb, output_object, hidden = self.decoder(input_verb, input_object, hidden, encoder_outputs, latent_img)

            # for first layer- input is concat of decoder hidden and image_unet_enc output
            class_intermediate_logits = self.embedding_mask_selection_first(torch.cat((latent_img.squeeze(2).squeeze(2), hidden), 1))
            # for second layer- input is first layer ouptut
            mask_class_logits = self.embedding_mask_selection_second(class_intermediate_logits)

             # store class predictions
            output_mask_classes[t, :] = mask_class_logits

            #place predictions in a tensor holding predictions for each token
            outputs_verbs[t] = output_verb
            outputs_objects[t] = output_object

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1_verb = output_verb.argmax(1)
            top1_object = output_object.argmax(1)

            # in test mode, stop at <END> token
            if mode == 'test' and (top1_object.item() == end_token_index or top1_verb.item() == end_token_index):
                break

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input_verb = trg_verb[t] if not teacher_force else top1_verb
            input_object = trg_object[t] if not teacher_force else top1_object
            # TODO- during test, a new input image should be input after each time step
            input_img = imgs[:, t , :, :, :]
            input_mask = masks[:, t, :, :, :]
            unet_input = torch.cat((input_img, input_mask), dim = 1)

        # return three outputs for three losses
        return outputs_verbs, outputs_objects, output_mask_classes
