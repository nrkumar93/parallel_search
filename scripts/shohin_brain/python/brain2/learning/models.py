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


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EncoderCNN(nn.Module):
    def __init__(self, embed_size = 512):
        super(EncoderCNN, self).__init__()
        
        # get the pretrained encoder model
        self.encoder = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # replace the classifier with a fully connected embedding layer
        self.encoder.fc = nn.Linear(in_features=2048, out_features=1024)
        
        # add another fully connected layer
        self.embed = nn.Linear(in_features=1024, out_features=embed_size)
        
        # dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
        # activation layers
        self.prelu = nn.PReLU()
        
    def forward(self, images):
        
        # get the embeddings from the encoder
        encoder_outputs = self.dropout(self.prelu(self.encoder(images)))
        
        # pass through the fully connected
        embeddings = self.embed(encoder_outputs)
        
        return embeddings


class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, bidirectional=False):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=bidirectional)
        
        if self.bidirectional is True:
            self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        else:
            self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, img_embeddings):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        if img_embeddings is not None:
            if self.bidirectional is True:
                stacked_imgs_embeddings = torch.stack((img_embeddings, img_embeddings))
            else:
                stacked_imgs_embeddings = torch.unsqueeze(img_embeddings, 0)
            # Check to see if we do not have enough image input
            #if stacked_imgs_embeddings.shape[1] == 1:
            #    stacked_imgs_embeddings = stacked_imgs_embeddings.repeat(1, embedded.shape[1], 1)
            outputs, hidden = self.rnn(embedded, stacked_imgs_embeddings)
        else:
            outputs, hidden = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        if self.bidirectional is True:
            hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        else:
            hidden = torch.tanh(self.fc(hidden[-1,:,:]))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, bidirectional=False):
        super().__init__()

        self.bidirectional = bidirectional

        if self.bidirectional is True:
            self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        else:
            self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)


class TransitionRNN(nn.Module):
    """ Very simple model that should capture the progress of the planner over
    time in order to say which actions can be executed from each state. """
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = dec_hid_dim
        self.rnn = nn.GRU(dec_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, verb_emb, obj_emb):
        rnn_input = torch.cat([verb_emb, obj_emb], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        return output, hidden

class DecoderRNN(nn.Module):
    """ Decoder RNN that interprets the input to get a sequence of commands to
    follow. This uses attention back to language + prev hidden state."""
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, bidirectional=False):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.bidirectional = bidirectional
        self.hidden_dim = dec_hid_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        if self.bidirectional is True:
            self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
            self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        else:
            self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim)
            self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        
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
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output.shape == hidden.shape)
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, image_encoder, device, img_input=1):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.image_encoder = None
        if img_input == 1:
            self.image_encoder = image_encoder

    def forward(self, src, trg, imgs, teacher_forcing_ratio = 0.0, mode='train', max_len=128, end_token_index=2):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        if self.image_encoder is not None:
            imgs_embeddings = self.image_encoder(imgs)
        else:
            imgs_embeddings = None
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, imgs_embeddings)

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        if mode == 'test':
            trg_len = max_len
            # <START> token for first time step
            input = torch.ones((batch_size), dtype=torch.long).to(self.device)

        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
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
            input = trg[t] if teacher_force else top1

        return outputs
