import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float, maxlen: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Positional Encoding Sinusoid Formula
        pos_encoding = torch.zeros(maxlen, d_model)
        pos = torch.arange(0, maxlen, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        denom = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)) / d_model) # 1000^(2i/d_model)        
        pos_encoding[:, 0::2] = torch.sin(pos * denom)
        pos_encoding[:, 1::2] = torch.cos(pos * denom)
        
        # saving encoingss without gradients
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, token_embedding):
        # Residual connection + positional encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):

    # Constructor
    def __init__(self, n_tokens: int, d_model: int=512, n_head: int=8, n_encoder: int=6, n_decoder: int=6, feed_forward: int=2048, dropout: float=0.1):         
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        # transformer layers
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoder(d_model=d_model, dropout=dropout, maxlen=5000)
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_head, num_encoder_layers=n_encoder, 
                                        num_decoder_layers=n_decoder, dim_feedforward=feed_forward, dropout=dropout)
        self.out = nn.Linear(d_model, n_tokens)
        
    def forward(self, src: torch.tensor, tgt: torch.tensor, tgt_mask: torch.tensor=None, src_pad_mask: torch.BoolTensor=None, tgt_pad_mask: torch.BoolTensor=None) -> torch.tensor:
        # src shape: (batch_size, src seq_length), tgt shape: (batch_size, tgt seq_length)

        # embedding + positional encoding shape: (batch_size, sequence length, d_model)
        src = self.embedding(src) * np.sqrt(self.d_model)
        tgt = self.embedding(tgt) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # reshape: (sequence length, batch_size, d_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # transformer inputs shape: (sequence length, batch_size, n_tokens)
        decoder_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)

        # compute linear function and apply softmax (Cross Entropy Loss already does this just return n_tokens as out)
        out = self.out(decoder_out)
        return out
    
    # create mask for tgt
    def get_tgt_mask(self, seq_length: int) -> torch.tensor:
        # keep decoder from peeking ahead (i.e. show one word at a time in the sequence)
        mask = torch.tril(torch.ones(seq_length, seq_length) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask
    
    # create binary encoded matrix ignore pad
    def create_pad_mask(self, matrix: torch.tensor, pad_val: int=0) -> torch.BoolTensor:
        # matrix = [3, 2, 1, 8, 0, 0, 0], pad_v = 0 -> [False, False, False, False, True, True, True]
        return (matrix == pad_val)


def train(net: Transformer, optimizer: torch.optim, loss_fn: torch.nn, dataloader: DataLoader, epochs: int=3, epoch_pct: float=0.25, device: torch.device=None):
    net.train()
    n = len(dataloader.dataset)
    m = len(dataloader)
    acc_loss = 0

    for epoch in range(epochs):
        samples_trained = 0
        
        for i, batch in enumerate(dataloader, 0):
            batch_size = batch[0].size(0)
            # get inputs and move to device
            inputs, labels = batch
            src, tgt = inputs.to(device), labels.to(device)

            # create SOS and EOS tokens
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

            # mask out pad tokens and mask target input to keep decoder from cheating
            tgt_len = tgt_input.size(1)
            tgt_mask = net.get_tgt_mask(tgt_len).to(device)
            src_pad_mask = net.create_pad_mask(src, pad_val=0).to(device)
            tgt_pad_mask = net.create_pad_mask(tgt_input, pad_val=0).to(device)

            # compute prediction (softmax)
            pred = net(src, tgt_input, tgt_mask, src_pad_mask, tgt_pad_mask)
            pred = pred.permute(1, 2, 0) # reshape

            # compute loss and backpropagate
            loss = loss_fn(pred, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_loss += loss.detach().item()
            samples_trained += batch_size
            
    # display information
            if (i + 1) % int(m * epoch_pct) == 0:
                print(f'{(i + 1) / m * 100:.0f}% of epoch completed\n{samples_trained if samples_trained < n else n}/{n} Samples trained\nLoss: {loss.item():.4f}')
        print(f'Epoch {epoch + 1} complete\n{n}/{n} Samples trained\nLoss: {loss.item():.4f}')
    print(f'Training complete\nAverage loss: {acc_loss / (m * epochs):.4f}')