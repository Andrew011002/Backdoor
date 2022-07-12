import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout, maxlen) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Positional Encoding Sinusoid Formula
        pos_encoding = torch.zeros(maxlen, d_model)
        pos = torch.arange(0, maxlen, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        denom = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)) / d_model) # 1000^(2i/d_model)        
        pos_encoding[:, 0::2] = torch.sin(pos * denom) # even indicies
        pos_encoding[:, 1::2] = torch.cos(pos * denom) # 
        
        # saving encoingss without gradients
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_encoding', pos_encoding) # save in network params
        
    def forward(self, token_embedding):
        # Residual connection + positional encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):

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
    

    def forward(self, src: torch.LongTensor, tgt: torch.LongTensor, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None) -> torch.LongTensor:
        # src shape: (batch_size, src seq_length), tgt shape: (batch_size, tgt seq_length)

        # embedding + positional encoding shape: (batch_size, sequence length, d_model)
        src = self.embedding(src) * np.sqrt(self.d_model)
        tgt = self.embedding(tgt) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # reshape: (seq_len, batch_size, d_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # transformer inputs shape: (seq_len, batch_size, n_tokens)
        decoder_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)

        # compute linear function and apply softmax (Cross Entropy Loss already does this just return n_tokens as out)
        out = self.out(decoder_out)
        return out # shape: (n_tokens, batch_size, seq_len)
    
    # create mask for tgt
    def get_tgt_mask(self, seq_length: int) -> torch.LongTensor:
        # keep decoder from peeking ahead (i.e. show one word at a time in the sequence)
        mask = torch.tril(torch.ones(seq_length, seq_length) == 1) # lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # convert ones to 0
        return mask
    
    # create binary encoded matrix ignore pad
    def create_pad_mask(self, matrix: torch.LongTensor, pad_val: int=0) -> torch.BoolTensor:
        # matrix = [3, 2, 1, 8, 0, 0, 0] & pad_val = 0 -> [False, False, False, False, True, True, True]
        return (matrix == pad_val)

# trains network 
def train(net: Transformer, optimizer: torch.optim, loss_fn: torch.nn, dataloader: DataLoader, pad_val: int=0, epochs: int=3, epoch_pct: float=0.25, device: torch.device=None):
    # prepare network & metrics for training
    net.train()
    n = len(dataloader.dataset)
    m = len(dataloader)
    accum_loss = 0

    # train net over dataset for e epochs
    for epoch in range(epochs):
        samples_trained = 0
        # train over each batch
        for i, batch in enumerate(dataloader, 0):
            # get inputs and move to device
            inputs, labels = batch
            batch_size = inputs.size(0)
            src, tgt = inputs.to(device), labels.to(device)

            # create SOS and EOS tokens
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

            # mask out pad tokens and mask target input to keep decoder from cheating
            tgt_len = tgt_input.size(1)
            tgt_mask = net.get_tgt_mask(tgt_len).to(device)
            src_pad_mask = net.create_pad_mask(src, pad_val=pad_val).to(device)
            tgt_pad_mask = net.create_pad_mask(tgt_input, pad_val=pad_val).to(device)

            # compute prediction (softmax)
            pred = net(src, tgt_input, tgt_mask, src_pad_mask, tgt_pad_mask)
            pred = pred.permute(1, 2, 0) # reshape (batch_size, seq_len, n_tokens)

            # compute loss and backpropagate
            loss = loss_fn(pred, tgt_output)
            optimizer.zero_grad() # reset gradient
            loss.backward() # compute gradient
            optimizer.step() # update net parameters from gradient
            accum_loss += loss.item() # accumulate loss
            samples_trained += batch_size # keep track of sample trained
            
            # display info
            if (i + 1) % int(m * epoch_pct) == 0:
                print(f'{(i + 1) / m * 100:.0f}% of epoch completed | {samples_trained if samples_trained < n else n}/{n} samples trained | loss: {loss.item():.4f}')
        print(f'Epoch {epoch + 1} complete\n{n}/{n} samples trained | loss: {loss.item():.4f}')
    print(f'Training complete\navg loss: {accum_loss / (m * epochs):.4f}')

# tests transformer given testloader
def test(net: Transformer, loss_fn: torch.nn, testloader: DataLoader, print_pct=0.25, device: torch.device=None):
    # prepare network and metrics
    net.eval()
    accum_loss = 0
    correct = 0
    m = len(testloader)

    # don't compute new gradients
    with torch.no_grad():
        outputs_seen = 0

        # test each batch
        for i, batch in enumerate(testloader):

            # get the inputs and labels & move to device
            inputs, labels = batch
            src, tgt = inputs.to(device), labels.to(device)

            # shift to get SOS & EOS tokens 
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

            # mask padding for src & tgt, also mask tgt to prevent decoder from cheating
            tgt_len = tgt_input.size(1)
            tgt_mask = net.get_tgt_mask(tgt_len).to(device)
            src_pad_mask = net.create_pad_mask(src).to(device)
            tgt_pad_mask = net.create_pad_mask(tgt_input).to(device)
            
            # get prediction
            pred = net(src, tgt_input, tgt_mask, src_pad_mask, tgt_pad_mask) # tensor of likelihoods (softmax not applied)
            pred = pred.permute(1, 2, 0) # reshape: (batch_size, seq_len, n_tokens)
            loss = loss_fn(pred, tgt_output) # find loss (softmax applied)
            accum_loss += loss.item() # accumulate average batch loss

            # find accuracy
            batch_size = inputs.size(0)
            pred = torch.argmax(pred, dim=1) # get max val for each token vector
            correct += (pred == tgt_output).sum().item() # find where tokens match target output
            outputs_seen += batch_size * tgt_len # keep track of amount of tokens seen

            # display info
            if (i + 1) % (m * print_pct) == 0:
                print(f'{(i + 1) / m * 100:.0f}% of test completed | loss: {loss.item():.4f} | acc: {correct / outputs_seen:.4f}')
        print(f'Testing complete | avg loss: {accum_loss / m:.4f} | acc: {correct / outputs_seen:.4f}')

# makes predictions from tokenized sequences
def predict(net: Transformer, inputs: np.array, pred_to: int, device: torch.device=None) -> list:    
    predictions = []

    # iterate each tokenized sequence
    for src in inputs:
        src = np.array([src]) # reshape: (batch_size, seq_len)
        seq_len = len(src[0]) # seq_len
        tgt_input = [[np.squeeze(src)[0]]] if seq_len > 1 else src.copy() # get SOS (start of sentence)
        src = torch.tensor(src, dtype=torch.long, device=device) # convert to tensor
        tgt_input = torch.tensor(tgt_input, dtype=torch.long, device=device) # convert to tensor

        # predict next 'pred_to' tokens
        for i in range(pred_to):
            # get the mask to keep decoder from cheating
            tgt_mask = net.get_tgt_mask(tgt_input.size(1)).to(device)

            pred = net(src, tgt_input, tgt_mask) # compute prediction
            token = pred.topk(1)[1].view(-1)[-1].item() # get next token of highest probability
            
            token = torch.tensor([[token]], dtype=torch.long, device=device) # reshape (batch_size, seq_len)
            tgt_input = torch.cat((tgt_input, token), dim=1) # combine with previous tgt_input

        # add tokenized predicted sequence to predictions 
        predictions.append(tgt_input[:, 1:].to('cpu').numpy().squeeze()) # reshape: (seq_len)
    return predictions