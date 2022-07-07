import numpy as np
import torch
import torch.nn as nn
from preprocessing import augment, encode, pad

class LSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, layers, dropout=0.5):
        super().__init__()
        self.output_size = output_size
        self.layers = layers
        self.hidden_dim = hidden_dim

        # embedding and lstm layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                            num_layers=layers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(0.3)
        # linear layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden):
        batch_size = x.size(0)

        # outputing values from lstm based on embeddings
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)

        # stack outputs from lstm layers, drop outputs, compute sigmoid
        x = x.contiguous().view(-1, self.hidden_dim)
        x = self.drop(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        # reshape to have batch size size first
        x = x.view(batch_size, -1)
        x = x[:, -1]

        return x, hidden

    def init_hidden(self, batch_size):
        # create new tensors initialized to zero for hidden state & lstm cell state
        w = next(self.parameters()).data
        hidden = (w.new(self.layers, int(batch_size), self.hidden_dim).zero_(), w.new(self.layers, int(batch_size), self.hidden_dim).zero_())
        
        return hidden

# trains the lstm network
def train(net: nn.Module, trainloader: torch.utils.data.DataLoader, optimizer: torch.optim, loss_fn=None, clip=5, verbose=0.25, epochs=3, device=None) -> None:

    torch.cuda.empty_cache() # free VRAM if possible

    # prepare metrics for training
    net.train() # notify layers to train
    n = len(trainloader.dataset)
    batch_size = trainloader.batch_size

    print('Training started')
    # iterate over epochs
    for epoch in range(epochs):
        samples_trained = 0
        h = net.init_hidden(batch_size) # init hidden state

        # iterate batches
        for i, data in enumerate(trainloader, 0):
            # grab input and batch infp
            inputs, labels = data
            
            h = tuple([tensor.data for tensor in h]) # create new tensors for each hidden state

            net.zero_grad() # reset gradient

            # transform data and get prediction
            inputs = inputs.type(torch.LongTensor)

            # move tensors to device if possible
            if device:
                inputs, labels = inputs.to(device), labels.to(device) # move to GPU (if applicable)

            outputs, h = net(inputs, h)
            # find loss, and update weights according to computed gradient
            loss = loss_fn(outputs.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip) # prevent exploding gradient
            optimizer.step() # make step against gradient (slope)

            samples_trained += batch_size
            if (i + 1) % (int(len(trainloader) * verbose)) == 0:
                print(f"epoch: {epoch + 1}/{epochs}\nsamples trained: {samples_trained}/{n}\nloss: {loss.item()}")
            torch.cuda.empty_cache() # free VRAM if possible
        print(f'epoch complete {n}/{n} samples trained')
    print(f'training complete')


# test peformance of network
def test(net: nn.Module, testloader: torch.utils.data.DataLoader, loss_fn=None, device=None) -> tuple:

    torch.cuda.empty_cache() # free VRAM if possible

    if device == None:
        net.to('cpu')

    # prepare metrics
    net.eval() # indicate to layers model is being tested
    n = len(testloader.dataset)
    batch_size = testloader.batch_size
    loss = 0
    correct = 0

    for _ in range(1): 

        h = net.init_hidden(batch_size) # init first hidden state
        
        print('Testing started')
        # find loss & num correct from predictions
        for i, data in enumerate(testloader, 0):
            
            inputs, labels = data
            h = tuple([tensor.data for tensor in h]) # get hidden state
            inputs = inputs.type(torch.LongTensor)

            # use GPU if possible
            if device:
                inputs, labels = inputs.to(device), labels.to(device)

            # find loss and total correct
            outputs, h = net(inputs, h)
            loss += loss_fn(outputs.squeeze(), labels.float())
            pred = torch.round(outputs.squeeze()) # round to nearest int
            correct += pred.eq(labels.float().view_as(pred)).sum().item()  # total correct in batch
            torch.cuda.empty_cache() # free VRAM if possible

    loss /= n # avg the loss
    acc = correct / n # get accuracy

    return loss, acc

def predict(net: nn.Module, embeddings: dict, stopwords: set, punctuation: set, maxlen: int) -> np.ndarray:
    
    net.to('cpu') # don't need GPU for compuations
    net.eval() # indicate to layers not to train

    reviews = []
    # get review(s)
    while True:
        review = input("input a review:")
        if review:
            reviews.append(review)
        elif reviews:
            break

    # transform review to proper tensor
    sequences = augment(reviews, stopwords, punctuation)
    encodings = encode(sequences, embeddings, default=0)
    padded = pad(encodings, maxlen, fill=0)
    inputs = torch.from_numpy(padded)
    inputs = inputs.type(torch.LongTensor)
    batch_size = inputs.size(0)


    # make prediction
    h = net.init_hidden(batch_size)
    outputs, h = net(inputs, h)
    pred = torch.round(outputs.squeeze())
    pred = pred.tolist()

    # sigular predictions
    if type(pred) is float:
        pred = [pred]
    
    # display prediction
    for i, label in enumerate(pred):
        print(f"{reviews[i]} is a {'positive' if label else 'negative'} review")