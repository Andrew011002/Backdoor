import numpy as np
import pandas as pd
import os
from collections import Counter

# removes punctuation
def remove_punc(word: str, p: set):
    i = word.find('br')
    if i != -1:
        if word[i:i+3] == 'br':           
            word = word.replace('br', '')   
    word = ''.join([c for c in word if c not in p])
    return word

# makes data more rich
def augment(data: np.ndarray, stopwords=set(), punctuation=set()) -> np.ndarray:
    augmented = []

    # modify sequences
    for sequence in data:
        sequence = sequence.lower().split() # lower the casing & transform to vector of word tokens
        sequence = [remove_punc(w, punctuation) for w in sequence] # remove the punction from defined punctuation set
        sequence = [w for w in sequence if w not in stopwords] # remove stopwords
        sequence = ' '.join(sequence) # vec2seq
        augmented.append(sequence)

    return np.array(augmented)

# creates embeddings from text sequences
def generate_embeddings(sequences: list) -> dict:
    alltext = ' '.join([seq for seq in sequences]) # concatenate all the sequences to one big sequence of words
    allwords = alltext.split() # get every word token from the entire sequence
    vocabulary = Counter(allwords) # define the vocabulary by occurences (word frequency)
    size = len(vocabulary) 
    mostcommon = vocabulary.most_common(size) # create mappings of most common words where the lower indices of words are the most frequent
    embeddings = {w: i + 1 for i, (w, c) in enumerate(mostcommon)} # create embeddings (omit 0 for paadding/unknown word purposes)

    return embeddings

# encodes sequences based on embeddings
def encode(sequences: np.ndarray, embeddings: dict, default=None) -> np.ndarray:
    encodings = []

    # modify sequences
    for sequence in sequences:
        sequence = [embeddings.get(w, default) for w in sequence.split()] # encode based on embeddings
        encodings.append(sequence)

    return np.array(encodings)

# encodes labels to real values
def encode_labels(labels: np.ndarray, encodings: dict) -> np.ndarray:
    return np.array([encodings[l] for l in labels]).astype(np.int64)


# pads the vectors or cuts them to a desired maxlen
def pad(vectors: np.ndarray, maxlen: int, fill=0) -> np.ndarray:
    padded = []

    # pad vectors
    for vector in vectors:
        n = len(vector)
        # vector smaller than maxlen (add pad)
        if n < maxlen:
            vector = np.pad(vector, (maxlen - n, 0), mode='constant', constant_values=fill)
        # vector larger than maxlen (cut vector)
        if n > maxlen:
            vector = vector[:maxlen]
        padded.append(vector)

    return np.array(padded).astype(np.int64)

# converts vectors to sequences based on embeddings
def vec2seq(vectors: np.ndarray, embeddings: dict, default: str) -> np.ndarray:
    sequences = []
    encodings = {i: w for w, i in embeddings.items()} # find encodings by flipping embeddings
    
    # modify vectors
    for vector in vectors:
        vector = [encodings.get(i, default) for i in vector] # find the word or default based on encoding value
        sequence = ' '.join(vector)
        sequences.append(sequence)

    return np.array(sequences)

# splits array to training and testing data
def splitter(inputs: np.ndarray, labels: np.ndarray, split=0.2) -> tuple:
    # validate size
    if len(inputs) == len(labels):
        n = len(inputs)

    # define split
    cut = int(n * split)
    x_train, y_train = inputs[:cut], labels[:cut]
    x_test, y_test = inputs[cut:], labels[cut:]

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    

# Example Run
if __name__ == '__main__':
    # load example data
    path = os.path.abspath(os.path.dirname(__file__))
    df = pd.read_csv(f'{path}/data/imbd_dataset.csv')
    data = df.review.to_numpy()[:100]

    # augmenting the data
    stopwords = {'the', 'has', 'that', 'am', 'as', 'are', 'to', 'is', 'it', 'for', 'with', 'this'}
    punctuation = {'.', '!', '@', '<', '>', '[', ']', '{', '}', '?', ',', ';', ':', '/', "'", '"'}
    sequences = augment(data, stopwords, punctuation)
    print(f'Example sample:\n{sequences[0]}')

    # creating embeddings
    embeddings = generate_embeddings(sequences)
    print(f'Embeddings: {embeddings}') # (key: word, value = frequency relevance)

    # encoding sequences to vectors
    vectorized = encode(sequences, embeddings, default=0)
    print(f'Example encoded vector:\n{vectorized[0]}') # should be strictly natural numbers (non-zero)

    # turning vectors back to sequences
    sequences_copy = vec2seq(vectorized, embeddings, default='NONE')
    print(f'sequences equal? {sequences_copy[0].split() == sequences[0].split()}') # should be identical

    # adding pad to vectors
    padded = pad(vectorized, maxlen=150, fill=0)
    print(f'Example of pre-padded vector:\n{padded[1]}')
    print(f'vectors same size? {len(max(padded, key=len)) == len(min(padded, key=len))}') # smallest vector shares length as largest vector

