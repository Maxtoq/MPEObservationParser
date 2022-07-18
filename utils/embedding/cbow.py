import json
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt

EMBED_DIM = 10
CONTEXT_SIZE = 2
EMBED_MAX_NORM = 1

class CBoW(nn.Module):
    def __init__(self, vocab_size):
        super(CBoW).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIM,
            # utile ?
            max_norm=EMBED_MAX_NORM,
        )
        self.linear1 = nn.Linear(
            # Dim of the input (embedding)
            in_features=EMBED_DIM,
            # Dim of the output (number of possible words)
            # Or 128 ??
            out_features=128
        )
        self.activation_func1 = nn.ReLu()

        self.linear2 = nn.Linear(
            in_features=128,
            out_features=vocab_size,
        )
        self.activation_func2 = nn.LogSoftmax(dim = -1)

    def forward(self, inputs):
        """x = self.embeddings(inputs)
        # mean ?
        x = x.mean(axis = 1)
        x = self.linear(x)
                
        return x"""

        embeds = sum(self.embeddings(inputs)).view(1,-1)
        # forward 
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)

        return out

    def get_word_emdedding(self, word, word_to_ix):
        word = torch.tensor([word_to_ix[word]])
        return self.embeddings(word).view(1,-1)


def embedding(vocab):

    print("---------- We try CBoW ----------")

    # Get the number of word in the vocabulary
    nb_word = len(vocab)
    print(nb_word)
    # Create an embedding for the words
    embeds = nn.Embedding(nb_word, 10)

    # Convert str to indices
    word_to_ix = {word : i for i, word in enumerate(vocab)}
    print(word_to_ix)
    ix_to_word = {i : word for i, word in enumerate(vocab)}
    print(ix_to_word)

    # Prepare the training data
    data = load()
    ngrams = []
    for sentence in data:
        ngram = [
        (
            [sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
            sentence[i]
        )
        for i in range(CONTEXT_SIZE, len(sentence))
        ]
        #print(ngrams[:3])
       
        ngrams.append(ngram)
    #print(ngrams[:3][:3])
    print(ngrams[0])
    print(ngrams[0][0])

    # Create the model
    """ Meme input_nb que output_nb ?"""
    model = CBoW(nb_word)

    # Print the parameters
    for param in model.parameters():
        print(param)

    """with torch.no_grad():
        sample = ["Located", "East", "Object", "South", "East"]
        bow_vec = make_bow_vec(sample, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

        model=CBoW

        criterion = nn.NLLLoss()
        optim = optim.SGD(list(model.parameters()) + list(model.parameters()), lr=0.001)"""

    losses = []
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    total_loss = 0

    # Training 
    for epoch in range(2):
        i = 0
        for sentence in ngrams:
            for context, target in sentence:

                # Turn the words into integer indices
                # And wrap them in tensors
                context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

                # Zero out the gradients 
                model.zero_grad()

                # Forward pass
                log_probs = model(context_idxs)

                # Loss function
                loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

                # Backward pass and update the gradient
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            i += 1
            if i % 10000 == 0:
                print(str(i))
                # Get the Python number from a 1-element Tensor by calling tensor.item()
                losses.append(total_loss)
                total_loss = 0
                
                
                
    print(losses)  # The loss decreased every iteration over the training data!

    # Representation with matplotlib
    y = np.array(losses)
    plt.plot(y)
    plt.show()

    # To get the embedding of a particular word
    print(word_to_ix)
    print(model.embeddings.weight)
    print(model.embeddings.weight[word_to_ix["West"]])



def load():
    # Sentences
    sentences = []
    # Open file
    with open("utils/data/Sentences_Generated_P2.json", 'r') as f:
        data = json.load(f)
    # Get the sentence of both agent for each step
    for step in range(len(data) -1 ) :
        sentences.append(data["Step " + str(step)]["Agent_0"]["Sentence"])
        sentences.append(data["Step " + str(step)]["Agent_1"]["Sentence"])
    
    if sentences[0][0] == '<SOS>':
        for sentence in sentences:
            # We delete first and last character
            sentence.pop(0)
            sentence.pop()
    return sentences
