import json
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class BoW(nn.Module):

    def __init__(self, vocab_size, num_labels):
        super(BoW, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)

# Count the number of appearrence 
# Of the words in the sentence
def make_bow_vec(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])
    """vec = torch.zeros(len(label_to_ix))
    for word in label:
        vec[label_to_ix[word]] += 1
    return vec.view(1, -1)"""


def embedding(vocab):
    # Get the number of word in the vocabulary
    nb_word = len(vocab)
    print(nb_word)
    # Create an embedding for the words
    embeds = nn.Embedding(nb_word, 10)

    # Convert str to indices
    word_to_ix = {word : i for i, word in enumerate(vocab)}
    print(word_to_ix)

    # Create the model
    """ Meme input_nb que output_nb ?"""
    model = BoW(nb_word,nb_word)

    # Print the parameters
    for param in model.parameters():
        print(param)

    with torch.no_grad():
        sample = ["Located", "East", "Object", "South", "East"]
        bow_vec = make_bow_vec(sample, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

    train(model,word_to_ix)

    """# Get indices based on the sentence
    # TO CHANGE 
    sentence = ["Located", "East", "Object", "South", "East"]
    indices = [word_to_ix[w] for w in sentence]
    print(indices)
    print(make_bow_vec(sentence, word_to_ix))

    # Create tensor
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    print(indices_tensor)

    # Embedding of the words
    emb = embeds(indices_tensor)
    print(emb.shape)
    print(emb)"""

def load():
    # Sentences
    sentences = []
    # Open file
    with open("utils/data/Sentences_Generated_P1.json", 'r') as f:
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

def train(model, word_to_ix):
    # Run to test before train
    with torch.no_grad():
        sample = ["Located", "East", "Object", "South", "East"]
        bow_vec = make_bow_vec(sample, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

    # Print the matrix column corresponding to "creo"
    print(next(model.parameters())[:, word_to_ix["Object"]])

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print("Load data")
    data = load()
    print("start training")
    
    for epoch in range(2):
        i = 0
        for sentence in data:
            for word in sentence:
                # Clear gradient
                model.zero_grad()

                bow_vec = make_bow_vec(sentence, word_to_ix)
                target = make_target(word, word_to_ix)

                # Forward
                log_probs = model(bow_vec)

                # Loss, gradients, update
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()
            i += 1
            if i % 10000 == 0 :
                print(i)
        print("half way")

    # Apres training
    print("training done")

    with torch.no_grad():
        sample = ["Located", "East", "Object", "South", "East"]
        bow_vec = make_bow_vec(sample, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

    print(next(model.parameters())[:, word_to_ix["Object"]])

