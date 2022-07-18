import json
from tqdm import tqdm
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 32)
        self.linear2 = nn.Linear(32, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def embedding(vocabulary):

    print("---------- We try n-gram ----------")

    CONTEXT_SIZE = 1
    EMBEDDING_DIM = 10
    #vocab = set(vocabulary)
    vocab_size = len(vocabulary)

    #test_sentence = ["Located North West Object South Landmark East West You Push Object North"].split()

    word_to_ix = {word : i for i, word in enumerate(vocabulary)}
    print(word_to_ix)

    # Prepare the training data
    data = load()
    """data = [
        ["Located", "South"], 
        ["Located", "Center", "Object", "East"], 
        ["Located", "South"], 
        ["Located", "East", "Object", "South", "East"], 
        ["Located", "South", "West"]
    ]"""


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
    print(data[0])
    print(ngrams[0])
    print(ngrams[0][0])

    
    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocabulary), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    total_loss = 0

    embeddings = model.embeddings.weight.cpu().detach().numpy()
    print(embeddings)

    # Show embeddings
    data_frame = pd.DataFrame(embeddings)
    tsne = TSNE(n_components=2)
    embed_df = tsne.fit_transform(data_frame)
    embed_df = pd.DataFrame(embed_df, columns=list('XY'))
    embed_df["Words"] = vocabulary


    # Create the fig
    fig = px.scatter(embed_df, x='X', y='Y', text='Words', log_x = False, size_max = 60)

    fig.update_traces(textposition='top center')
    # setting up the height and title
    fig.update_layout(
        height=600,
        title_text='Word embedding chart'
    )
    # displaying the figure
    fig.write_html("../n-gram_visu_B.html")




    """print("Similar word South: ")
    for word, sim in get_top_similar(model,word_to_ix,"South").items():
        print("{}: {:.3f}".format(word, sim))

    print("Similar word Object: ")
    for word, sim in get_top_similar(model,word_to_ix,"Object").items():
        print("{}: {:.3f}".format(word, sim))"""

    big_loss = []

    for epoch in tqdm(range(200)):
        i = 0
        batch = random.sample(ngrams,64)
        #batch = ngrams
        # Zero out the gradients 
        model.zero_grad()
        loss = 0
        for sentence in batch:
            # context_batch = 
            # target_batch = 
            for context, target in sentence:
                # Turn the words into integer indices
                # And wrap them in tensors
                #print(" -------------------")
                context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
                #print(context_idxs)

                # Forward pass
                log_probs = model(context_idxs)
                #print(word_to_ix[target])
                
                #print(log_probs)

                # Loss function
                loss += loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
                #print(loss)
                
                #print(model.embeddings.weight)
                total_loss += loss.item()
                
                if total_loss > 3:
                    n = []
                    n.extend([[[context], [context_idxs]], [[target], [word_to_ix[target]]], [total_loss]])
                    big_loss.append(n)
                total_loss = 0
        
            i += 1
            if i % 10000 == 0:
                print(str(i))
                print(log_probs)
                print(word_to_ix[target])
                print(loss)
                # Get the Python number from a 1-element Tensor by calling tensor.item()
        
        losses.append(loss.item() / 64)
        # Backward pass and update the gradient
        loss.backward()
        #(model.embeddings.weight.grad)
        #print(model.embeddings.weight)
        optimizer.step()
                
                
                
                
                
    #print(losses)  # The loss decreased every iteration over the training data!
    for p in big_loss:
        print(p)

    # Representation with matplotlib
    y = np.array(losses)
    plt.plot(y)
    plt.show()

    # To get the embedding of a particular word
    print(word_to_ix)
    print(model.embeddings.weight)
    print(model.embeddings.weight[word_to_ix["West"]])

    embeddings = model.embeddings.weight.cpu().detach().numpy()
    print(embeddings)
    
    """# normalization
    norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
    print(norms)
    norms = np.reshape(norms, (len(norms), 1))
    print(norms)
    embeddings_norm = embeddings / norms
    print(embeddings_norm)"""

    # Show embeddings
    data_frame = pd.DataFrame(embeddings)
    tsne = TSNE(n_components=2)
    embed_df = tsne.fit_transform(data_frame)
    print(embed_df)
    """embed_df = pd.DataFrame(embed_df)
    print(embed_df)
    
    # Words index
    embed_df.index = vocabulary

    # Create the fig
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=embed_df[0],
            y=embed_df[1],
            mode="text",
            text=embed_df.index,
            textposition="middle center",
        )
    )
    fig.write_html("../n-gram_visu.html")"""

    
    embed_df = pd.DataFrame(embed_df, columns=list('XY'))
    embed_df["Words"] = vocabulary
    print(embed_df["Words"])


    # Create the fig
    fig = px.scatter(embed_df, x='X', y='Y', text='Words', log_x = False, size_max = 60)

    fig.update_traces(textposition='top center')
    # setting up the height and title
    fig.update_layout(
        height=600,
        title_text='Word embedding chart'
    )
    # displaying the figure
    fig.write_html("../n-gram_visu.html")
    
    """# plotting a scatter plot
    fig = px.scatter(data_frame, x="X", y="Y", text="Words", log_x=True, size_max=60)
    # adjusting the text position
    fig.update_traces(textposition='top center')
    # setting up the height and title
    fig.update_layout(
        height=600,
        title_text='Word embedding chart'
    )
    # displaying the figure
    fig.show()"""


    """print("Similar word South: ")
    for word, sim in get_top_similar(model,word_to_ix,"South").items():
        print("{}: {:.3f}".format(word, sim))

    print("Similar word Object: ")
    for word, sim in get_top_similar(model,word_to_ix,"Object").items():
        print("{}: {:.3f}".format(word, sim))"""

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

def get_top_similar(model, word_to_ix, word: str, topN: int = 10):
    word_id = word_to_ix[word]

    # embedding from first model layer
    embeddings = list(model.parameters())[0]
    embeddings = embeddings.cpu().detach().numpy()
    norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
    norms = np.reshape(norms, (len(norms), 1))
    embeddings_norm = embeddings / norms

    word_vec = embeddings_norm[word_id]
    word_vec = np.reshape(word_vec, (len(word_vec), 1))
    dists = np.matmul(embeddings_norm, word_vec).flatten()
    topN_ids = np.argsort(-dists)[1 : topN + 1]

    topN_dict = {}
    for sim_word_id in topN_ids:
        for key, value in word_to_ix.items():
            if sim_word_id == value:
                mot = key
        sim_word = mot
        #.lookup_token(sim_word_id)
        topN_dict[sim_word] = dists[sim_word_id]
    return topN_dict

