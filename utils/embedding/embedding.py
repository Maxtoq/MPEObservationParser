import json
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

"""from utils.encoder import Encoder
from utils.decoder import Decoder
from utils.encoder_decoder import EncoderDecoder"""


def embedding(vocab):
    # Get the number of word in the vocabulary
    nb_word = len(vocab)
    print(nb_word)
    # Create an embedding for the words
    embeds = nn.Embedding(nb_word, 10)
    print(embeds.weight)
    print(embeds.weight.shape)

    # Convert str to indices
    word_to_ix = {word : i for i, word in enumerate(vocab)}
    print(word_to_ix)

    # Get indices based on the sentence
    # TO CHANGE 
    sentence = ["Located", "East", "Object", "South", "East"]
    indices = [word_to_ix[w] for w in sentence]
    print(indices)

    # Create tensor
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    print(indices_tensor)

    # Embedding of the words
    emb = embeds(indices_tensor)
    print(emb.shape)
    print(emb)



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
    print(sentences)

        
def training(vocab):

    # Parameters
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 64

    # Model parameters
    load_model = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size_encoder = len(vocab)
    input_size_decoder = len(vocab)
    output_size = len(vocab)

    encoder_embedding_size = 10
    decoder_embedding_size = 10

    hidden_size = 1000
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5

    # Tensorboard
    # writer = SummaryWriter(f'runs/loss_plot')
    # step = 0

    """train_iterator , valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = batch_size
        sort_within_batch = True,
        sort_key = lambda x: len(x.src),
        device = device)"""

    # Models
    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, 
    hidden_size, num_layers, enc_dropout).to(device)

    decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
    hidden_size, output_size, num_layers, dec_dropout).to(device)

    model = EncoderDecoder(encoder_net,decoder_net).to(device)

    """
    pad_idx ?
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    """

    optimizer = torch.optim.SGD(list(encoder_net.parameters()) + list(decoder_net.parameters()), lr=0.001)

    for epoch in range(num_epochs):
        print("epoch " + epoch + " / " + num_epochs)

        for batch_idx, batch in enumerate(train_iterator):
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # Call the model
            output = model(inp_data, target)

            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameter(), max_norm=1)

            optimizer.step()

            step += 1
            
