import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding
        """
        input_size : number of words
        embedding_size : dimension of the embedding
        """
        self.embedding = nn.Embedding(input_size, embedding_size)

        # We use GRU
        """
        input_size or embedding_size ?
        dropout ?
        bidirectional ?
        batch_first ?

        same hidden_size in Encoder and Decoder
        """
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

        # Fully connected
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # shape of x: (N) bu we want (1, N)
        x = x.unsqueeze(0)

        # Embedding
        embedding = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        predictions = self.fc(outputs)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell
