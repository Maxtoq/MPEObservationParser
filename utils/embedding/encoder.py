import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers = 1, dropout=0.):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Embedding
        """
        input_size : number of words
        embedding_size : dimension of the embedding
        """
        self.embedding = nn.Embedding(input_size, embedding_size)

        # self.dropout = nn.Dropout(dropout)

        # We use GRU
        """
        input_size or embedding_size ?
        dropout ?
        bidirectional ?
        batch_first ?
        """
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

        

    def forward(self, x, mask, lengths):
        """packed = torch._pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = torch._pad_packed_sequence(output,batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]
        
        return output, final"""

        # Embedding
        embedding = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell