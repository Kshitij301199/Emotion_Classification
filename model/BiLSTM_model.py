import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class BiLSTMModel(nn.Module):
    def __init__(self, vocabulary_size:int= 52000,
                 embedding_dim:int= 64, hidden_size:int= 64,
                 num_classes:int= 6,p:float= 0.3):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(p)
        self.dense = nn.Sequential(nn.Linear(hidden_size*2, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, num_classes),
                                   nn.Softmax(dim=1))
        
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change shape for LSTM
        _, (x, _) = self.lstm(x)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)  # Flatten LSTM output
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x