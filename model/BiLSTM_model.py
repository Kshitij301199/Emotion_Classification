import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, vocabulary_size:int= 52000,
                 embedding_dim:int= 64, hidden_size:int= 32,
                 num_classes:int= 6,p:float= 0.2):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(p)
        self.dense = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, num_classes),
                                #    nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change shape for LSTM
        _, (x, _) = self.lstm(x)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)  # Flatten LSTM output
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x
    
class BiLSTMWithAttention(nn.Module):
    def __init__(self, vocabulary_size=52000, embedding_dim=64, hidden_size=32, num_classes=6, dropout_p=0.2):
        super(BiLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout_p)
        self.attention_weight = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        self.dense = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        
        nn.init.xavier_uniform_(self.attention_weight)

    def forward(self, x):
        # Embedding layer
        x_embedded = self.embedding(x)
        
        # LSTM layer
        lstm_output, _ = self.lstm(x_embedded)
        
        # Attention mechanism
        attention_weights = torch.matmul(lstm_output, self.attention_weight)
        attention_weights = torch.squeeze(attention_weights, -1)
        attention_scores = self.softmax(attention_weights)
        attention_output = torch.sum(lstm_output * attention_scores.unsqueeze(-1), dim=1)
        
        # Final output layers
        x = self.batchnorm(attention_output)
        x = self.dropout(x)
        x = self.dense(x)
        
        return x