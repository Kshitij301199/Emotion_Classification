import torch.nn as nn
        
class CNNTextClassifier(nn.Module):
    def __init__(self,
                 vocab_size:int = 52000,
                 embedding_dim:int = 32,
                 input_length:int = 100,
                 num_filters:int = 24,
                 kernel_size:int = 3,
                 hidden_units:int = 16,
                 num_classes:int = 6):
        super(CNNTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                                            kernel_size=kernel_size),
                                  nn.MaxPool1d(kernel_size=input_length - kernel_size + 1),
                                  nn.ReLU(),
        )
        self.dense = nn.Sequential(nn.Linear(num_filters, hidden_units),
                                   nn.ReLU(),
                                   nn.Linear(hidden_units, num_classes),
                                #    nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Change shape for Conv1D
        x = self.conv(x)
        x = x.squeeze(2)
        x = self.dense(x)
        return x