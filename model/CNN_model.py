import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
class CNNTextClassifier(nn.Module):
    def __init__(self, 
                 vocab_size:int = 52000, 
                 embedding_dim:int = 16, 
                 input_length:int = 100, 
                 num_filters:int = 128, 
                 kernel_size:int = 5, 
                 hidden_units:int = 64, 
                 num_classes:int = 6):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=input_length - kernel_size + 1)
        self.fc1 = nn.Linear(num_filters, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Change shape for Conv1D
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.squeeze(2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
        