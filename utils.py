from typing import List, AnyStr, Dict
from itertools import product
import torch
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from torch.utils.data import Dataset, DataLoader, Subset
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns
import matplotlib.pyplot as plt

def tokenize(sentence: AnyStr) -> List:
    tokens = word_tokenize(sentence)
    return tokens

def remove_stopwords(tokens: List) -> AnyStr:
    stop_words = set(stopwords.words('english'))
    tokens_wo_stop_words = [token for token in tokens if token not in stop_words]
    # out = " ".join(tokens_wo_stop_words)
    return tokens_wo_stop_words


def stemming(tokens: List) -> List:
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(token) for token in tokens]
    return stemmed_words

def fit_tokenizer(training_data):
    tokenizer = Tokenizer(num_words=52000, oov_token="<OOV>")
    tokenizer.fit_on_texts(training_data)
    return tokenizer

def pad_sequence(sequence: List, tokenizer, max_len:int = 100) -> torch.TensorType:
    seq = tokenizer.texts_to_sequences(sequence)
    seq = [sub_seq if sub_seq != [] else [1] for sub_seq in seq]
    try:
        seq_ten = torch.tensor(seq).flatten()
        out_ten = torch.zeros(size= (max_len,)).long()
        out_ten[:seq_ten.size(0)] = seq_ten
        return out_ten
    except ValueError:
        print(seq)
        return torch.zeros(size= (max_len,)).long()
    
class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe[["padded","label"]]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        output = self.dataframe.iloc[index]
        return {
            "padded": output['padded'],
            "label": output['label']
        }
        
def get_batched_data(dataset: Dataset, batch_size:int = 64):
    return DataLoader(dataset, batch_size)

def plot_loss_acc(loss: List, accs: List, modelname: AnyStr, plot = False) -> None:
    plt.style.use("seaborn")
    fig, axes = plt.subplots(1,2, figsize= (12,6))
    num_of_items = len(loss)
    sns.lineplot(ax= axes[0], x= range(num_of_items), y= loss)
    sns.lineplot(ax= axes[1], x= range(num_of_items), y= accs)
    
    axes[0].set_xlabel("No. of Iterations", fontdict={"fontweight":"bold"})
    axes[1].set_xlabel("No. of Iterations", fontdict={"fontweight":"bold"})
    axes[0].set_ylabel("Loss", fontdict={"fontweight":"bold"})
    axes[1].set_ylabel("Validation Accuracies", fontdict={"fontweight":"bold"})
    axes[0].set_title("Loss", fontdict={"fontweight":"bold", "fontsize":15})
    axes[1].set_title("Validation Accuracy", fontdict={"fontweight":"bold", "fontsize":15})
    if plot:
        plt.show()
    else:
        if modelname == 'bilstm':
            fig.savefig(fname= "./images/bilstm_loss_acc.png", dpi=300)
        elif modelname == 'cnn':
            fig.savefig(fname= "./images/cnn_loss_acc.png", dpi=300)
               
def save_model(model, filepath):
    """
    Save PyTorch model parameters to a file.

    Args:
    - model (torch.nn.Module): PyTorch model to save.
    - filepath (str): Filepath to save the model parameters.
    """
    torch.save(model, filepath)
    print(f"Model parameters saved to '{filepath}'")

def load_model(model, filepath):
    """
    Load PyTorch model parameters from a file.

    Args:
    - model (torch.nn.Module): PyTorch model to load parameters into.
    - filepath (str): Filepath to the saved model parameters.
    """
    model = torch.load(filepath)
    print(f"Model parameters loaded from '{filepath}'")
    return model

def get_class_weights(y_train) -> torch.FloatTensor:
    class_weights = compute_class_weight(class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train)
    class_weights = torch.FloatTensor(class_weights)
    return class_weights

def split_train_val_dataloader(train_dataloader, val_size=0.2, shuffle=True, random_state=None):
    # Get indices for the train DataLoader
    train_indices = list(range(len(train_dataloader.dataset)))
    
    # Initialize KFold with shuffle and random_state parameters
    kfold = KFold(n_splits=int(1 / val_size), shuffle=shuffle, random_state=random_state)
    
    # Split train indices into train and validation indices
    train_indices_list = []
    val_indices_list = []
    for train_idx, val_idx in kfold.split(train_indices):
        train_indices_list.append(train_idx)
        val_indices_list.append(val_idx)
    
    # Separate train and validation datasets
    train_datasets = []
    val_datasets = []
    for train_idx, val_idx in zip(train_indices_list, val_indices_list):
        train_datasets.append(Subset(train_dataloader.dataset, train_idx))
        val_datasets.append(Subset(train_dataloader.dataset, val_idx))
    
    # Create train and validation DataLoaders
    train_loaders = [DataLoader(dataset, batch_size=train_dataloader.batch_size, shuffle= True,
                                num_workers=train_dataloader.num_workers, pin_memory=train_dataloader.pin_memory)
                     for dataset in train_datasets]
    val_loaders = [DataLoader(dataset, batch_size=train_dataloader.batch_size, shuffle=False,
                              num_workers=train_dataloader.num_workers, pin_memory=train_dataloader.pin_memory)
                   for dataset in val_datasets]
    
    return train_loaders, val_loaders

def grid_search(parameters):
    keys = parameters.keys()
    values = parameters.values()
    
    combinations = list(product(*values))
    
    parameter_configurations = [{k: v for k, v in zip(keys, combination)} for combination in combinations]
    
    return parameter_configurations