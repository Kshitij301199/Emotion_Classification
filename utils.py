import torch
from typing import List, AnyStr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer

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

def lemmatize(tokens: List) -> List:
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_words

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

def plot_loss_acc(loss: List, accs: List, modelname: AnyStr) -> None:
    fig, axes = plt.subplots(1,2, figsize= (12,6))
    num_of_items = len(loss)
    sns.lineplot(ax= axes[0], x= range(num_of_items), y= loss)
    sns.lineplot(ax= axes[1], x= range(num_of_items), y= accs)
    if modelname == 'bilstm':
        fig.savefig(fname= "images/bilstm_loss_acc.png", dpi=300)
    elif modelname == 'cnn':
        fig.savefig(fname= "images/cnn_loss_acc.png", dpi=300)
        