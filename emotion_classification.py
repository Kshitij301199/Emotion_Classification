
import argparse
import pandas as pd
from utils import tokenize, remove_stopwords, stemming, fit_tokenizer
from utils import pad_sequence, PandasDataset, get_batched_data, plot_loss_acc
from language_model import train, evaluate
from evaluation import plot_confusion_matrix
from sklearn.model_selection import train_test_split

from model.BiLSTM_model import BiLSTMModel
from model.CNN_model import CNNTextClassifier


def main():
    parser = argparse.ArgumentParser(description='Command line interface for model training and prediction')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions using the trained model')
    parser.add_argument('--model_name', type=str, help='Model name to train (required if --train is specified)')

    args = parser.parse_args()
    if args.train:
        print(f"{'Pre-processing the data for training':-^100}")
        
        data = pd.read_csv("../data/text.csv").drop(columns="Unnamed: 0")
        data.drop_duplicates(inplace= True)
        
        data['tokens'] = data['text'].apply(tokenize)
        data['tokens_stemm'] = data['tokens'].apply(stemming).apply(remove_stopwords)
        data = data[~(data['tokens_stemm'].apply(len) == 0)]
        
        x_train, x_test, y_train, y_test = train_test_split(data['tokens_stemm'], data['label'], test_size=0.2, random_state=42)
        # 0.125 x 0.8 = 0.1
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125, random_state=42)
        
        train_data = pd.concat((x_train,y_train), axis=1).reset_index()
        val_data = pd.concat((x_val,y_val), axis=1).reset_index()
        test_data = pd.concat((x_test,y_test), axis=1).reset_index()
        
        tokenizer = fit_tokenizer(x_train)
        train_data['padded'] = train_data['tokens_stemm'].apply(pad_sequence, tokenizer= tokenizer)
        val_data['padded'] = val_data['tokens_stemm'].apply(pad_sequence, tokenizer= tokenizer)
        test_data['padded'] = test_data['tokens_stemm'].apply(pad_sequence, tokenizer= tokenizer)
        
        train_dataset = PandasDataset(train_data)
        val_dataset = PandasDataset(val_data)
        test_dataset = PandasDataset(test_data)
        
        train_batched = get_batched_data(train_dataset, batch_size= 256)
        val_batched = get_batched_data(val_dataset, batch_size= 64)
        test_batched = get_batched_data(test_dataset, batch_size= 64)
        
        if args.model_name == 'bilstm':
            model = BiLSTMModel()
        elif args.model_name == 'cnn':
            model = CNNTextClassifier()
        
        model, losses, accs = train(model, train_batched,
                                    val_batched, num_epochs= 10,
                                    learning_rate= 0.001)
        plot_loss_acc(loss= losses, accs= accs, modelname= args.model_name)

        conf_mat = evaluate(model, test_batched)
        plot_confusion_matrix(conf_matrix= conf_mat, modelname= args.model_name)
        print("Results stored in images folder!")
    
if __name__ == '__main__':
    main()