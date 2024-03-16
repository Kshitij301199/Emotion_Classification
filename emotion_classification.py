
import os
import sys
import json
import argparse

import pandas as pd
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.model_selection import train_test_split

from utils import tokenize, remove_stopwords, stemming, fit_tokenizer, save_model
from utils import pad_sequence, PandasDataset, get_batched_data, plot_loss_acc, get_class_weights
from language_model import train, evaluate
from evaluation import plot_confusion_matrix, class_accuracy, class_f1_score, class_wise_precision_recall

from model.BiLSTM_model import BiLSTMModel
from model.CNN_model import CNNTextClassifier


def main():
    parser = argparse.ArgumentParser(description='Command line interface for model training and prediction')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--model_name', type=str, help='Model name to train (required if --train|--evaluate is specified)')

    args = parser.parse_args()
    if args.train:
        if args.model_name.lower() == 'bilstm':
            model = BiLSTMModel()
        elif args.model_name.lower() == 'cnn':
            model = CNNTextClassifier()
        else:
            sys.exit(f"Enter a valid model name! {args.model_name} is not BiLSTM|CNN.")
        print(f"{'Pre-processing the data for training':-^100}")
        data = pd.read_csv("./data/text.csv").drop(columns="Unnamed: 0")
        data.drop_duplicates(inplace= True)
        print("Read data file!")
        data['tokens'] = data['text'].apply(tokenize)
        data['tokens_stemm'] = data['tokens'].apply(stemming).apply(remove_stopwords)
        data = data[~(data['tokens_stemm'].apply(len) == 0)]
        print("Data tokenized and stemmed!")
        
        x_train, x_test, y_train, y_test = train_test_split(data['tokens_stemm'], data['label'], test_size=0.2, random_state=42)
        class_weights = get_class_weights(data['label'])
        train_data = pd.concat((x_train,y_train), axis=1).reset_index()
        test_data = pd.concat((x_test,y_test), axis=1).reset_index()
        print(f"Train size: {len(train_data)}\tTest size: {len(test_data)}")
        
        tokenizer = fit_tokenizer(data['tokens_stemm'])
        tokenizer_json = tokenizer.to_json()
        # Save tokenizer JSON string to a file
        with open('tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json))
        print("Tokenizer saved to file.")
        
        train_data['padded'] = train_data['tokens_stemm'].apply(pad_sequence, tokenizer= tokenizer)
        test_data['padded'] = test_data['tokens_stemm'].apply(pad_sequence, tokenizer= tokenizer)
        print("Data prepared for model!")
        train_dataset = PandasDataset(train_data)
        test_dataset = PandasDataset(test_data)
        train_batched = get_batched_data(train_dataset, batch_size= 256)
        test_batched = get_batched_data(test_dataset, batch_size= 64)
        print(f"{'Starting Training':-^100}")
        
        print(model)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        model, losses, accs = train(model, train_batched, num_epochs= 4,
                                    learning_rate= 0.0005,
                                    class_weights= class_weights)
        plot_loss_acc(loss= losses, accs= accs, modelname= args.model_name)
        conf_mat = evaluate(model, test_batched)
        plot_confusion_matrix(conf_matrix= conf_mat, modelname= args.model_name)
        print("Results stored in images folder!")
        if args.model_name.lower() == 'bilstm':
            save_model(model, "./model/bilstm.pth")
        elif args.model_name.lower() == 'cnn':
            save_model(model, "./model/cnn.pth")
    
    elif args.evaluate:
        if args.model_name.lower() == 'bilstm':
            model = BiLSTMModel()    
        elif args.model_name.lower() == 'cnn':
            model = CNNTextClassifier()
        else:
            sys.exit(f"Enter a valid model name! {args.model_name} is not BiLSTM|CNN.")
        
        data = pd.read_csv("./data/text.csv").drop(columns="Unnamed: 0")
        data.drop_duplicates(inplace= True)
        print("Read data file!")
        
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(json.loads(tokenizer_json))

        _, x_test, _, y_test = train_test_split(data['text'], data['label'], test_size=0.5)
        test_data = pd.concat((x_test,y_test), axis=1).reset_index()
        test_data['tokens'] = test_data['text'].apply(tokenize)
        test_data['tokens_stemm'] = test_data['tokens'].apply(stemming).apply(remove_stopwords)
        test_data = test_data[~(test_data['tokens_stemm'].apply(len) == 0)]
        print("Data tokenized and stemmed!")
        
        test_data['padded'] = test_data['tokens_stemm'].apply(pad_sequence, tokenizer= tokenizer)
        print("Data prepared for model!")
        
        test_dataset = PandasDataset(test_data)
        test_batched = get_batched_data(test_dataset, batch_size= 128)
        
        conf_mat = evaluate(model, test_batched)
        accuracies = class_accuracy(conf_mat)
        f1_scores = class_f1_score(conf_mat)
        precisions, recalls = class_wise_precision_recall(conf_mat)
        
        output_df = pd.DataFrame({"Emotion":['Sadness','Joy','Love','Anger','Fear','Surprise'],
                                  'Accuracy':accuracies, 'F1_Score':f1_scores,
                                  'Precision':precisions, 'Recall':recalls})
        print(output_df)
        
    
if __name__ == '__main__':
    main()