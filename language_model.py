import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import split_train_val_dataloader
from evaluation import calculate_confusion_matrix, class_accuracy, class_f1_score

def train(model, train_dataloader, num_epochs:int, lr:float, class_weights:torch.FloatTensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight= class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr= lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        #    T_max= num_epochs*5,
                                                        #    eta_min= lr_floor)
    print(criterion)
    print(optimizer)
    # print(scheduler)

    epoch_loss, epoch_acc = [], []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_running_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0
        train_loaders, val_loaders = split_train_val_dataloader(train_dataloader)
        loop_ctr = 0
        for train_dl, val_dl in zip(train_loaders, val_loaders):
            loop_ctr += 1
            for batch in tqdm(train_dl):
                inputs = batch['padded'].to(device)
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct_predictions += (predicted == labels).sum().item()
                train_total_predictions += labels.size(0)
            # scheduler.step()
            train_epoch_loss = train_running_loss / (len(train_dl.dataset)*loop_ctr)
            train_epoch_accuracy = train_correct_predictions / train_total_predictions
            # Validation phase
            model.eval()
            val_correct_predictions = 0
            val_total_predictions = 0
            with torch.no_grad():
                for batch in val_dl:
                    val_inputs = batch['padded'].to(device)
                    val_labels = batch['label'].to(device) 
                    val_outputs = model(val_inputs)
                    _, val_predicted = torch.max(val_outputs, 1)
                    val_correct_predictions += (val_predicted == val_labels).sum().item()
                    val_total_predictions += val_labels.size(0)
            val_epoch_accuracy = val_correct_predictions / val_total_predictions
            epoch_loss.append(train_epoch_loss)
            epoch_acc.append(val_epoch_accuracy)
            print(f'Epoch {epoch+1}/{num_epochs}, Iter {loop_ctr}/5 '
                f'Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy*100:.3f}%, '
                f'Val Accuracy: {val_epoch_accuracy*100:.3f}%')
            
    return model, epoch_loss, epoch_acc

def evaluate(model, test_dataloader):
    print(f"{'Evaluating model on test data':-^100}")
    cm = None
    for batch in tqdm(test_dataloader):
        inputs = batch['padded']
        labels = batch['label']
        cm_helper = calculate_confusion_matrix(inputs, labels, model)
        if cm is None:
            cm = cm_helper
        else:
            cm = np.add(cm, cm_helper)
        
    accuracies = class_accuracy(cm)
    f1_scores = class_f1_score(cm)
    average_accuracy = np.mean(accuracies)
    average_f1 = np.mean(f1_scores)

    print(f"Average acccuracy: {average_accuracy*100:.2f}%")
    print(f"Average F1: {average_f1*100:.2f}%")
    
    return cm