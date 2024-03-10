import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, train_dataloader, val_dataloader, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    epoch_loss, epoch_acc = [], []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_running_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0
        
        for batch in tqdm(train_dataloader):
            inputs = batch['padded'].to(device)
            labels = batch['label'].to(device)
            # print(inputs.size())
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            train_running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            train_correct_predictions += (predicted == labels).sum().item()
            train_total_predictions += labels.size(0)
        
        train_epoch_loss = train_running_loss / len(train_dataloader.dataset)
        train_epoch_accuracy = train_correct_predictions / train_total_predictions
        
        # Validation phase
        model.eval()
        val_correct_predictions = 0
        val_total_predictions = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                val_inputs = batch['padded'].to(device)
                val_labels = batch['label'].to(device)
                
                val_outputs = model(val_inputs)
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct_predictions += (val_predicted == val_labels).sum().item()
                val_total_predictions += val_labels.size(0)
        
        val_epoch_accuracy = val_correct_predictions / val_total_predictions
        
        epoch_loss.append(train_epoch_loss)
        epoch_acc.append(val_epoch_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.4f}, '
              f'Val Accuracy: {val_epoch_accuracy:.4f}')
        
    return model, epoch_loss, epoch_acc