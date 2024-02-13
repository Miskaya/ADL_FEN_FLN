import os
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import pickle
from model2 import FEN, FLN
import copy
import pandas as pd
from sklearn.metrics import f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dataset(X, y, time_steps, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        x = X.iloc[i : (i + time_steps)].values
        labels = y.iloc[i : i + time_steps]
        Xs.append(x)
        ys.append(stats.mode(labels)[0])
    Xs = np.swapaxes(Xs, 1, 2)
    return np.array(Xs), np.array(ys).reshape(-1, 1)

def training_loop(train_loader, val_loader, fen_model, fln_model, epochs, device, fen_lr, fln_lr, patience):
    criterion = nn.CrossEntropyLoss()
    fen_optimizer = optim.Adam(fen_model.parameters(), lr=fen_lr, weight_decay=1e-5)
    fln_optimizer = optim.Adam(fln_model.parameters(), lr=fln_lr, weight_decay=1e-5)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    val_acc_history = []
    val_loss_history = []
    
    for epoch in range(epochs):
        fen_model.train()
        fln_model.train()

        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            fen_optimizer.zero_grad()
            fln_optimizer.zero_grad()

            fen_outputs = [fen_model(inputs[:, i, :].unsqueeze(1)) for i in range(inputs.shape[1])]
            fen_outputs = torch.cat(fen_outputs, dim=1)
            fln_output = fln_model(fen_outputs.permute(0, 2, 1))
            loss = criterion(fln_output, labels)
            loss.backward()
            fen_optimizer.step()
            fln_optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(fln_output, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        fen_model.eval()
        fln_model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                fen_outputs = [fen_model(inputs[:, i, :].unsqueeze(1)) for i in range(inputs.shape[1])]
                fen_outputs = torch.cat(fen_outputs, dim=1)

                fln_output = fln_model(fen_outputs.permute(0, 2, 1))
                loss = criterion(fln_output, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(fln_output, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_fen_model_state = copy.deepcopy(fen_model.state_dict())
            best_fln_model_state = copy.deepcopy(fln_model.state_dict())
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    print(f"Best Validation Loss: {best_val_loss:.4f} with Accuracy: {best_val_acc:.4f}%")
    
    return fen_model, fln_model, best_fen_model_state, best_fln_model_state, best_val_loss, best_val_acc


def evaluate_model(test_loader, fen_model, fln_model, device):
    criterion = nn.CrossEntropyLoss()
    total_test_loss = 0.0
    correct_test = 0
    total_test = 0

    all_labels = []
    all_predictions = []

    fen_model.eval()
    fln_model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            fen_outputs = [fen_model(inputs[:, i, :].unsqueeze(1)) for i in range(inputs.shape[1])]
            fen_outputs = torch.cat(fen_outputs, dim=1)

            fln_output = fln_model(fen_outputs.permute(0, 2, 1))
            loss = criterion(fln_output, labels)
            total_test_loss += loss.item()
            _, predicted = torch.max(fln_output, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    test_loss = total_test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    micro_f1 = f1_score(all_labels, all_predictions, average='macro')
    
    return test_loss, test_accuracy, micro_f1

def split_data_by_label(data):
    train_data = []
    val_data = []
    test_data = []

    labels = data['label'].unique()
    for label in labels:
        data_label = data[data['label'] == label]
        train, temp = train_test_split(data_label, test_size=0.3, random_state=7)
        val, test = train_test_split(temp, test_size=0.5, random_state=7)
        train_data.append(train)
        val_data.append(val)
        test_data.append(test)

    return pd.concat(train_data), pd.concat(val_data), pd.concat(test_data)

fen_weights_path = './fen_weights.pth'
pretrained_fen_model = FEN().to(device)
pretrained_fen_model.load_state_dict(torch.load(fen_weights_path, map_location=device))

root_dir = './segmented_data/'
pkl_files = [file for file in os.listdir(root_dir) if file.endswith('modified_combined_sensor_data.pkl')]
chosen_file = pkl_files[0]
setup_seed(1)
print(f"Processing file: {chosen_file}")

with open(os.path.join(root_dir, chosen_file), 'rb') as f:
    data = pickle.load(f)

subjects = data['subject'].unique()
train_subjects, temp_subjects = train_test_split(subjects, test_size=0.3, random_state=7)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=7)
print(train_subjects, temp_subjects, val_subjects, test_subjects)
train_data = data[data['subject'].isin(train_subjects)]
val_data = data[data['subject'].isin(val_subjects)]
test_data = data[data['subject'].isin(test_subjects)]

time_steps = 400
step = 200

# train_data, val_data, test_data = split_data_by_label(data)

X_train, y_train = create_dataset(train_data.drop(columns=['label', 'subject']), train_data['label'], time_steps, step)
X_val, y_val = create_dataset(val_data.drop(columns=['label', 'subject']), val_data['label'], time_steps, step)
X_test, y_test = create_dataset(test_data.drop(columns=['label', 'subject']), test_data['label'], time_steps, step)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long).squeeze(1))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long).squeeze(1))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long).squeeze(1))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

fen_model = copy.deepcopy(pretrained_fen_model)

for param in fen_model.parameters():
    param.requires_grad = False

epochs = 50
print(X_train.shape)
d_model = int(64 * X_train.shape[1]) 

dim_feedforward = d_model * 1
nhead = 2
num_encoder_layers = 1
fen_lr = 0.0001
fln_lr = 0.0001

num_classes = len(np.unique(y_train))
print(d_model, dim_feedforward, num_classes)


fln_model = FLN(d_model, nhead, num_encoder_layers, dim_feedforward, num_classes).to(device)
fen_model, fln_model, best_fen_model_state, best_fln_model_state, best_val_loss, best_val_acc = training_loop(train_loader, val_loader, fen_model, fln_model, epochs, device, fen_lr, fln_lr, 10)

fen_model.load_state_dict(best_fen_model_state)
fln_model.load_state_dict(best_fln_model_state)

fen_weights_filename = f'lime_fen_weights.pth'
fln_weights_filename = f'lime_fln_weights.pth'
torch.save(best_fen_model_state, fen_weights_filename)
torch.save(best_fln_model_state, fln_weights_filename)
# test_loss, test_accuracy = evaluate_model(test_loader, fen_model, fln_model, device)
# print(f"File: {chosen_file}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
test_loss, test_accuracy, micro_f1 = evaluate_model(test_loader, fen_model, fln_model, device)
print(f'Test Loss for {chosen_file}: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, F1-score: {micro_f1:.4f}')