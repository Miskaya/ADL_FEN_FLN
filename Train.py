import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from scipy import stats
import pickle
from model import FEN, FLN
import time
from dataset_params import dataset_params
import copy
from sklearn.metrics import f1_score, precision_score, recall_score

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


def training_loop(dataset, train_loader, val_loader, fen_model, fln_model, epochs, device, fln_weights_dict, fen_lr, fln_lr, patience):
    if dataset in fln_weights_dict:
        fln_model.load_state_dict(fln_weights_dict[dataset])
        
    fen_optimizer = optim.Adam(fen_model.parameters(), lr=fen_lr)
    fln_optimizer = optim.Adam(fln_model.parameters(), lr=fln_lr)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    best_acc = 0.0

    for epoch in range(epochs):
        # train
        fen_model.train()
        fln_model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(train_loader):
            fen_optimizer.zero_grad()
            fln_optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            fen_outputs = []
            
            for signal in range(inputs.size(1)):
                signal_data = inputs[:, signal, :].unsqueeze(1)
                fen_output = fen_model(signal_data)
                fen_outputs.append(fen_output)

            fen_outputs = torch.cat(fen_outputs, dim=1)

            fln_output = fln_model(fen_outputs.permute(0, 2, 1))
            loss = criterion(fln_output, labels)

            loss.backward()
            fen_optimizer.step()
            fln_optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(fln_output, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        # eval
        fen_model.eval()
        fln_model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                fen_outputs = []
            
                for signal in range(inputs.size(1)):
                    signal_data = inputs[:, signal, :].unsqueeze(1)
                    fen_output = fen_model(signal_data)
                    fen_outputs.append(fen_output)

                fen_outputs = torch.cat(fen_outputs, dim=1)

                fln_output = fln_model(fen_outputs.permute(0, 2, 1))

                loss = criterion(fln_output, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(fln_output, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        print(f'Epoch {epoch + 1}/{epochs}, '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if val_acc >= best_acc:
            best_loss = val_loss
            best_acc = val_acc
            no_improvement_count = 0
            fln_weights_dict[dataset] = copy.deepcopy(fln_model.state_dict())
            fen_state = copy.deepcopy(fen_model.state_dict())
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    print(f"Best Validation Loss: {best_loss:.4f} with Accuracy: {best_acc:.4f}%")
    return fen_model, fln_model, fln_weights_dict, fen_state, best_acc, best_loss

def select_closest_half_d_model(d_model, options=[64, 128, 256, 512, 768, 1024, 1280, 1536, 1792]):
    half_d_model = d_model / 2
    closest = min(options, key=lambda x: abs(x - half_d_model))
    return closest
    
def evaluate_model(test_loader, fen_model, fln_model, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
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
            total_loss += loss.item()

            _, predicted = torch.max(fln_output, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    # print(total_loss,len(test_loader))
    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * total_correct / total_samples
    
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    
    return test_loss, test_accuracy, f1, precision, recall

def run_experiment(seed_num):
    fln_weights_dict = {}
    fen_state = None
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

    root_dir = './Datapool/'
    pkl_files = [file for file in os.listdir(root_dir) if file.endswith('.pkl') ]
    dataset_iteration_count = {os.path.splitext(file)[0]: 0 for file in pkl_files}
    best_accuracy_per_iteration = {os.path.splitext(file)[0]: [] for file in pkl_files}
    test_results = {os.path.splitext(file)[0]: {'accuracy': [], 'f1_score': [], 'precision': [], 'recall': []} for file in pkl_files}
    
    for _ in range(13):
        random.shuffle(pkl_files)
        print(pkl_files)
        for chosen_file in pkl_files:
            with open(os.path.join(root_dir, chosen_file), 'rb') as f:
                data = pickle.load(f)
            dataset_name = os.path.splitext(chosen_file)[0]
            dataset_iteration_count[dataset_name] += 1
                
            print(f"Processing dataset: {dataset_name}, Iteration: {dataset_iteration_count[dataset_name]}")
            subjects = data['subject'].unique()

            train_subjects, temp_subjects = train_test_split(subjects, test_size=0.3, random_state=42)
            val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)
            
            train_data = data[data['subject'].isin(train_subjects)]
            val_data = data[data['subject'].isin(val_subjects)]
            test_data = data[data['subject'].isin(test_subjects)]

            time_steps = 400
            step = 200

            X_train, y_train = create_dataset(train_data.drop(columns=['label', 'subject']), train_data['label'], time_steps, step)
            
            X_val, y_val = create_dataset(val_data.drop(columns=['label', 'subject']), val_data['label'], time_steps, step)
            
            X_test, y_test = create_dataset(test_data.drop(columns=['label', 'subject']), test_data['label'], time_steps, step)

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long).squeeze(1))
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long).squeeze(1))
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False)
            
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long).squeeze(1))
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

            d_model = dataset_params[dataset_name]['d_model']
            nhead = dataset_params[dataset_name]['nhead']
            dim_feedforward = dataset_params[dataset_name]['dim_feedforward']
            num_encoder_layers = dataset_params[dataset_name]['num_encoder_layers']
            fen_lr = dataset_params[dataset_name]['fen_lr']
            fln_lr = dataset_params[dataset_name]['fln_lr']
            epochs = dataset_params[dataset_name]['epoch']
        
            unique_labels = np.unique(y_train)
            num_classes = len(unique_labels)
            
            fen_model = FEN().to(device)
            if fen_state is not None:
                fen_model.load_state_dict(fen_state)
            
            fln_model = FLN(d_model, nhead, num_encoder_layers, dim_feedforward, num_classes).to(device)
            if dataset_name in fln_weights_dict:
                fln_model.load_state_dict(fln_weights_dict[dataset_name])
                
            fen_model, fln_model, fln_weights_dict, fen_state, best_acc, best_loss = training_loop(dataset_name, train_loader, val_loader, fen_model, fln_model, epochs, device, fln_weights_dict, fen_lr, fln_lr, 1000)
            best_acc_rounded = round(best_acc, 4)
            best_accuracy_per_iteration[dataset_name].append(best_acc_rounded)
            
            
            fen_model.load_state_dict(fen_state)
            fln_model.load_state_dict(fln_weights_dict[dataset_name])
            
            test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(test_loader, fen_model, fln_model, device)
            print(f'Test Loss for {dataset_name}: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, '
                f'F1-Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')          
            test_results[dataset_name]['accuracy'].append(test_accuracy)
            test_results[dataset_name]['f1_score'].append(test_f1)
            test_results[dataset_name]['precision'].append(test_precision)
            test_results[dataset_name]['recall'].append(test_recall)            

    for dataset, accuracies in best_accuracy_per_iteration.items():
        first_iteration_accuracy = accuracies[0]
        last_iteration_accuracy = accuracies[-1]
        print(f"Dataset: {dataset}, Best Accuracies per Iteration: {accuracies}")
        print(f"   First Iteration Accuracy: {first_iteration_accuracy:.2f}%, Last Iteration Accuracy: {last_iteration_accuracy:.2f}%")

    for dataset, metrics in test_results.items():
        accuracy_list = metrics['accuracy']
        print(f"Dataset: {dataset}, Test Accuracies: {accuracy_list}")
        first_accuracy = metrics['accuracy'][0]
        last_accuracy = metrics['accuracy'][-1]
        print(f"Dataset: {dataset}, First Iteration Accuracy: {first_accuracy}, Last Iteration Accuracy: {last_accuracy}")

    for dataset, accuracies in test_results.items():
        iterations = list(range(0, len(metrics['accuracy']) + 1))
        accuracies = [0] + metrics['accuracy']

        plt.plot(iterations, accuracies, marker='.', label=dataset)

    plt.xticks(range(0, len(iterations)))

    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy over Iterations for Each Dataset')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'final_result.png')

    return fen_state

for seed in range(1,2):
    print(f"Running experiment with seed: {seed}")
    setup_seed(seed)
    fen_state = run_experiment(seed)
    
    fen_weights_filename = f'fen_weights.pth'
    torch.save(fen_state, fen_weights_filename)