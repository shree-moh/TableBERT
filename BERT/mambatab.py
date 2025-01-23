import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import warnings
from collections import Counter
from mamba_ssm import Mamba
from train_val import train_model  # Ensure `train_model` is implemented properly
from config import config  # Ensure config includes 'BATCH', 'device', etc.

warnings.filterwarnings('ignore')
# Utility to read data
def read_data(file_path):
    """Read dataset from CSV file."""
    data = pd.read_csv(C:\Users\aiselab\PycharmProjects\pythonProject\BERT\EQ.csv)
    x_data = data.drop('target', axis=1).values  # Replace 'target' with your label column
    y_data = data['target'].values
    return x_data, y_data

# Classification evaluation
def get_clf_eval(y_test, pred):
    CM = confusion_matrix(y_test, pred)
    TN, FP, FN, TP = CM.ravel()
    recall = TP / (TP + FN)
    FPR = FP / (TN + FP)  # False Positive Rate
    balance = 1 - (math.sqrt((0 - FPR) ** 2 + (1 - recall) ** 2) / math.sqrt(2))
    fi = (TP + FP) / (TP + TN + FP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    fir = (recall - fi) / recall if recall > 0 else 0
    return recall, FPR, balance, fir, accuracy, precision

# Oversampling
def sampling(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, list(y_train))
    return X_resampled, y_resampled
class TabularDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y


def test_result(model, dataloader, device):
    """
    Inference on the test set.
    """
    model.eval()
    sig = torch.nn.Sigmoid()
    all_test_labels = []
    all_test_output_probas = []

    for inputs, labels in dataloader['test']:
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = sig(model(inputs)).cpu().numpy()
            all_test_output_probas.extend(outputs)
            all_test_labels.extend(labels.cpu().numpy())

    all_test_preds = (np.array(all_test_output_probas) > 0.5).astype(int)
    PD, PF, balance, fir, accuracy, precision = get_clf_eval(all_test_labels, all_test_preds)
    auc = roc_auc_score(all_test_labels, all_test_output_probas)
    return PD, PF, balance, fir, accuracy, precision, auc


if __name__ == "__main__":
    dataset_path = 'your_file_path.csv'  # Replace with your dataset
    x_data, y_data = read_data(dataset_path)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    pd_result, pf_result, bal_result = [], [], []
    fir_result, precision_result, accuracy_result, auc_result = [], [], [], []

    fold = 0
    for train_index, test_index in kf.split(x_data, y_data):
        fold += 1
        print(f"Fold {fold}")

        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        print("Before sampling:", Counter(y_train))
        X_resampled, y_resampled = sampling(x_train, y_train)
        print("After sampling:", Counter(y_resampled))

        train_set = TabularDataset(np.array(X_resampled), np.array(y_resampled))
        test_set = TabularDataset(np.array(x_test), np.array(y_test))

        dataloader = {
            'train': DataLoader(train_set, batch_size=config['BATCH'], shuffle=True),
            'test': DataLoader(test_set, batch_size=config['BATCH'], shuffle=False)
        }

        # Initialize model
        model = Mamba(input_features=X_resampled.shape[1], n_class=1).to(config['device'])
        model = train_model(model, config, dataloader)

        # Test model
        PD, PF, balance, fir, accuracy, precision, auc = test_result(model, dataloader, config['device'])
        pd_result.append(PD)
        pf_result.append(PF)
        bal_result.append(balance)
        fir_result.append(fir)
        accuracy_result.append(accuracy)
        precision_result.append(precision)
        auc_result.append(auc)

    print("===================================================")
    print(f"Average Recall (PD): {np.mean(pd_result):.4f}")
    print(f"Average FPR (PF): {np.mean(pf_result):.4f}")
    print(f"Average Balance: {np.mean(bal_result):.4f}")
    print(f"Average FIR: {np.mean(fir_result):.4f}")
    print(f"Average Accuracy: {np.mean(accuracy_result):.4f}")
    print(f"Average Precision: {np.mean(precision_result):.4f}")
    print(f"Average AUROC: {np.mean(auc_result):.4f}")
