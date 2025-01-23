import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from imblearn.over_sampling import ADASYN
from sklearn.metrics import confusion_matrix
import argparse


# Function to calculate evaluation metrics
def calculate_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    # Extract TP, FP, TN, FN from confusion matrix
    TP = cm[1, 1]  # True Positives
    FP = cm[0, 1]  # False Positives
    TN = cm[0, 0]  # True Negatives
    FN = cm[1, 0]  # False Negatives

    # PD: Probability of Detection (True Positive Rate)
    PD = TP / (TP + FN) if (TP + FN) > 0 else 0

    # PF: Probability of False Alarm
    PF = FP / (FP + TN) if (FP + TN) > 0 else 0

    # Balance: Balance of Detection and False Alarm
    balance = 1 - np.sqrt((1 - PD) ** 2 + PF ** 2) / np.sqrt(2)

    # FI: False Inclusion Rate (instead of False Alarm Rate)
    FI = (FP + TP) / cm.sum() if cm.sum() > 0 else 0

    # FIR: False Inclusion Rate as a ratio to PD
    FIR = (PD - FI) / PD if PD > 0 else 0

    return PD, PF, balance, FIR


# VanillaTableBERTDataset Class
class VanillaTableBERTDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        """
        Args:
            data (pd.DataFrame): The dataframe containing your features and labels.
            tokenizer (BertTokenizer): The tokenizer for encoding the text.
            max_len (int): The maximum length of the tokenized input.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.features = data.drop(columns=['class']).astype(str).values
        self.labels = data['class'].values

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data sample.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, and the label.
        """
        # Convert the feature row into a single string (depending on your tabular data structure)
        feature_row = " ".join(self.features[idx])

        # Tokenize the text using the BERT tokenizer
        encoding = self.tokenizer.encode_plus(
            feature_row,  # The input text (tabular data as a single string)
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=self.max_len,  # Ensure the length does not exceed max_len
            padding='max_length',  # Pad the sequences to max_len
            truncation=True,  # Truncate if the sequence is longer than max_len
            return_attention_mask=True,  # Return attention mask
            return_tensors='pt'  # Return PyTorch tensors
        )

        # Extract input_ids and attention_mask from encoding
        input_ids = encoding['input_ids'].squeeze(0)  # Remove the extra batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)  # Remove the extra batch dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert label to tensor

        # Return a dictionary containing input_ids, attention_mask, and label
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }


# Function to parse command-line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--dset_seed', default=5, type=int)
    parser.add_argument('--active_log', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a saved model checkpoint")
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_args()
    torch.manual_seed(opt.set_seed)
    np.random.seed(opt.set_seed)

    # Check if GPU is available and send model and data to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    data = pd.read_csv('EQ.csv')

    # Encode the 'class' column to numeric labels
    label_encoder = LabelEncoder()
    data['class'] = label_encoder.fit_transform(data['class'])
    print("Label mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

    # Split the data into training and testing sets
    y = data['class']
    X = data.drop(columns=['class'])
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42, stratify=y)

    # Normalize features
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train.drop(columns=['class']))
    X_test_normalized = scaler.transform(X_test.drop(columns=['class']))

    # Apply ADASYN for oversampling (more focused on generating synthetic samples for minority class)
    adasyn = ADASYN(random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_normalized, y_train)

    # Convert the resampled data back to DataFrame
    X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X_train.drop(columns=['class']).columns)
    y_train_resampled_df = pd.DataFrame(y_train_resampled, columns=['class'])
    train_data = pd.concat([X_train_resampled_df, y_train_resampled_df], axis=1)

    test_columns = [col for col in X_test.columns if col != 'class']
    test_data = pd.concat([pd.DataFrame(X_test_normalized, columns=test_columns), y_test.reset_index(drop=True)],
                          axis=1)

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset = VanillaTableBERTDataset(train_data, tokenizer)
    test_dataset = VanillaTableBERTDataset(test_data, tokenizer)
    trainloader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False)

    # Load BERT model with class weights (to address imbalance)
    class_weights = torch.tensor([2.0, 1.0]).to(device)  # Increased weight for the minority class (buggy)

    if opt.checkpoint:
        model = BertForSequenceClassification.from_pretrained(opt.checkpoint, num_labels=len(label_encoder.classes_))
    else:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              num_labels=len(label_encoder.classes_))

    model.to(device)

    # Ensuring the model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)

    # Using StepLR for learning rate scheduling
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Early stopping setup
    best_val_loss = float('inf')
    patience = 3
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(opt.epochs):
        model.train()
        running_loss = 0.0
        for batch in trainloader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Adjust learning rate
        scheduler.step()

        print(f"Epoch {epoch + 1}/{opt.epochs} - Loss: {running_loss / len(trainloader):.4f}")

        # Validation to check for early stopping
        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in testloader:
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        print(f"Validation Loss: {val_loss / len(testloader):.4f}")

        # Calculate and print custom metrics
        PD, PF, balance, FIR = calculate_metrics(y_true, y_pred)
        print(f"PD: {PD:.4f} | PF: {PF:.4f} | Balance: {balance:.4f} | FIR: {FIR:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if opt.save_model:
                torch.save(model.state_dict(), f"best_model_{opt.run_name}.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping...")
                break

    # Load best model and test
    if opt.save_model:
        model.load_state_dict(torch.load(f"best_model_{opt.run_name}.pt"))

    print("Training complete!")
