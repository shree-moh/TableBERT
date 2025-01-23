import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import ADASYN
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import confusion_matrix
import argparse
import matplotlib.pyplot as plt
from pyswarm import pso
import lime
import lime.lime_text
from sklearn.metrics import f1_score
from custom_bert import CustomBERTForSequenceClassification


# Function to calculate evaluation metrics
def calculate_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    # PD: Probability of Detection (True Positive Rate)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    # PF: Probability of False Alarm
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    # Balance: Balance of Detection and False Alarm
    balance = 1 - np.sqrt((1 - PD) ** 2 + PF ** 2) / np.sqrt(2)
    # FI: False Inclusion Rate (instead of False Alarm Rate)
    FI = (cm[0, 1] + cm[1, 1]) / cm.sum() if cm.sum() > 0 else 0
    # FIR: False Inclusion Rate as a ratio to PD
    FIR = (PD - FI) / PD if PD > 0 else 0
    return PD, PF, balance, FIR


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


# Custom Dataset Class for BERT
class TableBERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = " ".join([f"{col}: {val}" for col, val in row.items() if col != 'class'])
        label = int(row['class'])
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), label


# PSO Hyperparameter Optimization Function
def objective_function(hyperparams):
    learning_rate = hyperparams[0]
    batch_size = int(hyperparams[1])

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for inputs, masks, labels in trainloader:
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

    # Calculate evaluation metrics like F1 score
    def calculate_F1_score(model, testloader):
        y_true = []
        y_pred = []

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for inputs, masks, labels in testloader:
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

                outputs = model(input_ids=inputs, attention_mask=masks)
                _, predicted = torch.max(outputs.logits, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Calculate the F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        return f1

    # Return a valid numerical value (f1 score)
    f1_score_value = calculate_F1_score(model, testloader)
    return f1_score_value if f1_score_value is not None else float('inf')


# LIME for Model Interpretability
def explain_prediction(text, model, tokenizer):
    # Move model to the correct device
    model.eval().to(device)  # Ensure model is on the correct device

    # Tokenize the input text (handling both single and multiple texts)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # LIME explanation requires a function that returns probabilities
    def predict_proba(inputs):
        with torch.no_grad():
            # If inputs is a list, process each item
            if isinstance(inputs, list):
                # If it's a list, run the tokenizer for each item
                inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                    device)

            # Ensure everything is on the same device
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Ensure inputs are on the same device

            # Get model outputs
            outputs = model(**inputs)  # Pass the inputs correctly as a dictionary
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs.cpu().numpy()  # Move the result back to CPU

    # Initialize LIME explainer
    explainer = lime.lime_text.LimeTextExplainer(class_names=['clean', 'buggy'])
    explanation = explainer.explain_instance(text, predict_proba, num_features=10)

    # Visualize the explanation (or handle it as needed)
    explanation.show_in_notebook(text=True)


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
    train_dataset = TableBERTDataset(train_data, tokenizer)
    test_dataset = TableBERTDataset(test_data, tokenizer)
    trainloader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False)

    # Load BERT model with class weights (to address imbalance)
    class_weights = torch.tensor([2.0, 1.0]).to(device)  # Increased weight for the minority class (buggy)

    if opt.checkpoint:
        model = BertForSequenceClassification.from_pretrained(opt.checkpoint, num_labels=len(label_encoder.classes_))
    else:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              num_labels=len(label_encoder.classes_))

    model = model.to(device)

    # Run PSO to optimize hyperparameters (e.g., learning rate, batch size)
    lb = [1e-6, 16]  # Lower bounds (learning rate, batch size)
    ub = [1e-4, 64]  # Upper bounds (learning rate, batch size)
    best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=5)

    # Update model with optimized hyperparameters
    learning_rate = best_params[0]
    batch_size = int(best_params[1])

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training the model
    for epoch in range(opt.epochs):
        model.train()
        for batch_idx, (inputs, masks, labels) in enumerate(trainloader):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

        # Evaluate after each epoch
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, masks, labels in testloader:
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

                outputs = model(input_ids=inputs, attention_mask=masks)
                _, predicted = torch.max(outputs.logits, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Calculate metrics
        PD, PF, balance, FIR = calculate_metrics(y_true, y_pred)
        print(f"Epoch {epoch + 1}/{opt.epochs}, PD: {PD:.3f}, PF: {PF:.3f}, Balance: {balance:.3f}, FIR: {FIR:.3f}")

    # Save model if required
    if opt.save_model:
        model.save_pretrained(f"final_model_{opt.run_name}")

    # Evaluate final model with LIME for interpretability
    text_example = "Example sentence to explain."
    explain_prediction(text_example, model, tokenizer)

