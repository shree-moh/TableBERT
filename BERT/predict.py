import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from transformers import BertForSequenceClassification, BertTokenizer
from lime.lime_tabular import LimeTabularExplainer
import joblib

# Load the saved model, scaler, and label encoder
model_save_path = 'bert_model.pt'
scaler_save_path = 'scaler.pkl'
label_encoder_save_path = 'label_encoder.pkl'

# Load the scaler and label encoder
scaler = joblib.load(scaler_save_path)
label_encoder = joblib.load(label_encoder_save_path)

# Load the trained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and prepare the data
data = pd.read_csv('EQ.csv')
X = data.drop(columns=['class'])
y = data['class']

# Normalize the data using the loaded scaler
X_test_normalized = scaler.transform(X)

# LIME explainer setup
explainer = LimeTabularExplainer(
    X_test_normalized, mode="classification", training_labels=y, feature_names=X.columns
)

# Function to predict probabilities using the trained BERT model
def predict_proba(model, data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data).float().to(device)
        outputs = model(inputs)
        y_probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return y_probs.cpu().numpy()

# Pick an instance for interpretation
instance = X_test_normalized[0]  # Test on first instance

# Explain the instance
explanation = explainer.explain_instance(instance, lambda x: predict_proba(model, x), num_features=5)
explanation.show_in_notebook()
