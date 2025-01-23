config = {
    'BATCH': 32,                      # Batch size
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    'EPOCHS': 10,                     # Number of epochs
    'LEARNING_RATE': 0.001,           # Learning rate
    'WEIGHT_DECAY': 1e-5              # Weight decay for optimizer
}
