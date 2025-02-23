import pickle

# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

# Function to load a saved model
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model
