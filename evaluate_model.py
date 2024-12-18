# evaluate_model.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import random
from tensorflow.keras.preprocessing import image
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set environment variable to disable GPU (useful if no GPU is available or to avoid using GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMAGE_NUMBER = 2000
IMAGE_SIZE = 224
EPOCHS = 30

DATA_DIR = f"processed_data/processed_data_{IMAGE_NUMBER}-{IMAGE_SIZE}" 
MODEL_DIR = f"models/densenet/densenet_model-{IMAGE_NUMBER}-{IMAGE_SIZE}-{EPOCHS}.keras"

# Load the pre-trained model that was saved during training
print("Loading Model...")
model = load_model(MODEL_DIR)  # Load the model from the file
print("Model Loaded Successfully!")  # Confirmation message once the model is loaded

# Summary of the model
model.summary()

def run_evaluation():
   """Evaluate the model on the validation dataset."""
   # Load model
   print("Loading Model...")
   model = load_model(MODEL_DIR)
   print(f"Model path: ", MODEL_DIR)
   print("Model Loaded Successfully!")

   # Load validation dataset
   print("Loading Validation Data...")
   X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
   y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
   print(f"Validation Data Loaded: {X_val.shape}, {y_val.shape}")

   # Generate predictions
   print("Generating Predictions on Validation Data...")
   val_predictions = model.predict(X_val)
   pred_class = (val_predictions > 0.5).astype(int)

   print("Sample Predictions:")
   for i in range(5):  # Show a few predictions
      print(f"Predicted: {pred_class[i][0]}, Probability: {val_predictions[i][0]:.4f}, True label: {y_val[i]}")

   # Evaluate the model
   print("Evaluating Model...")
   evaluation_metrics = model.evaluate(X_val, y_val, verbose=1)
   print(f"Evaluation Results:\nLoss = {evaluation_metrics[0]:.4f}, Accuracy = {evaluation_metrics[1]:.4f}")

   print("\nClassification Report:")
   cr = classification_report(y_val, pred_class, target_names=["No Pneumonia", "Pneumonia"])
   print(cr)

   auc_score = roc_auc_score(y_val, val_predictions)
   print(f"AUC Score: {auc_score:.4f}")

   # Calculate additional metrics
   print("Confusion Matrix:")
   cm = confusion_matrix(y_val, pred_class)
   print(cm)

run_evaluation()