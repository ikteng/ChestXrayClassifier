# predictions.py
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Constants for configuring the data and model paths
IMAGE_NUMBER = 2000
IMAGE_SIZE = 224
EPOCHS = 30
datagen = 30

DATA_DIR = f"processed_data/processed_data_{IMAGE_NUMBER}-{IMAGE_SIZE}-{datagen}"
DENSENET_MODEL_DIR = f"models/densenet/densenet_model-{IMAGE_NUMBER}-{IMAGE_SIZE}-{EPOCHS}-{datagen}.keras"

# Load the test data
print("Loading the test data...")
X_test = np.load(f"test_images_{IMAGE_SIZE}.npy")
print("Test data loaded successfully.")

# Load the pre-trained models
print("Loading the pre-trained models...")
# model = load_model(MODEL_DIR)
densenet_model = load_model(DENSENET_MODEL_DIR)
# resnet_model = load_model(RESENET_MODEL_DIR)
print("Pre-trained models loaded successfully.")

def run_prediction():
    model = densenet_model
    model_name = "densnet_model"

    print(f"Generating predictions using {model_name}...")
    test_predictions = model.predict(X_test)
    pred_class = (test_predictions > 0.5).astype(int)
    print(f"Predictions using {model_name} generated successfully.")

    # Generate patient IDs based on index (you can change this if you have other ID generation logic)
    patient_ids = [f"patient_{i}" for i in range(len(X_test))]

    # Prepare data for the new file with detailed predictions
    prediction_data = []
    print(f"Preparing detailed prediction data for {model_name} predictions...")
    for i in range(len(X_test)):
        patient_id = patient_ids[i]
        predicted_class = pred_class[i][0]
        probability = test_predictions[i][0]
        prediction_data.append([patient_id, predicted_class, probability])

    print("Creating a DataFrame with detailed prediction data for single model predictions...")
    columns = ["PatientID", "PredictedClass", "Probability"]
    prediction_df = pd.DataFrame(prediction_data, columns=columns)

    print("Saving detailed prediction data for single model predictions to a file...")
    prediction_df.to_csv(f"{model_name}_predictions.csv", index=False)
    print(f"File '{model_name}_predictions.csv' with detailed predictions has been created successfully!")

def run_ensemble_predictions():
    print("Generate weighted ensemble predictions...")
    models = [densenet_model]
    weights = [0.5, 0.5]
    print("Calculating weighted ensemble predictions...")
    ensemble_pred_weighted = ensemble_predictions_weighted(models, X_test, weights)
    print("Weighted ensemble predictions calculated successfully.")

    # Generate patient IDs based on index (you can change this if you have other ID generation logic)
    patient_ids = [f"patient_{i}" for i in range(len(X_test))]

    # Display sample predictions
    print("Showing sample predictions:")
    num_samples_to_show = 5  # You can adjust this number to show more or fewer samples
    for i in range(min(num_samples_to_show, len(X_test))):
        patient_id = patient_ids[i]
        predicted_class = 1 if ensemble_pred_weighted[i][0] >= 0.5 else 0
        probability = ensemble_pred_weighted[i][0]
        print(f"Patient ID: {patient_id}, Predicted Class: {predicted_class}, Probability: {probability:.4f}")

    # Prepare data for the new file with detailed predictions
    prediction_data = []
    print("Preparing detailed prediction data...")
    for i in range(len(X_test)):
        patient_id = patient_ids[i]
        predicted_class = 1 if ensemble_pred_weighted[i][0] >= 0.5 else 0
        probability = ensemble_pred_weighted[i][0]
        prediction_data.append([patient_id, predicted_class, probability])

    print("Creating a DataFrame with detailed prediction data...")
    columns = ["PatientID", "PredictedClass", "Probability"]
    prediction_df = pd.DataFrame(prediction_data, columns=columns)

    print("Saving detailed prediction data to a file...")
    prediction_df.to_csv("predictions.csv", index=False)
    print("File 'predictions.csv' with detailed predictions has been created successfully!")

# Generate weighted ensemble predictions based on the provided weights for each model.
def ensemble_predictions_weighted(models, X_test, weights):
    print("Generating individual predictions for weighted ensemble...")
    predictions = [model.predict(X_test, verbose=0) for model in models]
    predictions = np.array(predictions)
    print("Individual predictions generated. Now reshaping weights for calculation...")
    weights = np.array(weights).reshape(-1, 1, 1)
    print("Calculating weighted ensemble prediction using the provided weights...")
    result = np.sum(predictions * weights, axis=0) / np.sum(weights)
    print("Weighted ensemble prediction calculation completed.")
    return result

if __name__ == "__main__":
    run_prediction()
    # run_ensemble_predictions()
    print("Predictions Completed!")