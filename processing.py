# processing.py
import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

# pip freeze > requirements.txt
# pip install -r requirements.txt

# Constants
IMAGE_NUMBER = 2000  # Number of images to process
IMAGE_SIZE = 224
ZIP_FILE = "rsna-pneumonia-detection-challenge.zip"  # Path to the zip file
EXTRACT_DIR = "data"  # Directory to extract the zip file contents
SAVE_DIR = f"processed_data/processed_data_{IMAGE_NUMBER}-{IMAGE_SIZE}"  # Directory to save processed data
TEST_DIR = f"test_images_{IMAGE_SIZE}.npy"

# Ensure output directories exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Extract the zip file
def extract_zip(zip_file, extract_to):
    """
    Extract a zip file to a specified directory.
    
    Parameters:
        zip_file (str): Path to the zip file.
        extract_to (str): Path to the extraction directory.
    """
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_file} to {extract_to}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print(f"Dataset already extracted at {extract_to}.")

# Function to load and preprocess images
def load_images(dicom_dir, labels, num_samples, target_size=(IMAGE_SIZE, IMAGE_SIZE), augment=False):
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    print(f"Total DICOM files found: {len(dicom_files)}")

    datagen = ImageDataGenerator(
        rotation_range=30,  # Increased rotation range
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Add vertical flip for more variation
        fill_mode="nearest",
        brightness_range=[0.6, 1.4],  # Broaden the brightness range
        contrast_range=[0.8, 1.2]  # Add contrast adjustment

    ) if augment else ImageDataGenerator()

    for file_name in tqdm(dicom_files[:num_samples], desc="Processing DICOM Files"):
        file_path = os.path.join(dicom_dir, file_name)
        try:
            dicom = pydicom.dcmread(file_path)
            img = dicom.pixel_array
            img = img / np.max(img)

            img_resized = np.array(Image.fromarray(img).resize(target_size))
            img_resized = np.stack([img_resized] * 3, axis=-1)

            patient_id = os.path.splitext(file_name)[0]
            label = labels.loc[labels['patientId'] == patient_id, 'Target'].values[0]

            img_resized = datagen.random_transform(img_resized)
            yield img_resized, label

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

def load_test_images():
    # Define a function to load and process individual DICOM images
    def load_dicom_image(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
        # Load the DICOM file
        dicom_data = pydicom.dcmread(img_path)
        
        # Extract the pixel data (assuming the pixel data is in the 'pixel_array' field)
        img_array = dicom_data.pixel_array
        
        # Check if the image is grayscale (2D array) and convert it to a 3D array (RGB format)
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = np.expand_dims(img_array, axis=-1)  # Add a channel dimension (1 channel)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array / np.max(img_array)  # Normalize to range [0, 1]
        
        # Ensure the image is in the correct dtype for image processing
        img_array = img_array.astype(np.float32)  # Convert to float32 for better precision
        
        # Resize the image to the target size (e.g., 224x224)
        img_resized = np.array(Image.fromarray(img_array.squeeze()).resize(target_size))  # .squeeze() removes single-dimensional entries

        # Check if the resized image is still 2D (grayscale)
        if img_resized.ndim == 2:
            img_resized = np.expand_dims(img_resized, axis=-1)  # Add a channel dimension if needed
        
        # Repeat the grayscale channel to simulate RGB (3 channels)
        img_resized = np.repeat(img_resized, 3, axis=-1)  # Repeat grayscale channel 3 times
        
        return img_resized

    # Directory containing the test DICOM images
    test_image_dir = os.path.join(EXTRACT_DIR, "stage_2_test_images")

    # List all DICOM images in the test directory
    test_image_filenames = os.listdir(test_image_dir)

    # Print the total number of images in the directory
    print(f"Total test images found: {len(test_image_filenames)}")

    # Print the number of images being loaded
    print(f"Loading {len(test_image_filenames)} test images...")

    # Load and preprocess the test images with a progress bar
    test_images = []

    # Wrap the loop with tqdm for progress tracking
    for img_name in tqdm(test_image_filenames, desc="Processing test images", unit="image"):
        img_path = os.path.join(test_image_dir, img_name)
        
        # Load the DICOM image and preprocess it
        img_array = load_dicom_image(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        test_images.append(img_array)

    # Convert the list of images into a NumPy array for the model
    test_images = np.array(test_images)

    print("Saving test images...")
    # Save the processed images to a .npy file
    np.save(TEST_DIR, test_images)
    print(f"Test images saved successfully as '{TEST_DIR}'.")

def analyze_pixel_distribution(images):
    """
    Analyze the distribution of pixel values in the given images.

    Parameters:
        images (numpy.ndarray): Array of images.
    """
    # Flatten the images to a 1D array for easier analysis
    flat_images = images.reshape(-1)

    # # Use numpy's histogram function to get bin counts and bin edges
    # bin_counts, bin_edges = np.histogram(flat_images, bins=50, density=True)

    # # Print the bin counts and corresponding pixel value ranges
    # for i in range(len(bin_counts)):
    #     print(f"Pixel value range: ({bin_edges[i]:.4f}, {bin_edges[i + 1]:.4f}) - Frequency: {bin_counts[i]:.4f}")

    plt.hist(flat_images, bins=50, density=True)
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.title('Pixel Value Distribution')
    plt.show()

def analyze_label_distribution(labels):
    """
    Analyze the distribution of labels using a bar plot.

    Parameters:
        labels (numpy.ndarray or pandas.Series): Array or Series of labels.
    """
    label_counts = pd.Series(labels).value_counts()
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution')
    plt.show()

# Main data preparation function
def prepare_data():
    # Step 1: Extract the ZIP file
    extract_zip(ZIP_FILE, EXTRACT_DIR)

    # Step 2: Define paths
    dicom_dir = os.path.join(EXTRACT_DIR, "stage_2_train_images")
    labels_file = os.path.join(EXTRACT_DIR, "stage_2_train_labels.csv")

    # Step 3: Verify the extracted files
    if not os.path.exists(dicom_dir):
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    # Step 4: Load Test images
    # load_test_images() # uncomment this to load the test images!

    # Step 5: Load labels
    labels = pd.read_csv(labels_file)
    print(f"Total labels: {len(labels)}")
    print(f"Label distribution:\n{labels['Target'].value_counts()}")
    # Add EDA for label distribution
    analyze_label_distribution(labels['Target'])

    # Step 6: Load and preprocess images
    images, labels_filtered = zip(*load_images(dicom_dir, labels, IMAGE_NUMBER))
    images = np.array(images, dtype=np.float32)
    labels_filtered = np.array(labels_filtered, dtype=np.int32)

    print(f"Total images processed: {len(images)}")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels_filtered.shape}")

    # Add EDA for pixel value distribution
    analyze_pixel_distribution(images)

    # Step 7: Split and handle class imbalance
    X_train, X_val, y_train, y_val = train_test_split(images, labels_filtered, test_size=0.2, stratify=labels_filtered)
    print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights:\n{class_weight_dict}")

    # Step 8: Save validation sets
    print("Saving Validation sets...")
    np.save(os.path.join(SAVE_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(SAVE_DIR, "y_val.npy"), y_val)

    # Step 9: Apply SMOTE for the training set
    print("Applying SMOTE...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
    X_train_res = X_train_res.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

    print(f"Resampled training set class distribution:\n{pd.Series(y_train_res).value_counts()}")

    # Step 10: Save training sets
    print("Saving Training sets...")
    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train_res)
    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train_res)
    print("Saved Training sets successfully!")

    print("Data preparation complete!")

# Run the data preparation pipeline
prepare_data()
