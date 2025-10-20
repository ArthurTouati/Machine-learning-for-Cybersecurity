import load_data as ld
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def treated_data():
    data_dict = ld.load_data()
    # Extract the 'arr' array from the dictionary
    data_array = data_dict['arr']

    # Separate features (images) and labels
    x = [item[0] for item in data_array]  # Images
    y = [item[1] for item in data_array]  # Labels

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, x_test):
    """
    Preprocess the image data for GRU models

    Args:
        x_train: Training images
        x_test: Testing images

    Returns:
        Preprocessed training and testing data
    """
    # Convert lists to numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # Reshape if needed (assuming images are 2D)
    if len(x_train.shape) == 3:  # If images are 2D (height, width)
        # Treat each row as a time step for GRU
        # No need to reshape, GRU can take 3D input (samples, timesteps, features)
        pass
    elif len(x_train.shape) == 4:  # If images are 3D (height, width, channels)
        # Reshape to (samples, height*width, channels)
        samples, height, width, channels = x_train.shape
        x_train = x_train.reshape(samples, height*width, channels)

        samples, height, width, channels = x_test.shape
        x_test = x_test.reshape(samples, height*width, channels)

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, x_test

def gru_svm():
    """
    Implement a hybrid GRU-SVM model for malware image classification

    The model uses a GRU network to extract features from the images,
    then feeds these features to an SVM classifier for the final prediction.

    Returns:
        Trained GRU-SVM model and evaluation metrics
    """
    # Load the treated data
    x_train, y_train, x_test, y_test = treated_data()

    # Preprocess the data
    x_train_processed, x_test_processed = preprocess_data(x_train, x_test)

    # Convert labels to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Get the input shape for the GRU model
    if len(x_train_processed.shape) == 3:
        timesteps, features = x_train_processed.shape[1], x_train_processed.shape[2]
    else:
        # If the data is 2D, treat each row as a timestep
        timesteps, features = x_train_processed.shape[1], 1
        x_train_processed = x_train_processed.reshape(x_train_processed.shape[0], timesteps, features)
        x_test_processed = x_test_processed.reshape(x_test_processed.shape[0], timesteps, features)

    print(f"Input shape: {x_train_processed.shape}")

    # Build the GRU model for feature extraction with 5 hidden layers and a dropout rate of 0.15
    gru = Sequential([
        # Define input_shape in the first layer for Sequential models
        GRU(256, return_sequences=True, input_shape=(timesteps, features)),
        Dropout(0.15),
        GRU(128, return_sequences=True),
        Dropout(0.15),
        GRU(96, return_sequences=True),
        Dropout(0.15),
        GRU(64, return_sequences=True),
        Dropout(0.15),
        GRU(32, return_sequences=False),
        Dropout(0.15),
        Dense(32, activation='softmax'),
    ])

    # Compile the GRU model with the learning rate set to 1e-3
    gru.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    gru.summary()

    # Train the GRU model
    history = gru.fit(
        x_train_processed, y_train,
        epochs=100,
        batch_size=256,
        validation_split=0.1,
        verbose=1
    )

    # Extract features using the trained GRU model.
    # The output of the trained 'gru' model is the feature vector from the last Dense layer.
    print("Extracting features using the trained GRU model...")
    train_features = gru.predict(x_train_processed)
    test_features = gru.predict(x_test_processed)

    print(f"GRU extracted features shape: {train_features.shape}")

    # Train SVM on the extracted features
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(train_features, y_train)

    # Make predictions
    y_pred = svm.predict(test_features)

    # Evaluate the model
    accu = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    repo = classification_report(y_test, y_pred)

    print(f"GRU-SVM Model Accuracy: {accu:.4f}")
    print(f"GRU-SVM Model Precision: {prec:.4f}")
    print(f"GRU-SVM Model Recall: {rec:.4f}")
    print(f"GRU-SVM Model F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(repo)

    # Create a table with metrics
    metrics_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Data Points', 'Epochs'],
        'Value': [f"{accu:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", 
                 f"{len(x_train_processed)} (train) / {len(x_test_processed)} (test)", "100"]
    })

    print("\nMetrics Table:")
    print(metrics_table.to_string(index=False))

    # Save metrics table to CSV
    metrics_table.to_csv('gru_svm_metrics.csv', index=False)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('gru_svm_confusion_matrix.png')
    plt.show()

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('GRU Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('GRU Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gru_svm_training_history.png')
    plt.show()

    # Plot training accuracy depending on a time step
    plt.figure(figsize=(10, 6))

    # Create a model to get intermediate outputs at each time step
    time_step_model = tf.keras.Model(
        inputs=gru.input,
        outputs=[layer.output for layer in gru.layers if isinstance(layer, GRU)]
    )

    # Get outputs at each time step for a sample of training data
    sample_size = min(100, len(x_train_processed))  # Use a smaller sample to avoid memory issues
    time_step_outputs = time_step_model.predict(x_train_processed[:sample_size])

    # Plot accuracy at each GRU layer (time step)
    accuracies = []
    layer_names = []

    for i, output in enumerate(time_step_outputs):
        # For the last layer, we need to reshape
        if i == len(time_step_outputs) - 1:
            features = output
        else:
            # For intermediate layers with return_sequences=True, take the last time step
            features = output[:, -1, :]

        # Train a simple classifier on these features
        temp_svm = SVC(kernel='linear', C=1.0)
        temp_svm.fit(features, y_train[:sample_size])

        # Predict on the same data (just to get an idea of how discriminative the features are)
        temp_pred = temp_svm.predict(features)
        temp_acc = accuracy_score(y_train[:sample_size], temp_pred)

        accuracies.append(temp_acc)
        layer_names.append(f"GRU Layer {i+1}")

    plt.bar(layer_names, accuracies)
    plt.title('Training Accuracy at Each GRU Layer')
    plt.xlabel('GRU Layer')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('gru_svm_timestep_accuracy.png')
    plt.show()

    return gru, svm, accu, prec, rec, f1, metrics_table, repo

if __name__ == "__main__":
    # Run the GRU-SVM model
    print("Running GRU-SVM model for malware image classification...")
    gru_model, svm_model, accuracy, precision, recall, f1_score, metrics_table, report = gru_svm()
    print("GRU-SVM model execution completed.")

    # Display the metrics table again
    print("\nFinal Metrics Table:")
    print(metrics_table.to_string(index=False))
