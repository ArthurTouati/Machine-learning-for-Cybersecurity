# from tensorflow.keras import Sequential

import load_data as ld
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

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
    Preprocess image data for a 2D CNN: ensure shape (N, H, W, 1) and normalize to [0, 1].
    """
    # Convert lists to numpy arrays
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)

    # Infer and fix dimensions
    def to_nhwc1(x):
        if x.ndim == 2:
            # Single image HxW -> 1,H,W,1
            x = x[None, ..., None]
        elif x.ndim == 3:
            # N,H,W -> N,H,W,1
            x = x[..., None]
        elif x.ndim == 4:
            # Assume already NHWC; if channels are not 1, keep as-is
            pass
        else:
            raise ValueError(f"Unexpected image array ndim={x.ndim}")
        return x

    x_train = to_nhwc1(x_train)
    x_test = to_nhwc1(x_test)

    # Normalize pixel values to [0, 1]
    x_train /= 255.0
    x_test /= 255.0

    return x_train, x_test

def l2_svm_loss(y_true, y_pred, C=10.0):
    # y_true may come as integer class ids; convert to one-hot with dynamic depth
    num_classes = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=num_classes)
    # Convert to -1/+1 targets as in SVMs
    y_true_pm = 2.0 * tf.cast(y_true_oh, y_pred.dtype) - 1.0
    # L2-hinge loss across all classes
    margins = 1.0 - y_true_pm * y_pred
    hinge = tf.reduce_mean(tf.square(tf.maximum(0.0, margins)))
    return C * hinge

def cnn_svm():
    """
    Implement a hybrid CNN-SVM model for malware image classification

    The model uses a CNN network to extract features from the images,
    then feeds these features to an SVM classifier for the final prediction.

    Returns:
        Trained CNN-SVM model and evaluation metrics
    """
    # Load the treated data
    x_train, y_train, x_test, y_test = treated_data()

    # Preprocess the data
    x_train_processed, x_test_processed = preprocess_data(x_train, x_test)

    # Convert labels to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Show the input shape for the CNN model
    print(f"Input shape: {x_train_processed.shape}")

    # Determine number of classes dynamically
    num_classes = np.unique(y_train).size

    # --- Definition of the CNN-SVM ---
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=x_train_processed.shape[1:]),
        tf.keras.layers.RandomZoom(0.1),

        tf.keras.layers.Conv2D(36, (5, 5), strides=1, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),

        tf.keras.layers.Conv2D(72, (5, 5), strides=1, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dropout(0.15),  # Drop 15 %

        # # Second hidden layer to meet the "2 hidden layer" requirement
        # tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        # tf.keras.layers.LeakyReLU(alpha=0.01),
        # tf.keras.layers.Dropout(0.15),  # Additional regularization to reduce overfitting

        tf.keras.layers.Dense(num_classes, activation='linear')
    ])

    # Compile the CNN model with the learning rate set to 1e-3
    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=l2_svm_loss,
                metrics=['accuracy']
                )

    # Print model summary
    cnn.summary()

    # Train the CNN model
    history = cnn.fit(
        x_train_processed, y_train,
        epochs=100,
        batch_size=256,
        validation_split=0.1,
        verbose=1
    )

    # Predict on test set
    print("Predicting on the test set...")
    y_pred = cnn.predict(x_test_processed)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Evaluate the model
    accu = accuracy_score(y_test, y_pred_labels)
    prec = precision_score(y_test, y_pred_labels, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_labels, average='weighted', zero_division=0)
    repo = classification_report(y_test, y_pred_labels, zero_division=0)

    print(f"CNN-SVM Model Accuracy: {accu:.4f}")
    print(f"CNN-SVM Model Precision: {prec:.4f}")
    print(f"CNN-SVM Model Recall: {rec:.4f}")
    print(f"CNN-SVM Model F1 Score: {f1:.4f}")
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
    metrics_table.to_csv('cnn_svm_metrics.csv', index=False)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('cnn_svm_confusion_matrix.png')
    plt.show()

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('cnn_svm_training_history.png')
    plt.show()

    return cnn, accu, prec, rec, f1, metrics_table, repo

if __name__ == "__main__":
    # Run the CNN-SVM model
    print("Running CNN-SVM model for malware image classification...")
    cnn_model, accuracy, precision, recall, f1_score, metrics_table, report = cnn_svm()
    print("CNN-SVM model execution completed.")

    # Display the metrics table again
    print("\nFinal Metrics Table:")
    print(metrics_table.to_string(index=False))