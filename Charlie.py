import argparse
import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help="Directory containing image dataset (JPEG format)")
args = parser.parse_args()

# Helper function to load images and assign labels based on folder names
def load_image_data(directory):
    images = []
    labels = []
    label_map = {}  # Maps folder names to integer labels
    current_label = 0  # Start labeling folders from 0

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            label_map[folder_name] = current_label
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if img_file.endswith(".jpg"):
                    try:
                        # Load and preprocess image
                        img = Image.open(img_path).convert("L")  # Convert to grayscale
                        img = img.resize((512, 512))  # Resize to 64x64 pixels
                        img_array = np.array(img).flatten()  # Flatten to 1D
                        images.append(img_array)
                        labels.append(current_label)  # Assign the current folder's label
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
            current_label += 1

    return np.array(images), np.array(labels), label_map

# Load the dataset
print("Loading dataset...")
X, Y, label_map = load_image_data(args.trainingdata)
print(f"Loaded {len(X)} images across {len(label_map)} classes: {label_map}")

# Reverse the label map for easy lookup
reverse_label_map = {v: k for k, v in label_map.items()}

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Evaluate the model
print("Evaluating model...")
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate classification report
class_names = [reverse_label_map[i] for i in range(len(label_map))]
classification_report_str = classification_report(Y_test, Y_pred, target_names=class_names)
print("Classification Report:")
print(classification_report_str)

# Start an MLflow run
with mlflow.start_run():
    # Log the model
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="RandomForestImageClassifier")
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log classification report
    mlflow.log_text(classification_report_str, artifact_file="classification_report.txt")
    
    # Confusion Matrix
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # ROC Curve
    Y_scores = model.predict_proba(X_test)
    plt.figure(figsize=(8, 6))
    for idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(Y_test == idx, Y_scores[:, idx])
        auc = roc_auc_score(Y_test == idx, Y_scores[:, idx])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    roc_curve_path = "roc_curve.png"
    plt.savefig(roc_curve_path)
    plt.close()
    mlflow.log_artifact(roc_curve_path)


    # Analyze errors
    misclassified_indices = np.where(Y_pred != Y_test)[0]
    for idx in misclassified_indices[:10]:  # Log first 10 misclassifications
        img = X_test[idx].reshape(512, 512)  # Assuming images are 64x64
        true_label = reverse_label_map[Y_test[idx]]
        pred_label = reverse_label_map[Y_pred[idx]]
        plt.imshow(img, cmap="gray")
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        error_img_path = f"error_{idx}.png"
        plt.savefig(error_img_path)
        plt.close()
        mlflow.log_artifact(error_img_path)


print("Model and metrics logged with MLflow.")

