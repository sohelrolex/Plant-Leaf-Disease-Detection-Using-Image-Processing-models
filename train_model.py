from pathlib import Path
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 128
MAX_IMAGES_PER_CLASS = 300


def preprocess_image(image_path):
    """
    Image preprocessing:
    1. Read image
    2. Resize image
    3. Apply Gaussian blur
    4. Convert image into HSV color space
    """
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.GaussianBlur(image, (3, 3), 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return image, hsv


def extract_color_features(hsv_image):
    """
    Extract color histogram features from HSV image.
    HSV is useful because diseased leaf areas often show color changes.
    """
    features = []

    for channel in range(3):
        hist = cv2.calcHist([hsv_image], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    return np.array(features)


def extract_texture_features(bgr_image):
    """
    Extract texture features using Local Binary Pattern.
    Diseased leaves usually have spots, patches, or texture changes.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    radius = 2
    points = 8 * radius

    lbp = local_binary_pattern(gray, points, radius, method="uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, points + 3),
        range=(0, points + 2)
    )

    hist = hist.astype("float")
    hist = hist / (hist.sum() + 1e-7)

    return hist


def extract_edge_features(bgr_image):
    """
    Extract edge density using Canny edge detection.
    Disease spots can increase edge/texture variation.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.sum(edges > 0) / edges.size

    return np.array([edge_density])


def extract_features(image_path):
    data = preprocess_image(image_path)

    if data is None:
        return None

    bgr_image, hsv_image = data

    color_features = extract_color_features(hsv_image)
    texture_features = extract_texture_features(bgr_image)
    edge_features = extract_edge_features(bgr_image)

    return np.concatenate([color_features, texture_features, edge_features])


def load_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError("dataset folder not found. Please create dataset folder.")

    X = []
    y = []

    class_folders = [folder for folder in DATASET_DIR.iterdir() if folder.is_dir()]

    if len(class_folders) == 0:
        raise FileNotFoundError("No class folders found inside dataset folder.")

    print("Classes found:")
    for folder in class_folders:
        print("-", folder.name)

    for class_folder in class_folders:
        image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg")) + list(class_folder.glob("*.png"))

        image_files = image_files[:MAX_IMAGES_PER_CLASS]

        for index, image_path in enumerate(image_files):
            features = extract_features(image_path)

            if features is not None:
                X.append(features)
                y.append(class_folder.name)

            if index % 100 == 0:
                print(f"Processing {class_folder.name}: {index}/{len(image_files)}")

    return np.array(X), np.array(y)


def main():
    X, y = load_dataset()

    print("Total images:", len(X))

    if len(X) == 0:
        raise ValueError("No images loaded. Please check dataset folder.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    print(report)

    with open(RESULT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("Plant Leaf Disease Detection Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=90)
    plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "confusion_matrix.png")
    plt.close()

    joblib.dump(model, MODEL_DIR / "plant_leaf_model.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")

    print("Model saved successfully in models folder.")
    print("Results saved successfully in results folder.")


if __name__ == "__main__":
    main()
