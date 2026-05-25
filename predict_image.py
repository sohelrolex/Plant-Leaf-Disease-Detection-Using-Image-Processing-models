from pathlib import Path
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern

IMG_SIZE = 128


def preprocess_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Image could not be loaded.")

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.GaussianBlur(image, (3, 3), 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return image, hsv


def extract_color_features(hsv_image):
    features = []

    for channel in range(3):
        hist = cv2.calcHist([hsv_image], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    return np.array(features)


def extract_texture_features(bgr_image):
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
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.sum(edges > 0) / edges.size

    return np.array([edge_density])


def extract_features(image_path):
    bgr_image, hsv_image = preprocess_image(image_path)

    color_features = extract_color_features(hsv_image)
    texture_features = extract_texture_features(bgr_image)
    edge_features = extract_edge_features(bgr_image)

    return np.concatenate([color_features, texture_features, edge_features])


def predict_leaf_disease(image_path):
    model_path = Path("models/plant_leaf_model.pkl")
    encoder_path = Path("models/label_encoder.pkl")

    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("Model not found. Please run python train_model.py first.")

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    features = extract_features(image_path).reshape(1, -1)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    class_name = label_encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities)) * 100

    return class_name, confidence


if __name__ == "__main__":
    image_path = input("Enter leaf image path: ")

    disease, confidence = predict_leaf_disease(image_path)

    print("Prediction:", disease)
    print("Confidence:", round(confidence, 2), "%")
