# Plant Leaf Disease Detection Using Image Processing

## Project Title
Plant Leaf Disease Detection Using Image Processing and Machine Learning

## Description
This is a simple and explainable CVVP mini project. The system detects whether a plant leaf is healthy or diseased, and can also classify the disease category if the dataset folders are class-wise.

The project uses:
1. Image preprocessing
2. Leaf image resizing
3. Gaussian blur
4. HSV color space conversion
5. Color histogram features
6. Texture features using Local Binary Pattern
7. Random Forest classifier
8. Streamlit web application

## Recommended Dataset
Use the PlantVillage dataset.

Dataset folder should be class-wise like this:

dataset/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___healthy/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___healthy/
└── ...

Each folder name becomes a class label.

## How to Run

### Step 1: Create virtual environment
```bash
python -m venv venv
```

### Step 2: Activate virtual environment
```bash
venv\Scripts\activate
```

### Step 3: Install libraries
```bash
python -m pip install -r requirements.txt
```

### Step 4: Put dataset
Create a folder named dataset and paste class folders inside it.

### Step 5: Train model
```bash
python train_model.py
```

### Step 6: Run Streamlit app
```bash
streamlit run app.py
```

## Output Files
After training, these files will be created:

models/plant_leaf_model.pkl
models/label_encoder.pkl
results/classification_report.txt
results/confusion_matrix.png

## Note
This project is for academic use only.
