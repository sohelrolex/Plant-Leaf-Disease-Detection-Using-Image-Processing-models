# Plant Leaf Disease Detection Using Image Processing

## 1. Abstract
Plant diseases affect crop quality and yield. Early detection of leaf diseases can help farmers take timely action. This project presents a plant leaf disease detection system using image processing and machine learning. The system processes plant leaf images, extracts color, texture, and edge features, and classifies the image into healthy or diseased categories. HSV color histograms, Local Binary Pattern texture features, and Canny edge features are used for feature extraction. A Random Forest classifier is trained using a plant leaf image dataset. A Streamlit web interface is also developed for easy image upload and prediction.

## 2. Problem Statement
Manual plant disease detection requires expert knowledge and may be time-consuming. Farmers may not identify early symptoms correctly. Therefore, an automated image processing-based system is useful for detecting plant leaf diseases.

## 3. Objectives
- To collect plant leaf images from a dataset.
- To preprocess leaf images using image processing techniques.
- To extract color, texture, and edge features.
- To train a machine learning model for disease classification.
- To evaluate the model using accuracy and confusion matrix.
- To develop a simple web application for prediction.

## 4. Existing System
Traditional plant disease detection is performed by visual inspection. This method depends on expert knowledge and may not be available to all farmers.

## 5. Proposed System
The proposed system takes a leaf image as input and uses image processing to extract important features. These features are classified using a machine learning model to detect the disease class.

## 6. Dataset
The PlantVillage dataset can be used. It contains images of healthy and diseased plant leaves across multiple crop species.

## 7. Image Processing Techniques Used
- Image resizing
- Gaussian blur
- HSV color conversion
- Color histogram extraction
- Local Binary Pattern texture extraction
- Canny edge detection

## 8. Methodology
1. Dataset collection
2. Image preprocessing
3. Feature extraction
4. Model training
5. Model testing
6. Result analysis
7. Web app development

## 9. Algorithm
1. Input leaf image
2. Resize image
3. Apply Gaussian blur
4. Convert image to HSV
5. Extract color histogram features
6. Extract LBP texture features
7. Extract edge density using Canny
8. Train Random Forest classifier
9. Predict disease class
10. Display output

## 10. System Architecture

Input Leaf Image
        ↓
Image Preprocessing
        ↓
Feature Extraction
        ↓
Machine Learning Classifier
        ↓
Disease Prediction
        ↓
Result Display

## 11. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## 12. Applications
- Smart farming
- Crop monitoring
- Plant disease screening
- Agriculture decision support
- Educational computer vision project

## 13. Advantages
- Easy to understand
- Suitable for CVVP subject
- Uses clear image processing techniques
- Works with public dataset
- Includes user interface

## 14. Limitations
- Accuracy depends on dataset quality
- Background and lighting affect prediction
- It is not a replacement for expert agricultural advice
- Feature-based ML may be less accurate than CNN

## 15. Future Scope
- Use CNN or transfer learning
- Add real-time mobile camera detection
- Add disease treatment recommendation
- Deploy as web or mobile app
- Improve segmentation of infected leaf regions

## 16. Conclusion
This project successfully demonstrates plant leaf disease detection using image processing and machine learning. The system extracts color, texture, and edge features from leaf images and classifies them using Random Forest. It is simple, explainable, and suitable for CVVP mini project submission.

## 17. Viva Explanation
My project is Plant Leaf Disease Detection using Image Processing. First, the input leaf image is resized and noise is reduced using Gaussian blur. Then the image is converted into HSV color space because disease symptoms are visible through color changes. After that, color histogram, LBP texture features, and Canny edge features are extracted. These features are given to a Random Forest classifier, which predicts the plant disease class. The output is displayed using a Streamlit web application.

## 18. References
1. PlantVillage dataset
2. OpenCV documentation
3. Scikit-learn documentation
4. Local Binary Pattern texture feature method
