# HeartDiseasePrediction

Heart Disease Prediction


This project predicts the likelihood of heart disease based on clinical and demographic data. Using machine learning models, we analyze various health parameters to classify whether a patient is at risk of heart disease.
📋 Table of Contents

    Project Overview
    Technologies Used
    Dataset
    Model Approach
    Installation
    Usage
    Results
    Contributing
    License

📖 Project Overview

Cardiovascular diseases are one of the leading causes of mortality worldwide. Early prediction of heart disease can significantly improve outcomes and save lives. This project leverages machine learning models to predict the presence of heart disease based on a patient’s health metrics such as age, cholesterol levels, blood pressure, etc.
💻 Technologies Used

    Python 3.12
    Libraries and Frameworks:
        Pandas
        NumPy
        Scikit-learn
        Matplotlib
        Seaborn

📂 Dataset

The dataset used in this project is the Heart Disease Dataset from the UCI Machine Learning Repository. It contains 14 clinical and demographic features used to predict the presence of heart disease.
Dataset Features:

    Age: Age of the patient
    Sex: Gender of the patient (1 = male, 0 = female)
    cp: Chest pain type (0-3)
    trestbps: Resting blood pressure (in mm Hg)
    chol: Serum cholesterol (in mg/dl)
    fbs: Fasting blood sugar (>120 mg/dl, 1 = true, 0 = false)
    restecg: Resting electrocardiographic results (0-2)
    thalach: Maximum heart rate achieved
    exang: Exercise-induced angina (1 = yes, 0 = no)
    oldpeak: ST depression induced by exercise relative to rest
    slope: Slope of the peak exercise ST segment (0-2)
    ca: Number of major vessels colored by fluoroscopy (0-3)
    thal: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
    target: Presence of heart disease (1 = disease, 0 = no disease)

🧠 Model Approach
1. Data Preprocessing

    Handling Missing Values: Filled or removed missing values.
    Feature Scaling: Applied Min-Max Scaling or Standardization to normalize features.
    Train-Test Split: Split the dataset into training (80%) and testing (20%) sets.

2. Model Selection

Several classification models were trained and evaluated:

    Logistic Regression
    K-Nearest Neighbors (KNN)
    Support Vector Machine (SVM)
    Decision Tree Classifier

3. Model Evaluation

Models were evaluated using metrics like:

    Accuracy
    Precision
    Recall
    F1-Score
    ROC-AUC Curve

🛠 Installation

    Clone the repository:

git clone https://github.com/yourusername/HeartDiseasePrediction.git  
cd HeartDiseasePrediction  

Install dependencies:

    pip install -r requirements.txt  

    Download the dataset and place it in the data/ folder. You can use the UCI Heart Disease Dataset.

🚀 Usage

    Train the Model:
    Train a classification model using the train.py script:

python train.py  

Evaluate the Model:
Evaluate the trained model’s performance on the test dataset:

python evaluate.py  

Make Predictions:
Use the trained model to predict heart disease risk for new patients:

    python predict.py  

    Interactive Notebook:
    Use the HeartDiseasePrediction.ipynb Jupyter Notebook for an interactive approach to data analysis, model training, and evaluation.

📊 Results

The best performing model achieved the following metrics:
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	80.0%	0.86	0.85	0.72	0.80
KNN	93.0%	0.93	0.94	0.94	0.94
Support Vector Machine	82.4%	0.90	0.74	0.92	0.84


🤝 Contributing

Contributions are welcome! To contribute to this project:

    Fork the repository.
    Create a new branch: git checkout -b feature-name.
    Commit your changes: git commit -m 'Add feature'.
    Push to the branch: git push origin feature-name.
    Submit a pull request.

📝 License

This project is licensed under the MIT License.
