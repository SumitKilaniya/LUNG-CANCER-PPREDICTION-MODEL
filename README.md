# LUNG-CANCER-PPREDICTION-MODEL

📄 Lung Cancer Prediction Model Documentation
This repository contains the code and documentation for a machine learning project focused on predicting the presence of Lung Cancer using various health and lifestyle indicators. The entire workflow, from data loading to model evaluation, is executed within a Jupyter/Colab notebook environment.

🚀 Project Goal and Libraries
Objective
The primary goal is to build and evaluate several classification models to accurately predict the binary outcome (YES or NO) for the LUNG CANCER target variable.

Dependencies
The following Python libraries are essential for running the code and performing the analysis :

Library	Primary Use
numpy	Fundamental numerical computing.
pandas	Data manipulation and analysis (e.g., handling DataFrames).
matplotlib.pyplot	Basic data visualization.
seaborn	Advanced statistical data visualization.
scikit-learn (sklearn)	Machine learning models, preprocessing, and metrics.

Export to Sheets
💾 Data Management
Data Source
The dataset, representing a survey on lung cancer factors, is loaded from a CSV file named 'survey lung cancer.csv'.

Dataset Dimensions
The imported dataset (data) is a pandas DataFrame with 309 entries (rows) and 16 columns (features).


Missing Value Analysis
A null-check confirmed that there are no missing values across any of the 16 columns. All 309 entries in every column are non-null.


Feature Details
The dataset includes the following features, categorized by their data type :


Column Name	Dtype	Description
GENDER	Object	Patient's gender (a nominal categorical feature).
LUNG CANCER	Object	Target variable (YES/NO).
AGE	Int64	Patient's age (continuous numerical, often treated as discrete counts).
13 Binary/Ordinal Features	Int64	Smoking, Yellow Fingers, Anxiety, Peer Pressure, Chronic Disease, Fatigue, Allergy, Wheezing, Alcohol Consuming, Coughing, Shortness of Breath, Swallowing Difficulty, and Chest Pain.

Export to Sheets
⚙️ Data Preprocessing Pipeline
The raw data requires several steps to be transformed into a format suitable for machine learning algorithms.

1. Categorical Encoding
The nominal categorical features were converted to numerical representations using Scikit-learn's LabelEncoder .

The features encoded were GENDER and the target variable LUNG CANCER.

2. Feature and Target Separation
The final data set was split into the feature matrix X (all columns except LUNG CANCER) and the target vector y (LUNG CANCER).

3. Data Splitting
The data was partitioned into training and testing sets to validate model performance:


Split Ratio: 80% for training (X 
train
​
 ,y 
train
​
 ) and 20% for testing (X 
test
​
 ,y 
test
​
 ).


Reproducibility: A random_state of 42 was used to ensure the split is reproducible.

4. Feature Scaling
The features in the training data were standardized using StandardScaler to normalize the range of independent variables . This is crucial for models like Logistic Regression and SVC:

The scaler was fitted only on the training data (X 
train
​
 ).

The scaler was then used to transform both the training and testing data (X 
test
​
 ).

🤖 Model Development and Evaluation
Three distinct classification models were trained and assessed.

1. Logistic Regression

Model: LogisticRegression() 


Training Accuracy: 92.71% 


Test Accuracy: 96.77% 

2. Support Vector Classifier (SVC)

Model: SVC() 


Training Accuracy: 94.74% 


Test Accuracy: 96.77% 


Overall Accuracy: The final calculated accuracy on the test set was 0.90774.


Performance Metrics: The classification report showed high performance, with a weighted average precision, recall, and f1-score of 0.97.

3. K-Nearest Neighbors (KNN)

Model: KNeighborsClassifier() 


This model was implemented to explore distance-based classification. An example boundary plot was generated illustrating a classifier with K=5.



📈 Visualizations
Key visualizations were generated during the exploratory phase:


Target Distribution: A count plot showed the distribution of the target variable LUNG CANCER, highlighting an imbalance where YES cases significantly outnumbered NO cases.


Age Distribution: A plot illustrated the count distribution across different AGE groups.

Feature Relationships: Plots were created to visualize the relationship between:


LUNG CANCER and SMOKING (using bar and violin plots).



ANXIETY (using a scatter plot).


CHRONIC DISEASE (using a bar plot)
