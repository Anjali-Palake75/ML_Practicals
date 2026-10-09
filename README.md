# Machine Learning Practical Experiments

This repository contains practical implementations of **Machine Learning and Data Science concepts using Python**.

The experiments cover data preprocessing, exploratory data analysis, regression, dimensionality reduction, classification, and clustering algorithms using Python and Scikit-learn.

## 📚 Experiments Included

| No. | Experiment | Dataset |
| --- | --- | --- |
| 1 | Data Preprocessing | Employee Dataset |
| 2 | Exploratory Data Analysis (EDA) | Titanic Dataset |
| 3 | Linear & Logistic Regression | USA Housing & Diabetes Datasets |
| 4 | Principal Component Analysis (PCA) | Iris Dataset |
| 5 | Support Vector Machine (SVM) for Classification | Iris Dataset |
| 6 | Decision Tree & K-Nearest Neighbour (KNN) | Iris Dataset |
| 7 | Comparison of Different Regression Algorithms | California Housing Dataset |
| 8 | Comparison of Different Classification Algorithms | Iris Dataset |
| 9 | K-Means Clustering | Mall Customers Dataset |

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 📂 Repository Structure

```text
MI LAB/
│
├── README.md
│
├── Exp1_Data_Pre-processing/
│   ├── Employee.csv
│   └── Data pre-processing.ipynb
│
├── Exp2_EDA/
│   ├── train.csv
│   └── EDA.ipynb
│
├── Exp3_Linear&Logistic_Regression/
│   ├── USA Housing.csv
│   ├── Diabetes.csv
│   └── Linear&Logistic_Regression.ipynb
│
├── Exp4_PCA/
│   ├── Iris.csv
│   └── PCA.ipynb
│
├── Exp5_SVMForClassification/
│   ├── iris.csv
│   └── SVM_Classification.ipynb
│
├── Exp6_Decision_Tree&KNN/
│   ├── iris.csv
│   └── Decision_Tree_KNN.ipynb
│
├── Exp7_Comparison of Different Regression Algorithms for Predictive Modeling/
│   └── Comparison_of_Different_Regression_Algorithms_for_Predictive_Modeling.ipynb
│
├── Exp8_Compare different classification algorithms/
│   ├── iris.csv
│   └── Compare_different_classification_algorithms_using_Precision_Recall,_Accuracy.ipynb
│
└── Exp9_K-Means Clustering/
    ├── Mall Customers.csv
    ├── K-Means_Clustering.ipynb
    └── Mall_Customers_with_Clusters.csv
```

## 🧪 Experiments

### Experiment 1 – Data Preprocessing

Performed data preprocessing on the **Employee Dataset** to improve data quality and prepare the dataset for machine learning.

The experiment includes:

- Handling missing values
- Removing duplicate records
- Detecting outliers using the IQR method
- Encoding categorical data
- Applying normalization
- Applying standardization

### Experiment 2 – Exploratory Data Analysis (EDA)

Performed Exploratory Data Analysis on the **Titanic Dataset** to understand the dataset, identify patterns, and analyze relationships between variables.

The experiment includes:

- Statistical analysis
- Data visualization
- Missing value analysis
- Distribution analysis
- Correlation analysis

### Experiment 3 – Linear & Logistic Regression

Implemented Linear Regression and Logistic Regression using two different datasets.

**Part A – Linear Regression**

**Dataset:** USA Housing Dataset

Linear Regression was implemented to predict house prices based on housing features.

The experiment includes:

- Loading and preparing the dataset
- Checking missing values and duplicate records
- Exploratory Data Analysis
- Defining features and target variables
- Splitting the dataset into training and testing sets
- Training the Linear Regression model
- Calculating model coefficients and intercept
- Predicting house prices
- Evaluating model performance
- Visualizing actual versus predicted prices

**Evaluation Metrics:**

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

**Output:**

```text
Linear Regression Evaluation
----------------------------
MSE       : 10089009300.89
RMSE      : 100444.06
MAE       : 80879.10
R2 Score  : 0.9180
```

**Observation:**

The Linear Regression model achieved an R² score of approximately 91.80%, indicating that it explained a large proportion of the variation in house prices on the test dataset.

**Part B – Logistic Regression**

**Dataset:** Diabetes Dataset

Logistic Regression was implemented to predict whether a person has diabetes based on medical and demographic features.

The experiment includes:

- Loading and preparing the dataset
- Checking missing values and duplicate records
- Exploratory Data Analysis
- Analyzing diabetes outcome distribution
- Defining input features and target variable
- Splitting the dataset using stratified sampling
- Applying StandardScaler for feature scaling
- Training the Logistic Regression model
- Predicting diabetes outcomes
- Calculating prediction probabilities
- Evaluating classification performance
- Visualizing the confusion matrix

**Evaluation Metrics:**

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

**Output:**

```text
Logistic Regression Evaluation
------------------------------
Accuracy  : 0.7143
Precision : 0.6087
Recall    : 0.5185
F1 Score  : 0.5600
ROC-AUC   : 0.8230
```

**Confusion Matrix:**

```text
[[82 18]
 [26 28]]
```

**Observation:**

The Logistic Regression model achieved approximately 71.43% accuracy and an ROC-AUC score of 0.8230 on the test dataset.

### Experiment 4 – Principal Component Analysis (PCA)

**Dataset:** Iris Dataset

Applied Principal Component Analysis to reduce the dimensionality of the Iris Dataset while preserving important information.

The original dataset contains four numerical features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The experiment includes:

- Loading and examining the dataset
- Checking missing values and duplicate records
- Separating features and target labels
- Standardizing the input features
- Applying PCA to reduce four dimensions to two
- Calculating explained variance
- Calculating cumulative explained variance
- Visualizing cumulative variance
- Comparing feature visualization before and after PCA
- Training Logistic Regression on PCA-transformed data
- Evaluating classification accuracy
- Visualizing the confusion matrix

**Explained Variance:**

```text
PC1 : 72.77%
PC2 : 23.03%
PC3 : 3.68%
PC4 : 0.52%
```

**Cumulative Explained Variance:**

```text
PC1       : 72.77%
PC1 + PC2 : 95.80%
PC1-PC3   : 99.48%
All PCs   : 100.00%
```

**Output:**

```text
PCA RESULT
------------------------------------------
Original dimensions       : 4
Reduced dimensions        : 2
PC1 variance              : 72.77%
PC2 variance              : 23.03%
Variance retained         : 95.80%
Classification accuracy  : 88.89%
```

**Confusion Matrix:**

```text
[[15  0  0]
 [ 0 14  1]
 [ 0  4 11]]
```

**Observation:**

PCA reduced the number of features from four to two while retaining 95.80% of the total variance. Logistic Regression achieved 88.89% classification accuracy using the reduced features.

### Experiment 5 – Support Vector Machine (SVM) for Classification

**Dataset:** Iris Dataset

Implemented Support Vector Machine to classify Iris flowers into three species: Setosa, Versicolor, and Virginica.

The experiment includes:

- Loading and examining the dataset
- Checking missing values
- Separating features and target labels
- Splitting the dataset into training and testing sets
- Applying StandardScaler for feature scaling
- Creating an SVM classifier using the RBF kernel
- Training the SVM model
- Predicting flower species
- Evaluating classification accuracy
- Generating a confusion matrix
- Generating a classification report
- Predicting the species of a new flower sample

**Important Concepts:**

- Hyperplane
- Support vectors
- Maximum margin
- Linear and non-linear classification
- Kernel functions
- RBF kernel
- Feature scaling
- Classification accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

**Model Configuration:**

```text
Algorithm : Support Vector Machine
Kernel    : RBF
C         : 1.0
Gamma     : scale
```

**Output:**

```text
Training Samples: 120
Testing Samples : 30

Accuracy: 100.0%
```

**Confusion Matrix:**

```text
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

**Classification Report:**

```text
              precision  recall  f1-score  support

setosa             1.00    1.00      1.00       10
versicolor         1.00    1.00      1.00        9
virginica          1.00    1.00      1.00       11

accuracy                              1.00       30
```

**New Sample Prediction:**

```text
Input Features: [5.1, 3.5, 1.4, 0.2]
Predicted Species: setosa
```

**Observation:**

The SVM classifier achieved 100% accuracy on the test dataset and correctly classified all 30 test samples.

### Experiment 6 – Decision Tree & K-Nearest Neighbour (KNN)

**Dataset:** Iris Dataset

Implemented and compared Decision Tree and K-Nearest Neighbour classification algorithms.

Algorithms implemented:

- Decision Tree Classifier
- K-Nearest Neighbour (KNN)

The experiment covers:

- Decision tree splitting
- Entropy
- Information Gain
- Gini Index
- K-Nearest Neighbour algorithm
- Euclidean distance
- Manhattan distance
- Feature scaling
- Classification accuracy
- Confusion Matrix
- Classification Report

**Output:**

```text
Model Comparison
-------------------------
Decision Tree Accuracy: 100.0 %
KNN Accuracy: 100.0 %
```

**Observation:**

Both algorithms achieved 100% accuracy in the recorded output.

### Experiment 7 – Comparison of Different Regression Algorithms

**Dataset:** California Housing Dataset

Compared multiple regression algorithms for predictive modeling using the California Housing Dataset.

Algorithms implemented:

- Linear Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression (SVR)

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

**Output:**

```text
Model Comparison
-----------------------------
Linear Regression -> MAE: 0.5332 | MSE: 0.5559 | R2: 0.5758
Polynomial Regression -> MAE: 0.4670 | MSE: 0.4643 | R2: 0.6457
Decision Tree -> MAE: 0.4547 | MSE: 0.4952 | R2: 0.6221
Random Forest -> MAE: 0.3275 | MSE: 0.2554 | R2: 0.8051
SVR -> MAE: 0.3973 | MSE: 0.3542 | R2: 0.7297
```

**Observation:**

Based on the recorded results, Random Forest Regression achieved the lowest MAE and MSE and the highest R² score among the compared models.

### Experiment 8 – Comparison of Different Classification Algorithms

**Dataset:** Iris Dataset

Implemented and compared multiple classification algorithms using the Iris Dataset.

Algorithms implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbour (KNN)
- Naive Bayes

The models were evaluated using:

- Precision
- Recall
- Accuracy

**Output:**

```text
Comparison of Classification Algorithms:

             Algorithm  Precision  Recall  Accuracy
0  Logistic Regression        1.0     1.0       1.0
1        Decision Tree        1.0     1.0       1.0
2        Random Forest        1.0     1.0       1.0
3                  SVM        1.0     1.0       1.0
4                  KNN        1.0     1.0       1.0
5          Naive Bayes        1.0     1.0       1.0
```

**Observation:**

All six algorithms achieved 100% precision, recall, and accuracy in the recorded output.

### Experiment 9 – K-Means Clustering

**Dataset:** Mall Customers Dataset

**Output Dataset:** Mall_Customers_with_Clusters.csv

Implemented K-Means Clustering to group customers into clusters based on the similarity of their features.

**The experiment includes:**

- Loading and examining the customer dataset
- Selecting relevant features
- Applying the K-Means Clustering algorithm
- Determining the number of clusters
- Assigning customers to clusters
- Visualizing the resulting clusters
- Saving the clustered dataset to a CSV file

**Important Concepts:**

**1. K-Means Clustering**

K-Means is an unsupervised machine learning algorithm that divides data points into K clusters based on feature similarity.

**2. Euclidean Distance**

Euclidean distance measures the distance between a data point and a cluster centroid.

Formula:

d(x, μ) = √Σ(xᵢ − μᵢ)²

**3. Within-Cluster Sum of Squares (WCSS)**

WCSS measures the sum of squared distances between data points and their assigned cluster centroids.

Formula:

WCSS = Σₖ₌₁ᴷ Σₓ∈Cₖ ||x − μₖ||²

**4. Elbow Method**

The Elbow Method helps identify a suitable number of clusters by examining how WCSS changes as the number of clusters increases.

**Output:**

The experiment generates the following output file:

```text
Mall_Customers_with_Clusters.csv
```

The file contains customer records along with their assigned cluster labels.

**Applications:**

- Customer segmentation
- Market analysis
- Customer behavior analysis
- Pattern recognition
- Grouping customers based on purchasing behavior

## 🎯 Learning Outcomes

Through these experiments, the following concepts are practiced:

- Data preprocessing and cleaning
- Handling missing values
- Removing duplicate records
- Outlier detection
- Data encoding
- Normalization and standardization
- Exploratory Data Analysis
- Statistical analysis and visualization
- Linear Regression
- Logistic Regression
- Principal Component Analysis
- Dimensionality reduction
- Support Vector Machine
- Decision Tree
- K-Nearest Neighbour
- Random Forest
- Polynomial Regression
- Support Vector Regression
- Classification algorithm comparison
- Regression algorithm comparison
- K-Means Clustering
- Feature scaling
- Customer segmentation
- Model evaluation
- Accuracy, Precision, and Recall
- F1 Score and ROC-AUC
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Confusion Matrix

## 👩‍💻 Author

**Anjali Palake**

B.Tech – Information Technology  
Government College of Engineering, Karad
