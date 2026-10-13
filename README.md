# Machine Learning Practical Experiments

This repository contains practical implementations of Machine Learning and Data Science concepts using Python. The experiments cover data preprocessing, exploratory data analysis, regression, dimensionality reduction, classification, clustering, and a basic artificial neural network.

## Table of Contents

- [About the Repository](#about-the-repository)
- [Technologies Used](#technologies-used)
- [Repository Structure](#repository-structure)
- [Experiments](#experiments)
  - [Experiment 1: Data Preprocessing](#experiment-1--data-preprocessing)
  - [Experiment 2: Exploratory Data Analysis (EDA)](#experiment-2--exploratory-data-analysis-eda)
  - [Experiment 3: Linear and Logistic Regression](#experiment-3--linear-and-logistic-regression)
  - [Experiment 4: Principal Component Analysis (PCA)](#experiment-4--principal-component-analysis-pca)
  - [Experiment 5: Support Vector Machine (SVM)](#experiment-5--support-vector-machine-svm-for-classification)
  - [Experiment 6: Decision Tree and KNN](#experiment-6--decision-tree-and-k-nearest-neighbour-knn)
  - [Experiment 7: Regression Algorithm Comparison](#experiment-7--comparison-of-different-regression-algorithms)
  - [Experiment 8: Classification Algorithm Comparison](#experiment-8--comparison-of-different-classification-algorithms)
  - [Experiment 9: K-Means Clustering](#experiment-9--k-means-clustering)
  - [Experiment 10: Single-Layer ANN](#experiment-10--iris-flower-classification-using-a-single-layer-artificial-neural-network-ann)
- [Installation and Execution](#installation-and-execution)
- [Learning Outcomes](#learning-outcomes)
- [Author](#author)

## About the Repository

The purpose of these practicals is to understand and implement fundamental Machine Learning techniques using Python and common data science libraries. Each experiment uses a dataset, applies an algorithm or analysis technique, and evaluates or visualizes the result where appropriate.

## Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Repository Structure

The folder and file names below describe the intended organization. Keep the actual names in your local repository consistent with your files.

```text
ML_Practicals/
├── README.md
├── Exp1_Data_Pre-processing/
│   ├── Employee.csv
│   └── Data pre-processing.ipynb
├── Exp2_EDA/
│   ├── train.csv
│   └── EDA.ipynb
├── Exp3_Linear&Logistic_Regression/
│   ├── USA Housing.csv
│   ├── Diabetes.csv
│   └── Linear&Logistic_Regression.ipynb
├── Exp4_PCA/
│   ├── Iris.csv
│   └── PCA.ipynb
├── Exp5_SVMForClassification/
│   ├── iris.csv
│   └── SVM_Classification.ipynb
├── Exp6_Decision_Tree&KNN/
│   ├── iris.csv
│   └── Decision_Tree_KNN.ipynb
├── Exp7_Comparison of Different Regression Algorithms for Predictive Modeling/
│   └── Comparison_of_Different_Regression_Algorithms_for_Predictive_Modeling.ipynb
├── Exp8_Compare different classification algorithms/
│   ├── iris.csv
│   └── Compare_different_classification_algorithms_using_Precision_Recall,_Accuracy.ipynb
├── Exp9_K-Means Clustering/
│   ├── Mall Customers.csv
│   ├── K-Means_Clustering.ipynb
│   └── Mall_Customers_with_Clusters.csv
└── Exp10_Single_Layer_ANN/
    ├── iris.csv
    └── single_layer_ann_iris.py
```

## Experiments

### Experiment 1 – Data Preprocessing

**Dataset:** Employee Dataset

**Objective:** Clean and transform the Employee Dataset so that it is suitable for further analysis and Machine Learning.

**Tasks performed:**
- Inspect the dataset structure and data types.
- Identify and handle missing values.
- Detect and remove duplicate records.
- Detect potential outliers using the Interquartile Range (IQR) method.
- Encode categorical columns into numerical values.
- Apply normalization to scale values to a selected range.
- Apply standardization to transform features using their mean and standard deviation.

**Concepts:** Data cleaning, missing-value handling, duplicate removal, outlier detection, categorical encoding, normalization, and standardization.

**Outcome:** The experiment demonstrates how preprocessing improves data quality and prepares raw data for Machine Learning.

### Experiment 2 – Exploratory Data Analysis (EDA)

**Dataset:** Titanic Dataset (`train.csv`)

**Objective:** Explore the Titanic dataset to understand its variables, distributions, missing values, and relationships.

**Tasks performed:**
- Load and inspect the dataset.
- Calculate descriptive statistics.
- Check data types and missing values.
- Examine distributions of numerical and categorical variables.
- Visualize data using plots such as histograms, count plots, and box plots.
- Analyze relationships between variables using correlation analysis and visualizations.
- Explore patterns related to passenger survival.

**Concepts:** Descriptive statistics, data visualization, distributions, missing-value analysis, and correlation.

**Outcome:** EDA helps identify patterns, data-quality issues, and relationships that may be useful for subsequent analysis or modeling.

### Experiment 3 – Linear and Logistic Regression

This experiment applies two supervised learning algorithms to two different datasets.

#### Part A – Linear Regression

**Dataset:** USA Housing Dataset (`USA Housing.csv`)

**Objective:** Predict house prices using housing-related features.

**Tasks performed:**
- Load and inspect the dataset.
- Check missing values and duplicate records.
- Perform exploratory data analysis.
- Select input features and the target house-price column.
- Split the data into training and testing sets.
- Train a Linear Regression model.
- Examine model coefficients and intercept.
- Predict house prices for the test set.
- Compare actual and predicted values.

**Evaluation metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² score.

**Recorded output:**

| Metric | Value |
|---|---:|
| MSE | 10,089,009,300.89 |
| RMSE | 100,444.06 |
| MAE | 80,879.10 |
| R² score | 0.9180 |

**Observation:** The recorded R² score is approximately 0.918, meaning the model explains about 91.8% of the variation in the test-set house prices for that run.

#### Part B – Logistic Regression

**Dataset:** Diabetes Dataset (`Diabetes.csv`)

**Objective:** Predict the diabetes outcome from the available medical and demographic features.

**Tasks performed:**
- Load and inspect the dataset.
- Check missing values and duplicates.
- Explore the target-class distribution.
- Separate features and target labels.
- Split the data using stratified sampling.
- Standardize the input features.
- Train a Logistic Regression classifier.
- Predict class labels and probabilities.
- Evaluate the classifier and visualize the confusion matrix.

**Evaluation metrics:** Accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.

**Recorded output:**

| Metric | Value |
|---|---:|
| Accuracy | 0.7143 |
| Precision | 0.6087 |
| Recall | 0.5185 |
| F1 score | 0.5600 |
| ROC-AUC | 0.8230 |

**Confusion matrix:**

```text
[[82 18]
 [26 28]]
```

**Observation:** In the recorded run, Logistic Regression achieved approximately 71.43% accuracy and a ROC-AUC score of 0.8230.

### Experiment 4 – Principal Component Analysis (PCA)

**Dataset:** Iris Dataset (`Iris.csv`)

**Objective:** Reduce the number of input features while retaining as much information from the original dataset as possible.

The Iris dataset has four numerical features:
- Sepal length
- Sepal width
- Petal length
- Petal width

**Tasks performed:**
- Load and inspect the dataset.
- Check missing values and duplicates.
- Separate features and target labels.
- Standardize the input features.
- Apply PCA to reduce the four features to two principal components.
- Calculate explained and cumulative explained variance.
- Visualize the variance retained by the components.
- Train Logistic Regression using the reduced features.
- Evaluate the model and visualize the confusion matrix.

**Recorded explained variance:**

| Component | Explained variance |
|---|---:|
| PC1 | 72.77% |
| PC2 | 23.03% |
| PC3 | 3.68% |
| PC4 | 0.52% |

The first two components retain approximately **95.80%** of the total variance.

**Recorded output:**

| Item | Result |
|---|---|
| Original dimensions | 4 |
| Reduced dimensions | 2 |
| Variance retained | 95.80% |
| Classification accuracy | 88.89% |

**Confusion matrix:**

```text
[[15  0  0]
 [ 0 14  1]
 [ 0  4 11]]
```

**Observation:** PCA reduces the feature space from four dimensions to two while retaining most of the dataset's variance. Classification accuracy depends on the split and model settings used.

### Experiment 5 – Support Vector Machine (SVM) for Classification

**Dataset:** Iris Dataset (`iris.csv`)

**Objective:** Classify Iris flowers into Setosa, Versicolor, and Virginica using a Support Vector Machine.

**Tasks performed:**
- Load and inspect the Iris dataset.
- Separate the four flower measurements from the species label.
- Split the dataset into training and testing sets.
- Standardize the features.
- Train an SVM classifier using the Radial Basis Function (RBF) kernel.
- Predict species for test samples.
- Calculate accuracy and generate a confusion matrix and classification report.
- Predict the species of a new flower sample.

**Model configuration:**
- Algorithm: Support Vector Machine
- Kernel: RBF
- `C`: 1.0
- `gamma`: `scale`

**Recorded output:**
- Training samples: 120
- Testing samples: 30
- Accuracy: 100%

**Confusion matrix:**

```text
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

**New sample prediction:**

```text
Input features: [5.1, 3.5, 1.4, 0.2]
Predicted species: setosa
```

**Concepts:** Hyperplane, support vectors, maximum margin, kernels, RBF kernel, feature scaling, precision, recall, F1 score, and confusion matrix.

**Observation:** The recorded SVM run correctly classified all 30 test samples. Results may vary if the data split or model settings change.

### Experiment 6 – Decision Tree and K-Nearest Neighbour (KNN)

**Dataset:** Iris Dataset (`iris.csv`)

**Objective:** Implement and compare Decision Tree and KNN classifiers for Iris flower classification.

**Algorithms:**
1. Decision Tree Classifier
2. K-Nearest Neighbour Classifier

**Tasks performed:**
- Load the Iris dataset and select features and target labels.
- Split the data into training and testing sets.
- Apply feature scaling where required, particularly for KNN.
- Train a Decision Tree classifier.
- Train a KNN classifier.
- Predict species for test samples.
- Compare accuracy.
- Generate confusion matrices and classification reports.

**Concepts:**
- Decision-tree splitting
- Entropy
- Information gain
- Gini index
- K-nearest neighbours
- Euclidean and Manhattan distance
- Feature scaling
- Classification metrics

**Recorded output:**

| Algorithm | Accuracy |
|---|---:|
| Decision Tree | 100% |
| KNN | 100% |

**Observation:** Both models achieved 100% accuracy in the recorded run. Performance can vary with the train-test split and hyperparameters.

### Experiment 7 – Comparison of Different Regression Algorithms

**Dataset:** California Housing Dataset

**Objective:** Compare different regression algorithms for predicting housing values.

**Algorithms implemented:**
1. Linear Regression
2. Polynomial Regression
3. Decision Tree Regression
4. Random Forest Regression
5. Support Vector Regression (SVR)

**Tasks performed:**
- Load and inspect the California Housing dataset.
- Select input features and the target variable.
- Split the dataset into training and testing sets.
- Train each regression model.
- Generate predictions for the test data.
- Compare models using MAE, MSE, and R² score.

**Evaluation metrics:**
- **MAE:** Average absolute difference between actual and predicted values.
- **MSE:** Average squared difference between actual and predicted values.
- **R² score:** Indicates how well the model explains the variation in the target variable.

**Recorded model comparison:**

| Model | MAE | MSE | R² score |
|---|---:|---:|---:|
| Linear Regression | 0.5332 | 0.5559 | 0.5758 |
| Polynomial Regression | 0.4670 | 0.4643 | 0.6457 |
| Decision Tree Regression | 0.4547 | 0.4952 | 0.6221 |
| Random Forest Regression | 0.3275 | 0.2554 | 0.8051 |
| Support Vector Regression (SVR) | 0.3973 | 0.3542 | 0.7297 |

**Observation:** In the recorded comparison, Random Forest Regression achieved the lowest MAE and MSE and the highest R² score among the five models.

### Experiment 8 – Comparison of Different Classification Algorithms

**Dataset:** Iris Dataset (`iris.csv`)

**Objective:** Compare the performance of multiple classification algorithms on the Iris dataset.

**Algorithms implemented:**
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbour (KNN)
6. Naive Bayes

**Tasks performed:**
- Load and inspect the Iris dataset.
- Separate features and target labels.
- Split the dataset into training and testing sets.
- Apply feature scaling where needed.
- Train each classification model.
- Predict the test labels.
- Compare models using precision, recall, and accuracy.

**Evaluation metrics:**
- **Precision:** Proportion of predicted positive cases that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **Accuracy:** Proportion of all predictions that are correct.

**Recorded output:**

| Algorithm | Precision | Recall | Accuracy |
|---|---:|---:|---:|
| Logistic Regression | 1.00 | 1.00 | 1.00 |
| Decision Tree | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 |
| SVM | 1.00 | 1.00 | 1.00 |
| KNN | 1.00 | 1.00 | 1.00 |
| Naive Bayes | 1.00 | 1.00 | 1.00 |

**Observation:** All six algorithms achieved 100% precision, recall, and accuracy in the recorded run. These results depend on the dataset split and model configuration.

### Experiment 9 – K-Means Clustering

**Dataset:** Mall Customers Dataset (`Mall Customers.csv`)

**Output file:** `Mall_Customers_with_Clusters.csv`

**Objective:** Group customers into clusters based on similarities in their characteristics and spending behaviour.

The dataset commonly includes:
- Customer ID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

**Tasks performed:**
- Load and inspect the customer dataset.
- Select relevant numerical features for clustering.
- Scale features when appropriate.
- Use the Elbow Method to help choose the number of clusters.
- Apply K-Means clustering.
- Assign a cluster label to each customer.
- Visualize the resulting clusters and their centroids.
- Save the dataset with cluster assignments to a new CSV file.

**Important concepts:**

**K-Means:** An unsupervised learning algorithm that groups data points into `K` clusters.

**Euclidean distance:** Measures the distance between a data point and a cluster centroid.

```text
d(x, μ) = sqrt(sum((x_i - μ_i)^2))
```

**Within-Cluster Sum of Squares (WCSS):** Measures the sum of squared distances between data points and their assigned cluster centroids.

**Elbow Method:** Compares WCSS for different values of `K`; the bend in the curve helps identify a suitable number of clusters.

**Output:** `Mall_Customers_with_Clusters.csv` contains customer records and their assigned cluster labels.

**Applications:**
- Customer segmentation
- Market analysis
- Customer behaviour analysis
- Identifying groups with similar spending patterns

**Observation:** K-Means provides cluster labels without requiring predefined target classes.

### Experiment 10 – Iris Flower Classification Using a Single-Layer Artificial Neural Network (ANN)

**Dataset:** Iris Dataset (`iris.csv`)

**Program file:** `single_layer_ann_iris.py`

**Objective:** Build a single-layer artificial neural network for a simple binary classification problem using the Iris dataset.

**Classes used:**
- Setosa → `0`
- Versicolor → `1`

Virginica is excluded from this experiment, so the model performs binary classification on two Iris species.

**Input features:**
- Sepal length
- Sepal width
- Petal length
- Petal width

**Model architecture:** Four input features connect to one sigmoid output neuron. The output is converted into class `0` or class `1` using a threshold of `0.5`.

**Tasks performed:**
- Load the Iris data from a CSV file using Pandas.
- Select Setosa and Versicolor samples.
- Encode the species labels as `0` and `1`.
- Split the dataset into training and testing sets.
- Standardize the input features.
- Initialize weights and bias.
- Implement the sigmoid activation function using NumPy.
- Train the model using gradient descent over multiple epochs.
- Predict labels for the test data.
- Evaluate the model using accuracy, confusion matrix, precision, recall, and F1 score.
- Predict the class of a new flower sample.
- Visualize the confusion matrix.

**Core concepts:**

**Weighted sum:**

```text
z = X · weights + bias
```

**Sigmoid activation function:**

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

The sigmoid function converts the weighted sum into a value between 0 and 1.

**Class prediction:**

```text
If output >= 0.5, predict Versicolor (1).
Otherwise, predict Setosa (0).
```

**Training:** The model calculates prediction errors, computes gradients, and updates the weights and bias using the learning rate.

**Evaluation:** The program prints the actual and predicted classes, accuracy, confusion matrix, and classification report. It also predicts a new sample:

```text
[5.1, 3.5, 1.4, 0.2]
```

This sample is expected to be classified as Setosa based on its measurements.

**Implementation note:** NumPy is used to implement the neural-network training logic manually. Scikit-learn is used for train-test splitting, standardization, and evaluation metrics; it is not used to create the ANN model.

**Outcome:** The experiment demonstrates how a basic single-layer neural network learns weights and bias to classify two flower species.

## Installation and Execution

### 1. Requirements

Install Python 3 and the libraries used in the experiments:

```bash
python3 -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 2. Run Jupyter Notebook

Open a terminal in the repository directory and run:

```bash
jupyter notebook
```

Then open the required experiment notebook.

### 3. Run Experiment 10 as a Python script

Make sure `iris.csv` is available at the path used by the script, or update the path in the code. For example:

```python
df = pd.read_csv("/home/anjali/Downloads/iris.csv")
```

Run the script using:

```bash
python3 single_layer_ann_iris.py
```

The CSV should contain columns named `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, and `species`. If your CSV uses different column names, update the column selections in the code.

## Learning Outcomes

After completing these experiments, the following concepts are practiced:

- Data cleaning and preprocessing
- Missing-value handling and duplicate removal
- Outlier detection and categorical encoding
- Normalization and standardization
- Exploratory Data Analysis and visualization
- Linear Regression and Logistic Regression
- Regression evaluation using MAE, MSE, RMSE, and R²
- Principal Component Analysis and dimensionality reduction
- Support Vector Machine classification
- Decision Tree and K-Nearest Neighbour classification
- Comparing classification and regression algorithms
- K-Means clustering and customer segmentation
- Artificial neural network fundamentals
- Sigmoid activation, weights, bias, gradient descent, and binary classification
- Model evaluation using accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrices

## Author

**Anjali Palake**  
B.Tech – Information Technology  
Government College of Engineering, Karad

GitHub: [Anjali-Palake75](https://github.com/Anjali-Palake75)  
Repository: [ML_Practicals](https://github.com/Anjali-Palake75/ML_Practicals)

---

*Note: Metrics listed above are the recorded results from the corresponding practical runs. Results may change if the dataset, preprocessing, train-test split, random seed, or model parameters are changed.*
