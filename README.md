# Project: Car Evaluation Classification

This project focuses on classifying the acceptability of cars based on a set of categorical features. The Random Forest Classifier is employed for this task, chosen for its effectiveness in handling categorical data (when appropriately encoded) and its robustness, particularly with potentially imbalanced datasets.

---

## Dataset

The analysis utilizes the `car.data` dataset from the UCI Machine Learning Repository. This dataset contains information about car attributes, all ofwhich are categorical. The features include:

* `buying`: Buying price
* `maint`: Maintenance price
* `doors`: Number of doors
* `persons`: Capacity in terms of persons to carry
* `lug_boot`: Size of luggage boot
* `safety`: Estimated safety of the car
* `class`: Target variable representing the acceptability class (unacc, acc, good, vgood)

---

## Methodology

The project follows these key steps:

1.  **Data Loading and Preparation:**
    * The dataset is loaded using pandas, with column names explicitly defined as "buying", "maint", "doors", "persons", "lug_boot", "safety", and "class".
    * The dataset is split into features (X) by dropping the 'class' column, and the target variable (y) which is the 'class' column.
    * The data is then split into training (70%) and testing (30%) sets using `train_test_split` with `random_state=7`.

2.  **Data Preprocessing (Categorical Encoding):**
    * Since all features are categorical, `LabelEncoder` from scikit-learn is used to convert these categorical text values into numerical representations.
    * A separate `LabelEncoder` is fitted for each categorical feature in the training set (X_train).
    * The same fitted encoders are then used to transform the corresponding columns in the test set (X_test) to ensure consistency.

3.  **Model Training:**
    * A `RandomForestClassifier` is instantiated.
    * The model is trained using the numerically encoded training data (`X_train`, `y_train`).

4.  **Model Evaluation:**
    * Predictions are made on the encoded test set (`X_test`).
    * The model's performance is evaluated using:
        * **Accuracy Score:** To get an overall measure of correctly classified instances.
        * **Classification Report:** To obtain precision, recall, and F1-score for each class, providing a more detailed view of the model's performance across different acceptability levels.
        * **Confusion Matrix:** Visualized using `ConfusionMatrixDisplay` (normalized 'true') to understand the types of errors the classifier is making (e.g., which classes are being confused with others).

---

## Results

* The Random Forest Classifier achieved an **Accuracy Score of 98.27%** on the test set.
* The **Classification Report** indicates high precision, recall, and F1-scores across most classes:
    * `acc`: Precision=0.98, Recall=0.96, F1-score=0.97
    * `good`: Precision=1.00, Recall=0.82, F1-score=0.90
    * `unacc`: Precision=0.99, Recall=1.00, F1-score=0.99
    * `vgood`: Precision=0.93, Recall=1.00, F1-score=0.96
* The **Normalized Confusion Matrix** visually confirms the model's strong performance, showing high true positive rates for each class and minimal misclassifications.

The choice of Random Forest Classifier, along with label encoding for categorical features, proved effective for this classification task, yielding high accuracy even with the imbalanced nature of the dataset.

---

## Files in This Repository

* `carS.ipynb`: The Jupyter Notebook containing the Python code for the analysis.
* `car.data`: The dataset used for this project.
* `README.md`: This file, providing an overview of the project.

---

## How to Run

1.  Ensure you have Python installed.
2.  Install the necessary libraries:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```
3.  Download the `Atividade_dia_10_01_2025.ipynb` notebook and the `car.data` dataset into the same directory.
4.  Open and run the Jupyter Notebook using an environment like Jupyter Lab or Google Colab.

---
