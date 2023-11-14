#  ML vs DL: Credit Card Fraud Detection.
## Project Overview
This project aims to detect credit card fraud using advanced machine learning (ML) and deep learning (DL) techniques. It demonstrates data preprocessing, various ML classifiers, and a neural network model to analyze and predict fraudulent transactions in a credit card dataset.

## Data Preprocessing
Data preprocessing includes scaling of features using `RobustScaler` and normalization of time. The dataset is also randomly downsampled to handle class imbalance.

## Exploratory Data Analysis
Histograms and Kernel Density Estimation (KDE) plots provide insights into the distribution of fraud and normal transactions over time.

![fraud_normal_transactions_plots](https://github.com/SimoneParvizi/ML-DL-models-fraud-detection/assets/75120707/337f07c3-274a-48df-9a2b-daa59f6ab762)

## Model Architecture
The neural network model consists of:
- **Input Layer**: 128 neurons, ReLU activation, L2 regularization (0.001).
- **Dropout Layer**: Reduces overfitting by deactivating 20% of neurons randomly during training.
- **Hidden Layer**: 64 neurons, ReLU activation, L2 (0.001).
- **Output Layer**: A single neuron with a sigmoid activation function for binary classification.

## Training and Evaluation

![train val loss](https://github.com/SimoneParvizi/ML-DL-models-fraud-detection/assets/75120707/dd10ba0e-1c12-43d4-bcbf-c4def347a75e)

During training:
- A sharp decrease in training loss initially, indicating effective learning from the dataset.
- The validation loss decreases in tandem but plateaus towards the end, suggesting the model's generalization without overfitting.
- The implementation of early stopping after 30 epochs prevented overtraining and ensured optimal model performance.


## Model Comparison
The comparison between traditional ML models and the DL model is visually summarized in the bar chart below. It clearly indicates that while the DL model performs adequately, traditional ML models, particularly RandomForest and XGBoost, exhibit higher precision and F1 scores. This suggests that for tabular data, such as this dataset, traditional ML algorithms continue to be highly competitive.

![ML vs DL](https://github.com/SimoneParvizi/ML-DL-models-fraud-detection/assets/75120707/53e32c09-8526-45da-9ff9-c79b04334995)


## Discussion
Despite advancements, the results from this project suggest that ML models hold their ground in handling tabular data, often outperforming DL models in precision and F1 scores. This phenomenon can be attributed to the inherent structure of tabular datasets, where feature interactions are more effectively captured by tree-based models. The XGBoost model's performance stands out, combining high precision with recall balance, which is crucial for fraud detection tasks where both false positives and false negatives have significant consequences.


## Dependencies
- Pandas
- Matplotlib
- Seaborn
- NumPy
- Scikit-Learn
- XGBoost
- TensorFlow
