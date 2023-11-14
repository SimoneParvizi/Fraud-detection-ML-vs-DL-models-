import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import FalseNegatives, FalsePositives , TrueNegatives, TruePositives, Precision, Recall



df = pd.read_csv('credit.csv')
df['Class'].value_counts() #492/284807 = Fraud/Not Fraud
df.hist(bins=30, figsize=(30, 30)) #Columns V1,V2, etc are PCA transformed features

# Preprocessing
new_df = df.copy()
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].to_numpy().reshape(-1, 1))
time = new_df['Time'] 
new_df['Time'] = (time - time.min()) / (time.max() - time.min())
new_df = new_df.sample(frac=1, random_state=1)

# Splitting the data into features and target
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# Graphs
fraud_times = df.Time[df.Class == 1].dropna()
normal_times = df.Time[df.Class == 0].dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Fraud Transactions
sns.histplot(fraud_times, bins=100, kde=True, color='darkorange', ax=ax1, alpha=0.5)
ax1.set_title('Histogram and KDE Plot for Fraud Transactions\n')
ax1.set_ylabel('Number of Transactions\n')
ax1.grid(True)

# Normal Transactions
sns.histplot(normal_times, bins=100, kde=True, ax=ax2, alpha=0.5)
ax2.set_title('\nHistogram and KDE Plot for Normal Transactions\n')
ax2.set_xlabel('\nTime (in Seconds)\n')
ax2.set_ylabel('Number of Transactions\n')
ax2.grid(True)

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


# Splitting the dataset into training and temp (test + validation) sets
x_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)

# Test and validation sets
X_test, x_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=1)

print(x_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(x_val.shape, y_val.shape)

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(X.shape[-1],)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.summary()

# Compiling
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics = [
                  FalseNegatives(name="fn"),
                  FalsePositives(name="fp"),
                  TrueNegatives(name="tn"),
                  TruePositives(name="tp"),
                  Precision(name="precision"),
                  Recall(name="recall")
                  ])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Calculate class weights
neg, pos = np.bincount(y_train)
total = neg + pos
class_weight = {0: 1, 1: 5}



# Training
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=[es], class_weight=class_weight)

# Model Evaluation
y_pred = (model.predict(X_test) > 0.5).astype("int32")

def neural_net_predictions(model, x):
  return (model.predict(x).flatten() > 0.5).astype(int)



# score precision, recall, and f1
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Plot only the losses from history
losses = history.history['loss']
val_losses = history.history['val_loss']

plt.figure(figsize=(10, 7))
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



# Downsampling for ML models
def random_downsample(dataset, samples_target_0_to_keep):
    idx_target1 = dataset[dataset['Class'] == 1].index
    idx_target0 = dataset[dataset['Class'] == 0].index

    # Selecting randomly indexes from target 0
    idx_target0_holdout = np.random.choice(idx_target0, samples_target_0_to_keep, replace=False)
    index_to_holdout = idx_target1.union(idx_target0_holdout)


    # Remove those indexes from dataset
    down_sampled_dtset  = dataset.loc[index_to_holdout]

    return down_sampled_dtset

down_sampled_dtset = random_downsample(df, samples_target_0_to_keep= 492)

# Split the downsampled dataset into features and target
X_down = down_sampled_dtset.drop('Class', axis=1)
y_down = down_sampled_dtset['Class']


# Comparesion ML vs DL
classifiers = {
    "RandomForest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier()
}

# Plotting
cv_results = {}
for name, clf in classifiers.items():
    precision_scores = cross_val_score(clf, X_down, y_down, cv=5, scoring='precision')
    recall_scores = cross_val_score(clf, X_down, y_down, cv=5, scoring='recall')
    f1_scores = cross_val_score(clf, X_down, y_down, cv=5, scoring='f1')
    cv_results[name] = {
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1': np.mean(f1_scores)
    }

# Add Neural Network results
y_pred_nn = neural_net_predictions(model, X_test)
f1_nn = f1_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)

cv_results['NeuralNetwork'] = {
    'precision': precision_nn,
    'recall': recall_nn,
    'f1': f1_nn
}

# Data for plotting
labels = cv_results.keys()

precision_scores = [cv_results[model]['precision'] for model in labels]
recall_scores = [cv_results[model]['recall'] for model in labels]
f1_scores = [cv_results[model]['f1'] for model in labels]

x = np.arange(len(labels))  # Label locations
width = 0.25  # Bars width 

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, precision_scores, width, label='Precision')
rects2 = ax.bar(x, recall_scores, width, label='Recall')
rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score')

ax.set_xlabel('Models')
ax.set_ylabel('Scores\n')
ax.set_title('Model Comparison by Precision, Recall, and F1 Score\n')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Attach value above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()