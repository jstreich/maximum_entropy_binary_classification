################################################################################
# Run Maximum Entropy Logistic Regression for 0/1 classification of 1D Vectors
# Authors: Jared Streich and ChatGPT
# Notes: This was drafted by ChatGPT and took significant modification to work
#  - most notably updating called packages, variable names, and data format
#  - plotting library versions didn't work, column format of second input file
# Date Created Feb 25th 2023
# Version 0.1.3
# email: streich.jared@gmail.com, if at ornl ju0@ornl.gov
################################################################################

################################################################################
################################# Load Libraries ###############################
################################################################################

print("import libraries...")
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pickle


################################################################################
############################ Load and setup Data Sets ##########################
################################################################################

##### Load data from CSV file
print("load data...")
data = pd.read_csv("HS_145k-pixels_withNames_train_setDel.txt", sep = "\t")

##### Split the data into X (features) and y (target)
print("split data on y-vector and features...")
X = data.iloc[:,2:].values
y = data.iloc[:,1].values

##### Train and validate the maximum entropy model using k-fold cross-validation
print("set up k-fold validation...")

### Number of k fold splits
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
roc_auc_list = []

#### Loop Through Data Set
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    maxent = LogisticRegression(max_iter=10000)
    maxent.fit(X_train, y_train)
    probs = maxent.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
    roc_auc_list.append(auc(fpr, tpr))

##### Calc Average AUC Across k-fold Splits
print("create AUC k-fold info...")
roc_auc = np.mean(roc_auc_list)
print("ROC AUC Mean = ", roc_auc)

##### Plot ROC Curve
print("create ROC AUC plots...")
fpr, tpr, _ = roc_curve(y, maxent.predict_proba(X)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc.pdf")

##### Calc and Plot Feature Importance
print("calculate feature importance...")
feature_importance = np.abs(maxent.coef_[0])
feature_importance /= feature_importance.max()
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(10,10))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, data.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig("feature_importance.pdf")

##### Save the model to a file
print("save model to file...")
with open("maxent_model.pkl", "wb") as f:
    pickle.dump(maxent, f)

##### Read in prediction data from CSV file
print("read in second prediction dataset...")
pred_data = pd.read_csv("HS_145k-pixels_wRowNames_tabDelim.txt", sep = "\t")

###### Split the prediction data into X_pred (features) and sample_names
print("split to only prediction features...")
sample_names = pred_data.iloc[:,0].values

##### Split Second Data Set to Test 1st Model
print("split data on y-vector and features...")
X_pred = pred_data.iloc[:,2:].values

##### Run Predictions Prediction Data
print("run prediction on prediction only data...")
pred_probs = maxent.predict_proba(X_pred)[:, 1]

##### Save Predictions CSV File
print("save prediction file's data to file...")
pred_df = pd.DataFrame({'Sample Name': sample_names, 'Prediction Probability': pred_probs})
pred_df.to_csv("predictions.csv", index=False)
