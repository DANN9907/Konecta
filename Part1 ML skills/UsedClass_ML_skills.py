from Class_ML_skills import ML_skills
import pandas as pd

ml = ML_skills()
data, inference = ml.open_csv('Part1 ML skills\\train.csv', 'Part1 ML skills\\inference.csv')

data = ml.data_treatment(data, 'encoding')
inference = ml.data_treatment(inference, 'encoding')

# ---------Drop columns without relevant information------------- #
data = ml.data_treatment(data, 'drop_c', ['id', 'CustomerId', 'Surname', 'Geography'])
inference = ml.data_treatment(inference, 'drop_c', ['id', 'CustomerId', 'Surname', 'Geography'])

# ---------Data exploration ----------- #
ml.data_exploration(data)

# ---------Train test split ----------- #
X_train, X_val, y_train, y_val = ml.train_test_split(data)

# ---------Calculating the Most Important Features ----------#
ml.importances(X_train, y_train)

# ---------Accuracy metrics of each method------------- #
acc1 = ml.logistic_regression(X_train, X_val, y_train, y_val)
acc2 = ml.decision_tree(X_train, X_val, y_train, y_val)
acc3 = ml.random_forest(X_train, X_val, y_train, y_val)
acc4 = ml.MLP(X_train, X_val, y_train, y_val)

print(f'Accuracy logistic regression: {acc1}')
print(f'Accuracy decision tree: {acc2}')
print(f'Accuracy random forest: {acc3}')
print(f'Accuracy MLP: {acc4}')

# ---------Predictions on inference and add new column with the predictions------------- #
predictions = ml.model_rf.predict(inference)
inference = pd.read_csv('inference.csv')
inference['Churn'] = predictions

print(inference.head(10))
inference.to_csv('inference_target.csv')