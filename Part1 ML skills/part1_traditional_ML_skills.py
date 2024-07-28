import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.losses import BinaryCrossentropy

class ML_skills():
    def __init__(self):
        """
        Initializes the ML_skills class with default values.
        Sets the target column for classification.
        """
        self.target_col = 'Churn'

    def open_csv(self, path_df: str, path_inference: str):
        """
        Opens CSV files for training and inference data.

        Args:
            path_df (str): The file path to the training data CSV.
            path_inference (str): The file path to the inference data CSV.

        Returns:
            tuple: A tuple containing two pandas DataFrames (training data, inference data).
        """
        data = pd.read_csv(path_df)
        inference = pd.read_csv(path_inference)
        return data, inference

    def data_description(self, data: DataFrame, method: Literal['head', 'info', 'nulls']):
        """
        Provides a description of the data based on the specified method.

        Args:
            data (pd.DataFrame): The data to be described.
            method (str): The method of description ('head', 'info', 'nulls').

        Returns:
            None
        """
        match method:
            case 'head':
                print(data.head(5))
            case 'info':
                print(data.info())
            case 'nulls':
                print(data.isnull().sum())

    def data_treatment(self, data: DataFrame, method: Literal['drop_c', 'delete_null', 'encoding', 'transform'], drop_c: list=None):
        """
        Cleans and preprocesses the data based on the specified method.

        Args:
            data (pd.DataFrame): The data to be cleaned.
            method (str): The method of data cleaning ('drop_c', 'delete_null', 'encoding', 'transform').
            drop_c (list, optional): The columns to drop if method is 'drop_c'.

        Returns:
            pd.DataFrame: The cleaned data.
        """
        match method:
            case 'drop_c':
                data = data.drop(columns=(drop_c))
                return data
            case 'delete_null':
                data = data.dropna()
                return data
            case 'encoding':
                data.Gender = data.Gender.apply(lambda x: x=='Male').astype(int)
                dummy_df = pd.get_dummies(data['Geography'])
                data = pd.concat([data, dummy_df], axis=1)
                return data
            case 'transform':
                cols = data.columns
                scaler = StandardScaler()
                data = pd.DataFrame(scaler.fit_transform(data), columns=cols)
                return data

    def train_test_split(self, data: DataFrame):
        """
        Splits the data into training and testing sets.

        Args:
            data (pd.DataFrame): The data to be split.

        Returns:
            tuple: A tuple containing the training and testing sets (X_train, X_val, y_train, y_val).
        """
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        return X_train, X_val, y_train, y_val

    def pred_acc(self, y_true: int, y_pred: int):
        """
        Calculates the prediction accuracy.

        Args:
            y_true (pd.Series): The true labels.
            y_pred (pd.Series): The predicted labels.

        Returns:
            float: The prediction accuracy in percentage.
        """
        return metrics.accuracy_score(y_true, y_pred) * 100

    def importances(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains a Random Forest classifier and prints the feature importances.

        Args:
            X_train (pd.DataFrame): The training feature set.
            X_test (pd.DataFrame): The testing feature set.
            y_train (pd.Series): The training labels.
            y_test (pd.Series): The testing labels.

        Returns:
            None
        """
        model = RandomForestClassifier(n_estimators=150, min_samples_split=40).fit(X_train, y_train)
        feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.show()

    def logistic_regression(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """
        Trains a Logistic Regression classifier and prints the training and validation accuracy.

        Args:
            X_train (pd.DataFrame): The training feature set.
            X_test (pd.DataFrame): The testing feature set.
            y_train (pd.Series): The training labels.
            y_test (pd.Series): The testing labels.

        Returns
        -------
        float
            The accuracy of the prediction in validation
        """
        self.model_lg = LogisticRegression().fit(X_train, y_train)
        train_pred = self.model_lg.predict(X_train)
        val_pred = self.model_lg.predict(X_test)
        train_acc = self.pred_acc(y_train, train_pred)
        val_acc = self.pred_acc(y_test, val_pred)
        print('Training Accuracy forest = {:.4f} %'.format(train_acc))
        print('Validation Accuracy forest = {:.4f} %'.format(val_acc))
        return val_acc

    def svc(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """
        Trains a Support Vector Machine classifier and prints the training and validation accuracy.

        Args:
            X_train (pd.DataFrame): The training feature set.
            X_test (pd.DataFrame): The testing feature set.
            y_train (pd.Series): The training labels.
            y_test (pd.Series): The testing labels.

        Returns
        -------
        float
            The accuracy of the prediction in validation
        """
        self.model_svc = SVC().fit(X_train, y_train)
        train_pred = self.model_svc.predict(X_train)
        val_pred = self.model_svc.predict(X_test)
        train_acc = self.pred_acc(y_train, train_pred)
        val_acc = self.pred_acc(y_test, val_pred)
        print('Training Accuracy forest = {:.4f} %'.format(train_acc))
        print('Validation Accuracy forest = {:.4f} %'.format(val_acc))
        return val_acc

    def decision_tree(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """
        Trains a Decision Tree classifier and prints the training and validation accuracy.

        Args:
            X_train (pd.DataFrame): The training feature set.
            X_test (pd.DataFrame): The testing feature set.
            y_train (pd.Series): The training labels.
            y_test (pd.Series): The testing labels.

        Returns
        -------
        float
            The accuracy of the prediction in validation
        """
        self.model_dt = DecisionTreeClassifier().fit(X_train, y_train)
        train_pred = self.model_dt.predict(X_train)
        val_pred = self.model_dt.predict(X_test)
        train_acc = self.pred_acc(y_train, train_pred)
        val_acc = self.pred_acc(y_test, val_pred)
        print('Training Accuracy forest = {:.4f} %'.format(train_acc))
        print('Validation Accuracy forest = {:.4f} %'.format(val_acc))
        return val_acc

    def random_forest(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """
        Trains a Random Forest classifier and prints the training and validation accuracy.

        Args:
            X_train (pd.DataFrame): The training feature set.
            X_test (pd.DataFrame): The testing feature set.
            y_train (pd.Series): The training labels.
            y_test (pd.Series): The testing labels.

        Returns
        -------
        float
            The accuracy of the prediction in validation
        """
        self.model_rf = RandomForestClassifier(n_estimators=150, min_samples_split=40).fit(X_train, y_train)
        train_pred = self.model_rf.predict(X_train)
        val_pred = self.model_rf.predict(X_test)
        train_acc = self.pred_acc(y_train, train_pred)
        val_acc = self.pred_acc(y_test, val_pred)
        print('Training Accuracy forest = {:.4f} %'.format(train_acc))
        print('Validation Accuracy forest = {:.4f} %'.format(val_acc))
        return val_acc

    def MLP(self, X_train: np.ndarray, X_val: np.ndarray, y_train, y_val: np.ndarray):
        """
        Trains a Multilayer Perceptron (MLP) neural network and evaluates its performance.

        Args:
            X_train (pd.DataFrame): The training feature set.
            X_val (pd.DataFrame): The validation feature set.
            y_train (pd.Series): The training labels.
            y_val (pd.Series): The validation labels.

        Returns
        -------
        float
            The accuracy of the prediction in validation
        """
        self.model_mlp = Sequential([
            Input((X_train.shape[1],)),
            Dense(64, activation='relu', kernel_regularizer=l2(1e-3)),
            Dropout(0.5),
            Dense(16, activation='relu', kernel_regularizer=l2(1e-3)),
            Dropout(0.3),
            Dense(1, activation='sigmoid'),
        ])

        self.model_mlp.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(learning_rate=0.01),
            metrics=['accuracy']
        )

        self.model_mlp.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

        results = self.model_mlp.evaluate(X_val, y_val)
        print("test loss, test acc:", results)
        return results[1]


ml = ML_skills()
data, inference = ml.open_csv('train.csv', 'inference.csv')

data = ml.data_treatment(data, 'encoding')
inference = ml.data_treatment(inference, 'encoding')

data = ml.data_treatment(data, 'drop_c', ['id', 'CustomerId', 'Surname', 'Geography'])
inference = ml.data_treatment(inference, 'drop_c', ['id', 'CustomerId', 'Surname', 'Geography'])

X_train, X_val, y_train, y_val = ml.train_test_split(data)

ml.importances(X_train, y_train)

acc1 = ml.logistic_regression(X_train, X_val, y_train, y_val)
acc2 = ml.decision_tree(X_train, X_val, y_train, y_val)
acc3 = ml.random_forest(X_train, X_val, y_train, y_val)
acc4 = ml.MLP(X_train, X_val, y_train, y_val)

print(f'Accuracy logistic regression: {acc1}')
print(f'Accuracy decision tree: {acc2}')
print(f'Accuracy random forest: {acc3}')
print(f'Accuracy MLP: {acc4}')

predictions = ml.model_rf.predict(inference)
inference = pd.read_csv('inference.csv')
inference['Churn'] = predictions

print(inference.head(10))
inference.to_csv('inference_target.csv')








