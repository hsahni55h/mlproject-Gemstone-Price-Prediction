"""
This file contains utility functions used across the project. 
It includes functionality to save objects to disk, ensuring that important objects such as models and preprocessors can be persisted and later retrieved.

"""

import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import Custom_Exception
from src.logger import logging

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.

    Args:
    file_path (str): The path where the object should be saved.
    obj: The object to be saved.

    Raises:
    Custom_Exception: If any exception occurs during the save operation, it raises a custom exception with the error details.
    """
    try:
        # Get the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object using pickle
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if any error occurs
        raise Custom_Exception(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return a report with their performance on the test set.

    Args:
    X_train: Training data features.
    y_train: Training data target.
    X_test: Test data features.
    y_test: Test data target.
    models (dict): A dictionary containing model names as keys and instantiated model objects as values.
    param (dict): A dictionary containing the hyperparameters for each model.

    Returns:
    dict: A report containing the test R^2 scores for each model.

    Raises:
    Custom_Exception: If any exception occurs during model evaluation, it raises a custom exception with the error details.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

           # Train model
            model.fit(X_train,y_train)

            # Predict Training data
            y_train_pred = model.predict(X_train)

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    
    except Exception as e:
        logging.info('Exception occured during model training')
        raise Custom_Exception(e,sys)
    
def model_metrics(true, predicted):
    """
    Calculate evaluation metrics for the model.

    Args:
    true (array-like): True target values.
    predicted (array-like): Predicted target values.

    Returns:
    tuple: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R^2 score.

    Raises:
    CustomException: If any exception occurs during metric calculation.
    """
    try:
        mae = mean_absolute_error(true, predicted)  # Calculate Mean Absolute Error
        mse = mean_squared_error(true, predicted)   # Calculate Mean Squared Error
        rmse = np.sqrt(mse)                         # Calculate Root Mean Squared Error
        r2_square = r2_score(true, predicted)       # Calculate R^2 score
        return mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception occurred while evaluating metric')
        raise Custom_Exception(e, sys)


def print_evaluated_results(xtrain, ytrain, xtest, ytest, model):
    """
    Print evaluated results for both training and test sets.

    Args:
    xtrain (array-like): Training data features.
    ytrain (array-like): Training data target.
    xtest (array-like): Test data features.
    ytest (array-like): Test data target.
    model: Trained model used for prediction.

    Raises:
    CustomException: If any exception occurs during result evaluation or printing.
    """
    try:
        ytrain_pred = model.predict(xtrain)  # Predict on training data
        ytest_pred = model.predict(xtest)    # Predict on test data

        # Evaluate metrics for training and test datasets
        model_train_mae, model_train_rmse, model_train_r2 = model_metrics(ytrain, ytrain_pred)
        model_test_mae, model_test_rmse, model_test_r2 = model_metrics(ytest, ytest_pred)

        # Printing training set results
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')

        # Printing test set results
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))

    except Exception as e:
        logging.info('Exception occurred during printing of evaluated results')
        raise Custom_Exception(e, sys)

    
def model_metrics(true, predicted):
    try :
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise Custom_Exception(e,sys)
    

def print_evaluated_results(xtrain,ytrain,xtest,ytest,model):
    try:
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)

        # Evaluate Train and Test dataset
        model_train_mae , model_train_rmse, model_train_r2 = model_metrics(ytrain, ytrain_pred)
        model_test_mae , model_test_rmse, model_test_r2 = model_metrics(ytest, ytest_pred)

        # Printing results
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')
    
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
    
    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise Custom_Exception(e,sys)
    
def load_object(file_path):
    """
    Load an object from a file using pickle.

    Args:
    file_path (str): The path to the file from which the object should be loaded.

    Returns:
    The loaded object.

    Raises:
    Custom_Exception: If any exception occurs during the load operation, it raises a custom exception with the error details.
    """
    try:
        # Open the file in read-binary mode and load the object using pickle
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        # Raise a custom exception if any error occurs
        logging.info('Exception Occured in load_object function utils')
        raise Custom_Exception(e, sys)