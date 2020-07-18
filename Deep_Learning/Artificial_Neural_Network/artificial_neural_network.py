# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:03:39 2020

@author: Rahul Kumar
"""

# Artificial Neural Network

# Importing the library

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from csv import writer
from sklearn.metrics import confusion_matrix, accuracy_score


def copy_csv(file_name):
    new_dataset = pd.read_csv(file_name)
    df = new_dataset.iloc[:, 1:]
    df.to_csv('copy_of_' + file_name)
    return 'copy_of_' + file_name


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


if __name__ == "__main__":

    # Importing the dataset
    file_name = 'Churn_Modelling.csv'
    dataset = pd.read_csv(file_name)
    x = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values

    cp_file_name = copy_csv(file_name)

    # Take the new prediction criteria so that it can be pre-processed
    row_number = dataset.values[-1][0]
    customer_id = dataset.values[-1][1]
    row_number_append = row_number + 1
    customer_id_append = customer_id + 1
    surname = input('Enter your surname here')
    list_of_elem = [row_number_append, customer_id_append, surname, 600,
                    'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]
    append_list_as_row(cp_file_name, list_of_elem)
    dataset_cp = pd.read_csv(cp_file_name)
    x_cp = dataset_cp.iloc[:, 3:-1].values
    os.remove(cp_file_name)

    # Encode the categorical data

    # Label encoding for Gender column
    le = LabelEncoder()
    x[:, 2] = le.fit_transform(x[:, 2])
    x_cp[:, 2] = le.fit_transform(x_cp[:, 2])

    # One Hot Encoding for Geography column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
                           remainder='passthrough')
    x = np.array(ct.fit_transform(x))
    x_cp = np.array(ct.fit_transform(x_cp))

    # Split the dataset into Training and Test Set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                        random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    x_cp_train = sc.transform(x_cp)

    # Building the ANN

    # Initializing the ANN

    ann = tf.keras.models.Sequential()

    # Adding the input layer and the first hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Add the output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    # Use activation='softmax' if using other than binary classification
    # Since we are using activation at last layer for binary classification
    # We use activation='sigmoid'

    # Training the ANN

    # Compiling the ANN
    # For non binary classification use
    # loss='categorical_crossentropy'
    ann.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])

    # Training the ANN on Training Set
    ann.fit(x_train, y_train, batch_size=32, epochs=150)

    # Predicting the sample Data
    required_data = list(x_cp[-1])
    print(required_data)
    print("Will", surname, "leave the bank?")
    print(ann.predict(sc.transform([required_data])) > 0.5)

    y_pred = (ann.predict(x_test) > 0.5)

    print(np.concatenate((
                        y_pred.reshape(len(y_pred), 1),
                        y_test.reshape(len(y_test), 1)
                      ), axis=1))

    cm = confusion_matrix(y_test, y_pred, labels=None)
    print(cm)
    print(accuracy_score(y_test, y_pred))
