import os
import pandas as pd

def import_data(dataset_train, dataset_test):
    train_dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__), dataset_train))
    test_dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__), dataset_test))
    return train_dataframe, test_dataframe

def check_if_file_exist(filename):
    file_exists = os.path.exists(filename)
    if file_exists:
        return True
    else:
        return False