import os
import pandas as pd

#Constants
train_dataset_path = 'assets/sign_mnist_train/sign_mnist_train.csv'
test_dataset_path = 'assets/sign_mnist_test/sign_mnist_test.csv'

def importData(dataset_train, dataset_test):
    train_dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__), dataset_train))
    test_dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__), dataset_test))
    return train_dataframe, test_dataframe

def main():
    df_train, df_test = importData(train_dataset_path, test_dataset_path)

main()