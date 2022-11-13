import os
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import visualization
import constants as const

# import data from csv dataset
def import_raw_dataset(ref_Index):
    dataframe = pd.read_csv(os.path.join
    (os.path.dirname(__file__), 
        const.RAW_DATASET_ADDR[ref_Index]), 
        usecols=const.COL_HEADERS[ref_Index], 
        thousands=","
    )
    return dataframe

# format inconsistent date format i.e 12 hr to 24 hr
def format_date(val):
    return pd.to_datetime(val) if val[-1].isdigit() else pd.to_datetime(val).strftime('%d/%m/%Y %H:%M:%S')

def reindex_date_col(df, col, drop_cols):
    dataframe = df.set_index(pd.DatetimeIndex(df[col]))
    dataframe.drop(drop_cols, axis=1, inplace=True)
    return dataframe

def pm25_preprocessing(pm25_dataframe):
    date_time_col = const.COL_HEADERS[const.REFINDEX_PM25][0]

    pm25_dataframe[date_time_col] = pm25_dataframe[date_time_col].apply(format_date)
    pm25_dataframe = reindex_date_col(pm25_dataframe, date_time_col, date_time_col)

    # Average PM2.5 data hourly to daily 
    avg_pm25_dataframe = pm25_dataframe.resample('D').mean(numeric_only = True)

    # Normalize the data between 0 to 1
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_avg_vals = min_max_scaler.fit_transform(avg_pm25_dataframe)
    scaled_pm25_dataframe = pd.DataFrame(scaled_avg_vals, 
        index = avg_pm25_dataframe.index, 
        columns = avg_pm25_dataframe.columns
    )

    # To drop rows with null values
    scaled_pm25_dataframe = scaled_pm25_dataframe.dropna()

    # Categorize the average data as '0' if < 0.5 or '1' if >= 0.5
    scaled_pm25_dataframe['Level'] = pd.cut(scaled_pm25_dataframe['Average'], 
        bins = [0.0, const.PM25_NORMALIZATION_THRESHOLD, 1.0], 
        labels = const.LEVEL
    )
    scaled_pm25_dataframe.drop('Average', axis=1, inplace=True)

    return scaled_pm25_dataframe

def traffVol_preprocessing(traffVol_dataframe):
    date_col = const.COL_HEADERS[const.REFINDEX_TRAFFVOL][0]
    county_col = const.COL_HEADERS[const.REFINDEX_TRAFFVOL][1]

    ## To filter out records for Halifax region only
    traffVol_dataframe = traffVol_dataframe[traffVol_dataframe[county_col] == "HFX"]

    traffVol_dataframe = reindex_date_col(traffVol_dataframe, date_col, [date_col, county_col])

    # To drop rows with null values
    traffVol_dataframe = traffVol_dataframe.dropna()

    # Get mean of same day traffic volume
    avg_traffVol_dataframe = traffVol_dataframe.groupby(traffVol_dataframe.index).mean()

    return avg_traffVol_dataframe

def merge_dataframe(df1, df2):
    merged_dataframe = df2.merge(df1, left_index=True, right_index=True)
    merged_dataframe = merged_dataframe.dropna()
    return merged_dataframe

# Splitting train and test data 50/50
def split_data(dataframe):
    global inputdata_features, inputdata_targetvector
    inputdata_features = dataframe.drop(dataframe.columns[1], axis=1)
    inputdata_targetvector = dataframe.Level

    return train_test_split(
        inputdata_features, 
        inputdata_targetvector, 
        test_size=const.TEST_SIZE, 
        random_state=const.RANDOM_STATE
    )

def train_data(DecisionTreeClassifier_Criteria, X_train, y_train):
    tree_entropy = DecisionTreeClassifier(
        criterion=DecisionTreeClassifier_Criteria,
         max_depth=10, 
         min_samples_split = 3,
         min_samples_leaf=3, 
         random_state=const.RANDOM_STATE
    )
    print("Training Data...")
    tree_entropy = tree_entropy.fit(X_train, y_train)
    return tree_entropy

def prediction(X_test, classification_object):
    y_pred = classification_object.predict(X_test)
    print("Predicted values: ")
    print(y_pred)
    return y_pred

def accuracy(y_test, y_pred):
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
    print("ClassificationReport: \n", classification_report(y_test, y_pred, zero_division=True))

def cross_validation(model, X, y, kf):
    cv_result = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=kf,
        n_jobs=4,
        return_estimator=True
    )
    return cv_result

def cv_visualization(model, X, y, kf):
    results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=kf,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        n_jobs=4,
        return_train_score=True)
    return results
    
##
## Main 
##
def main():
    pm25_dataframe = import_raw_dataset(const.REFINDEX_PM25)
    pm25_dataframe = pm25_preprocessing(pm25_dataframe)

    traffVol_dataframe = import_raw_dataset(const.REFINDEX_TRAFFVOL)
    traffVol_dataframe = traffVol_preprocessing(traffVol_dataframe)

    ready_dataframe = merge_dataframe(pm25_dataframe, traffVol_dataframe)

    visualization.dataframe_visualization(ready_dataframe, 'scatter')

    X_train, X_test, y_train, y_test = split_data(ready_dataframe)

    entropy_result = train_data(const.DTREE_CRITERIA, X_train, y_train)
    
    # # Report and Result
    entropy_y_pred = prediction(X_test, entropy_result)
    accuracy(y_test, entropy_y_pred)

    kf = KFold(n_splits=const.KFOLD_SPLIT, random_state=const.RANDOM_STATE, shuffle=True)
    cv_result = cross_validation(entropy_result, inputdata_features, inputdata_targetvector, kf)
    print("Accuracy:", cv_result['test_score'].mean()*100)
    print("Standard Deviation:", cv_result['test_score'].std()*100)

    cv_visualization(entropy_result, inputdata_features, inputdata_targetvector, kf)

if __name__=='__main__':
    main()