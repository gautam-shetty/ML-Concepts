import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Constants
refIndex_pm25 = 0
refIndex_traffVol = 1
raw_dataset_addr = ['assets/NS_PM2.5_Data.csv', 'assets/Traffic_Volumes_Data.csv']
col_headers = [['Date & time', 'Average'], ['Date', 'COUNTY', 'ADT']]
pm25_normalization_threshold = 0.3
level = [0, 1]
tree_png = 'DecisionTree1.png'
random_state_constant=50
test_size_constant = 0.5
dtree_criteria = 'entropy'
dtree_max_depth = 2
dtree_min_samples_leaf = 5
kfold_split = 10

# import data from csv dataset
def import_raw_dataset(ref_Index):
    dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__),raw_dataset_addr[ref_Index]), usecols=col_headers[ref_Index], thousands=",")
    return dataframe

# format inconsistent date format i.e 12 hr to 24 hr
def format_date(val):
    return pd.to_datetime(val) if val[-1].isdigit() else pd.to_datetime(val).strftime('%d/%m/%Y %H:%M:%S')

def reindex_date_col(df, col, drop_cols):
    dataframe = df.set_index(pd.DatetimeIndex(df[col]))
    dataframe.drop(drop_cols, axis=1, inplace=True)
    return dataframe

def pm25_preprocessing(pm25_dataframe):
    date_time_col = col_headers[refIndex_pm25][0]

    pm25_dataframe[date_time_col] = pm25_dataframe[date_time_col].apply(format_date)
    pm25_dataframe = reindex_date_col(pm25_dataframe, date_time_col, date_time_col)

    # Average PM2.5 data hourly to daily 
    avg_pm25_dataframe = pm25_dataframe.resample('D').mean(numeric_only = True)

    # Normalize the data between 0 to 1
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_avg_vals = min_max_scaler.fit_transform(avg_pm25_dataframe)
    scaled_pm25_dataframe = pd.DataFrame(scaled_avg_vals, index = avg_pm25_dataframe.index, columns = avg_pm25_dataframe.columns)

    # To drop rows with null values
    scaled_pm25_dataframe = scaled_pm25_dataframe.dropna()

    # Categorize the average data as '0' if < 0.5 or '1' if >= 0.5
    scaled_pm25_dataframe['Level'] = pd.cut(scaled_pm25_dataframe['Average'], bins = [0.0, pm25_normalization_threshold, 1.0], labels = level)
    scaled_pm25_dataframe.drop('Average', axis=1, inplace=True)

    return scaled_pm25_dataframe

def traffVol_preprocessing(traffVol_dataframe):
    date_col = col_headers[refIndex_traffVol][0]
    county_col = col_headers[refIndex_traffVol][1]

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

def plot_dataframe(dataframe, plot_kind):
    dataframe.plot(x = dataframe.columns[0], y=dataframe.columns[1], kind = plot_kind)
    plt.show()


# Splitting train and test data 50/50
def split_data(dataframe):
    global inputdata_features, inputdata_targetvector
    inputdata_features = dataframe.drop(dataframe.columns[1], axis=1)
    inputdata_targetvector = dataframe.Level

    return train_test_split(
        inputdata_features, 
        inputdata_targetvector, 
        test_size=test_size_constant, 
        random_state=random_state_constant
    )

def train_data(DecisionTreeClassifier_Criteria, X_train, y_train):
    tree_entropy = DecisionTreeClassifier(
        criterion=DecisionTreeClassifier_Criteria,
         max_depth=10, 
         min_samples_split = 3,
         min_samples_leaf=3, 
         random_state=random_state_constant
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
    pm25_dataframe = import_raw_dataset(refIndex_pm25)
    pm25_dataframe = pm25_preprocessing(pm25_dataframe)

    traffVol_dataframe = import_raw_dataset(refIndex_traffVol)
    traffVol_dataframe = traffVol_preprocessing(traffVol_dataframe)

    ready_dataframe = merge_dataframe(pm25_dataframe, traffVol_dataframe)

    plot_dataframe(ready_dataframe, 'scatter')

    X_train, X_test, y_train, y_test = split_data(ready_dataframe)

    entropy_result = train_data(dtree_criteria, X_train, y_train)
    
    # # Report and Result
    entropy_y_pred = prediction(X_test, entropy_result)
    accuracy(y_test, entropy_y_pred)


    kf = KFold(n_splits=kfold_split, random_state=random_state_constant, shuffle=True)
    cv_result = cross_validation(entropy_result, inputdata_features, inputdata_targetvector, kf)
    print("Accuracy:", cv_result['test_score'].mean()*100)
    print("Standard Deviation:", cv_result['test_score'].std()*100)

    cv_visualization(entropy_result, inputdata_features, inputdata_targetvector, kf)

if __name__=='__main__':
    main()