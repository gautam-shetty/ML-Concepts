import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, rand_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.evaluate import paired_ttest_kfold_cv
import seaborn as sns

import constants as const

# import data from csv dataset
def import_raw_dataset(raw_dataset_addr):
    dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__), raw_dataset_addr))
    return dataframe

def descriptive_analysis(dataframe):
    dataframe.head()
    dataframe.tail()
    dataframe.shape
    dataframe.info()

    #Plot col target with col sex
    fig = sns.countplot(x = 'target', data = dataframe, hue = 'sex')
    fig.set_xticklabels(
        labels=["Doesn't have heart disease", 'Has heart disease'],
        rotation=0
    )
    plt.legend(['Female', 'Male'])
    plt.title("Heart Disease Frequency for Sex")
    plt.show()

    #Plot for heartrate by age
    plt.scatter(dataframe.age[dataframe.target==1], 
            dataframe.thalach[dataframe.target==1], 
            c="tomato")
    plt.scatter(dataframe.age[dataframe.target==0], 
            dataframe.thalach[dataframe.target==0], 
            c="lightgreen")
    plt.title("Heart Disease w.r.t Age and Max Heart Rate")
    plt.xlabel("Age")
    plt.legend(["Disease", "No Disease"])
    plt.ylabel("Max Heart Rate")
    plt.show()

    #Correlation heatmap plot
    corr = dataframe.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, 
        mask=mask, cmap=cmap, vmax=.3, 
        center=0, square=True, linewidths=.5, 
        cbar_kws={"shrink": .5}
        )

def preprocessing_df(dataframe):
    # Converting categorical variable into dummy/indicator variables for proper PCA
    new_dataframe = pd.get_dummies(dataframe, columns = const.CTG_COL_HEADERS)
    minMaxScaler = preprocessing.MinMaxScaler()
    minMaxScaler.fit(new_dataframe[const.CNT_COL_HEADERS])
    new_dataframe[const.CNT_COL_HEADERS] = minMaxScaler.transform(new_dataframe[const.CNT_COL_HEADERS])
    return new_dataframe

def clustering():
    # Initialize the clusterer with n_clusters value and a random generator
    for n_clusters in const.CLUSTER_RANGE:

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(inputdata_features) + (n_clusters + 1) * 10])

        clusterer = KMeans(
            n_clusters=n_clusters, 
            random_state=const.RANDOM_STATE
            )
        cluster_labels = clusterer.fit_predict(inputdata_features)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(inputdata_features, cluster_labels)
        const.CLUSTER_AVG_SILHOTTE_SCORE.append(silhouette_avg)
        print(
            "For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(inputdata_features, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        pca = PCA(n_components=2)
        components = pca.fit_transform(inputdata_features)

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(components[:, 0], components[:, 1],
                    marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k'
                    )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')

    plt.show()
    return const.CLUSTER_RANGE[const.CLUSTER_AVG_SILHOTTE_SCORE.index(max(const.CLUSTER_AVG_SILHOTTE_SCORE))]

# def plot_histogram(dataframe, heading):
#     for col in dataframe.columns:
#         dataframe[col].hist(figsize=(5,5))
#         plt.suptitle(heading)
#         plt.title(col)
#         plt.show()

def clustering_by_no_of_clusters(dataframe, n_clusters):
    clusterer = KMeans(
            n_clusters=n_clusters, 
            random_state=const.RANDOM_STATE
        )
    cluster_labels = clusterer.fit_predict(inputdata_features)
    print("Cluster labels: ", cluster_labels)    
    print("Rand score: ", rand_score(inputdata_targetvector, cluster_labels))


def feature_selection(dataframe):
    global inputdata_features, inputdata_targetvector
    inputdata_features = dataframe.drop('target', axis=1)
    inputdata_targetvector = dataframe.target

# Splitting train and test data 50/50
def split_data(dataframe):
    return train_test_split(
        inputdata_features, 
        inputdata_targetvector, 
        test_size=const.TEST_SIZE, 
        random_state=const.RANDOM_STATE
    )

def prediction(X_test, classification_object):
    y_pred = classification_object.predict(X_test)
    print("Predicted values: ")
    print(y_pred)
    return y_pred

def accuracy(y_test, y_pred):
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
    print("ClassificationReport: \n",
        classification_report(y_test, y_pred, zero_division=True)
        )

def cross_validation(model, X, y):
    # 10-fold cross validation
    cv_result = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=10,
        n_jobs=4,
        return_estimator=True
    )
    return cv_result

##
## Main 
##
def main():
    heart_dataframe = import_raw_dataset(const.RAW_DATASET_ADDR)

    descriptive_analysis(heart_dataframe)

    modeified_heart_dataframe = preprocessing_df(heart_dataframe)

    feature_selection(modeified_heart_dataframe)

    # Clusturing
    best_cluster_size = clustering()

    clustering_by_no_of_clusters(modeified_heart_dataframe, best_cluster_size)

    X_train, X_test, y_train, y_test = split_data(modeified_heart_dataframe)

    ##
    ## Naïve bayes
    ##

    #Create a Gaussian Classifier
    gnb = GaussianNB()

    #Train the model using the training sets
    print("Training Data...")
    gaussian_result = gnb.fit(X_train, y_train)
    
    # Report and Result
    gaussian_y_pred = prediction(X_test, gaussian_result)
    accuracy(y_test, gaussian_y_pred)

    cv_result = cross_validation(gaussian_result, inputdata_features, inputdata_targetvector)
    print("Accuracy:", cv_result['test_score'].mean()*100)
    print("Standard Deviation:", cv_result['test_score'].std()*100)


    ##
    ## Decision Tree Classifier
    ##

    tree_entropy = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=6, 
        random_state=const.RANDOM_STATE
    )
    print("Training Data...")
    entropy_result = tree_entropy.fit(X_train, y_train)

    # Report and Result
    entropy_y_pred = prediction(X_test, entropy_result)
    accuracy(y_test, entropy_y_pred)

    cv_result = cross_validation(entropy_result, inputdata_features, inputdata_targetvector)
    print("Accuracy:", cv_result['test_score'].mean()*100)
    print("Standard Deviation:", cv_result['test_score'].std()*100)


    # Classification comparison result
    statistic, p_value = paired_ttest_kfold_cv(
        estimator1=tree_entropy,
        estimator2=gnb,
        X=inputdata_features.to_numpy(),
        y=inputdata_targetvector.to_numpy()
        )

    print("Statistic:", statistic)
    print("PValue:", p_value)


if __name__=='__main__':
    main()