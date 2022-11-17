import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import save_model, load_model


import helpers
import visualization
import constants as const
import cnn

def feature_selection(dataframe):
    X = dataframe.drop(['label'], axis=1)
    y = dataframe.label
    return X, y

def preprocess_df(dataframe):
    # Normalize
    dataframe = dataframe/255

    #Reshape
    dataframe = dataframe.values.reshape(-1, 28, 28, 1)

    return dataframe

def binary_encode(dataframe):
    # binary encoding
    lb = LabelBinarizer()
    return lb.fit_transform(dataframe)

# Splitting train and test data 85/15
def split_data(inputdata_features, inputdata_targetvector):
    return train_test_split(
        inputdata_features, 
        inputdata_targetvector, 
        test_size=0.2, 
        random_state=100
    )

def main():
    df_train, df_test = helpers.import_data(const.TRAIN_DATASET_PATH, const.TEST_DATASET_PATH)

    X_train, y_train = feature_selection(df_train)
    X_test, y_test = feature_selection(df_test)

    visualization.data_visualization(y_train)

    X_train = preprocess_df(X_train)
    X_test = preprocess_df(X_test)
    # Encode labels to one hot vectors
    y_train = binary_encode(y_train)
    X_train, X_val, y_train, y_val = split_data(X_train, y_train)

    # visualizing first (6x6) 36 images
    visualization.image_visualization(X_train, 6, 6)

    modelsCNN = []
    for model_index, model in enumerate(const.CNN_MODEL_NAMES):
        if helpers.check_if_file_exist(model):
            print("MODEL EXIST.")
            modelsCNN.insert(model_index, load_model(model))
            print(f"MODEL - {model} - LOADED FROM DISK. index - {model_index}")
        else:
            # Describing the CNN model
            if(model_index==0):
                modelsCNN.insert(model_index, cnn.cnn_model_A())
            elif(model_index==1):
                modelsCNN.insert(model_index, cnn.cnn_model_B())
            elif(model_index==2):
                modelsCNN.insert(model_index, cnn.cnn_model_C())

            input_shape = (None, 28, 28, 1)
            modelsCNN[model_index].build(input_shape)
            modelsCNN[model_index].summary()

            history = cnn.cnn_model_training(modelsCNN[model_index], X_train, X_val, y_train, y_val)
            print("TRAINING COMPLETE.")

            save_model(modelsCNN[model_index], model)
            print(f"MODEL - {model} - SAVED TO DISK. index - {model_index}")

            # Training process analysis and visualization
            visualization.training_visualization(history)
        
        model_index += 1

    best_model_accuracy = 0
    best_model_index = None

    for curr_index, cnnModel in enumerate(modelsCNN): 
        current_model_accuracy = cnnModel.evaluate(X_val, y_val)[1]*100

        print("Model accuracy - " , current_model_accuracy, "%")
        
        if(current_model_accuracy > best_model_accuracy):
            best_model_accuracy = current_model_accuracy
            best_model_index = curr_index

        predictions = np.argmax(cnnModel.predict(X_test),axis=1)
        for i in range(len(predictions)):
            if(predictions[i] >= 9):
                predictions[i] += 1
        predictions[:5]

        classes = ["Label " + str(i) for i in range(25) if i != 9]
        print(classification_report(df_test['label'], predictions, target_names = classes))

        c_matrix = confusion_matrix(df_test['label'], predictions)
        c_matrix = pd.DataFrame(c_matrix , index = [i for i in range(25) if i != 9] , columns = [i for i in range(25) if i != 9])
        
        visualization.prediction_heatmap_visualization(c_matrix)

    print(f"Final Model Index - {best_model_index}")

    for optimizer in const.OPTIMIZERS:
        history, model = cnn.cnn_model_training_diff_optimizers(best_model_index, optimizer, X_train, X_val, y_train, y_val)
        visualization.training_visualization(history)

        print("Model accuracy - " , model.evaluate(X_val, y_val)[1]*100, "%")

        predictions = np.argmax(model.predict(X_test),axis=1)
        for i in range(len(predictions)):
            if(predictions[i] >= 9):
                predictions[i] += 1
        predictions[:5]

        classes = ["Label " + str(i) for i in range(25) if i != 9]
        print(classification_report(df_test['label'], predictions, target_names = classes))
    
    better_model = cnn.cnn_model_C()
    history = cnn.cnn_model_training_with_dataAug(better_model, X_train, X_val, y_train, y_val)
    print("TRAINING COMPLETE.")

    visualization.training_visualization(history)

    print("Model accuracy - " , better_model.evaluate(X_val, y_val)[1]*100, "%")

    predictions = np.argmax(better_model.predict(X_test),axis=1)
    for i in range(len(predictions)):
        if(predictions[i] >= 9):
            predictions[i] += 1
    predictions[:5]

    classes = ["Label " + str(i) for i in range(25) if i != 9]
    print(classification_report(df_test['label'], predictions, target_names = classes))


if __name__ == '__main__':
    main()