import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator

import helpers
import visualization
import constants as const

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
        test_size=0.15, 
        random_state=42
    )

def cnn_model():
    model = Sequential()

    model.add(Conv2D(
        filters=75,
        kernel_size=(3,3),
        strides=1,
        padding='same',
        activation='relu',
        input_shape=(28,28,1)
    ))
    model.add(MaxPool2D(
        pool_size=(2,2),
        strides = 2,
        padding='same'
    ))

    model.add(Conv2D(
        filters=50,
        kernel_size=(3,3),
        strides=1,
        padding='same',
        activation='relu'
    ))
    model.add(MaxPool2D(
        pool_size=(2,2),
        strides = 2,
        padding='same'
    ))

    model.add(Conv2D(
        filters=25,
        kernel_size=(3,3),
        strides=1,
        padding='same',
        activation='relu'
    ))
    model.add(MaxPool2D(
        pool_size=(2,2),
        strides = 2,
        padding='same'
    ))

    model.add(Flatten())

    model.add(Dense(units = 512 , activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units = 24 , activation = 'softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    return model

def cnn_model_training(model, X_train, X_val, y_train, y_val):
    adam_optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer = adam_optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    # Data Augmentation to prevent overfitting
    train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=15,  # randomly rotate images in the range 15 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.15,  # randomly shift images horizontally 15%
        height_shift_range=0.15,  # randomly shift images vertically 15%
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    train_datagen.fit(X_train)

    history = model.fit_generator(
        train_datagen.flow(X_train,y_train, batch_size=const.BATCH_SIZE), 
        epochs = const.EPOCH,
        validation_data = (X_val,y_val), 
        steps_per_epoch = X_train.shape[0] // const.BATCH_SIZE
    )

    return history

def main():
    df_train, df_test = helpers.import_data(const.TRAIN_DATASET_PATH, const.TEST_DATASET_PATH)

    X_train, y_train = feature_selection(df_train)
    X_test, y_test = feature_selection(df_test)

    # visualization.data_visualization(y_train)

    X_train = preprocess_df(X_train)
    X_test = preprocess_df(X_test)
    # Encode labels to one hot vectors
    y_train = binary_encode(y_train)
    X_train, X_val, y_train, y_val = split_data(X_train, y_train)

    # visualizing first (6x6) 36 images
    # visualization.image_visualization(X_train, 6, 6)

    if helpers.check_if_file_exist(const.CNN_MODEL_NAME):
        print("MODEL EXIST.")
        modelCNN = load_model("model.h5")
        print("MODEL LOADED FROM DISK.")
    else:
        # Describing the CNN model
        modelCNN = cnn_model()
        modelCNN.summary()

        history = cnn_model_training(modelCNN, X_train, X_val, y_train, y_val)
        print("TRAINING COMPLETE.")

        save_model(modelCNN, "model.h5")
        print("MODEL SAVED TO DISK.")

        # Training process analysis and visualization
        visualization.training_visualization(history)

    print("Model accuracy - " , modelCNN.evaluate(X_val,y_val)[1]*100 , "%")

    predictions = np.argmax(modelCNN.predict(X_test),axis=1)
    for i in range(len(predictions)):
        if(predictions[i] >= 9):
            predictions[i] += 1
    predictions[:5]

    classes = ["Class " + str(i) for i in range(25) if i != 9]
    print(classification_report(df_test['label'], predictions, target_names = classes))

    c_matrix = confusion_matrix(df_test['label'], predictions)
    c_matrix = pd.DataFrame(c_matrix , index = [i for i in range(25) if i != 9] , columns = [i for i in range(25) if i != 9])
    
    visualization.prediction_heatmap_visualization(c_matrix)

if __name__ == '__main__':
    main()