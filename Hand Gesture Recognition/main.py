import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

#Constants
train_dataset_path = 'assets/sign_mnist_train.csv'
test_dataset_path = 'assets/sign_mnist_test.csv'

epoch = 25
batch_size = 200


def import_data(dataset_train, dataset_test):
    train_dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__), dataset_train))
    test_dataframe = pd.read_csv(os.path.join(os.path.dirname(__file__), dataset_test))
    return train_dataframe, test_dataframe

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

def data_visualization(dataframe):
    plt.figure(figsize=(15,7))
    fig = sns.countplot(x=dataframe.index, data=dataframe, palette="icefire")
    plt.title("Number of digit classes")
    plt.show()

def image_visualization(dataframe):
    fig = plt.figure(figsize=(15, 7))
    rows = 4
    columns = 6
    for x in range(0, (rows*columns)):
        fig.add_subplot(rows, columns, x+1)
        plt.imshow(dataframe[x][:,:,0],cmap="gray")
        plt.axis('off')
        plt.title(x)
    plt.show()

def training_visualization(history):
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    fig.set_size_inches(12,4)

    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Training Accuracy vs Validation Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')

    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Training Loss vs Validation Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def main():
    df_train, df_test = import_data(train_dataset_path, test_dataset_path)

    X_train, y_train = feature_selection(df_train)
    X_test, y_test = feature_selection(df_test)

    # data_visualization(y_train)

    X_train = preprocess_df(X_train)
    X_test = preprocess_df(X_test)
    # Encode labels to one hot vectors
    y_train = binary_encode(y_train)

    X_train, X_val, y_train, y_val = split_data(X_train, y_train)

    modelCNN = cnn_model()

    adam_optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999)
    modelCNN.compile(optimizer = adam_optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

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

    history = modelCNN.fit_generator(
        train_datagen.flow(X_train,y_train, batch_size=batch_size), 
        epochs = epoch,
        validation_data = (X_val,y_val), 
        steps_per_epoch = X_train.shape[0] // batch_size
    )

    training_visualization(history)

main()