from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from keras.preprocessing.image import ImageDataGenerator

import constants as const

# Baisc Model
def cnn_model_A():
    model = Sequential()

    # Layer 1
    model.add(Conv2D(
        filters=64,
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

    # Layer 2
    model.add(Conv2D(
        filters=32,
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

    # Layer 3
    model.add(Conv2D(
        filters=16,
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

    model.add(Dense(units = 64 , activation = 'relu'))
    model.add(Dense(units = 24 , activation = 'softmax'))

    return model

# Model with batch normalization
def cnn_model_B():
    model = Sequential()

    # Layer 1
    model.add(BatchNormalization())
    model.add(Conv2D(
        filters=64,
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

    # Layer 2
    model.add(BatchNormalization())
    model.add(Conv2D(
        filters=32,
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

    # Layer 3
    model.add(BatchNormalization())
    model.add(Conv2D(
        filters=16,
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

    model.add(Dense(units = 64 , activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units = 24 , activation = 'softmax'))

    return model

# Model with batch normalization and dropout layers
def cnn_model_C():
    model = Sequential()

    # Layer 1
    model.add(BatchNormalization())
    model.add(Conv2D(
        filters=64,
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
    model.add(Dropout(const.DROPOUT_RATE))

    # Layer 2
    model.add(BatchNormalization())
    model.add(Conv2D(
        filters=32,
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
    model.add(Dropout(const.DROPOUT_RATE))

    # Layer 3
    model.add(BatchNormalization())
    model.add(Conv2D(
        filters=16,
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
    model.add(Dropout(const.DROPOUT_RATE))

    model.add(Flatten())

    model.add(Dense(units = 64 , activation = 'relu'))
    model.add(Dropout(const.DROPOUT_RATE))

    model.add(Dense(units = 24 , activation = 'softmax'))

    return model


def cnn_model_training(model, X_train, X_val, y_train, y_val):
        
    selected_optimizer = Adam() # Default

    model.compile(optimizer = selected_optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, 
        y_train, 
        batch_size = const.BATCH_SIZE, 
        epochs = const.EPOCH, 
        verbose = 2, # Will log the epoch number only
        validation_data = (X_val, y_val)
    )

    return history

def cnn_model_training_with_dataAug(model, X_train, X_val, y_train, y_val):
        
    selected_optimizer = Adam() # Default

    model.compile(optimizer = selected_optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

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
        train_datagen.flow(X_train, y_train, batch_size=const.BATCH_SIZE), 
        epochs = const.EPOCH,
        verbose = 2, # Will log the epoch number only
        validation_data = (X_val, y_val), 
        steps_per_epoch = X_train.shape[0] // const.BATCH_SIZE
    )

    return history

def cnn_model_training_diff_optimizers(model_index, optimizer, X_train, X_val, y_train, y_val):
    
    model = None
    # Describing the CNN model
    if(model_index==0):
        model = cnn_model_A()
    elif(model_index==1):
        model = cnn_model_B()
    elif(model_index==2):
        model = cnn_model_C()

    selected_optimizer = None
    if(optimizer == 'Adam'):
        selected_optimizer = Adam()
    elif(optimizer == 'SGD'):
        selected_optimizer = SGD(learning_rate=const.LEARNING_RATE)
    elif(optimizer == 'RMSprop'):
        selected_optimizer = RMSprop(learning_rate=const.LEARNING_RATE, epsilon=const.EPSILON)
    elif(optimizer == 'Adagrad'):
        selected_optimizer = Adagrad(learning_rate=const.LEARNING_RATE, epsilon=const.EPSILON)
        
    model.compile(optimizer = selected_optimizer, loss = "categorical_crossentropy", metrics=["accuracy"]) 

    history = model.fit(
        X_train, 
        y_train, 
        batch_size = const.BATCH_SIZE, 
        epochs = const.EPOCH, 
        verbose = 2, # Will log the epoch number only
        validation_data = (X_val, y_val)
    )
    print(f"TRAINING COMPLETE FOR {optimizer} OPTIMIZER.")
    return history, model