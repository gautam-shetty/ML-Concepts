import matplotlib.pyplot as plt
import seaborn as sns

def data_visualization(dataframe):
    plt.figure(figsize=(15,7))
    fig = sns.countplot(x=dataframe.index, data=dataframe, palette="icefire")
    plt.title("Number of digit classes")
    plt.show()

def image_visualization(dataframe, plotRows, plotCols):
    fig, axes = plt.subplots(plotRows, plotCols) 
    fig.set_size_inches(8, 8)
    k = 0
    for i in range(plotRows):
        for j in range(plotCols):
            axes[i,j].imshow(dataframe[k].reshape(28, 28) , cmap = "gray")
            k += 1
        plt.tight_layout()  
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

def prediction_heatmap_visualization(confusion_matrix):
    plt.figure(figsize = (15,15))
    sns.heatmap(confusion_matrix, cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
    plt.show()