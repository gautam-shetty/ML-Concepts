import matplotlib.pyplot as plt

def dataframe_visualization(dataframe, plot_kind):
    dataframe.plot(x = dataframe.columns[0], y=dataframe.columns[1], kind = plot_kind)
    plt.show()