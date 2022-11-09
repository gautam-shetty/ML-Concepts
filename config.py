import os
import requests
from io import BytesIO
from zipfile import ZipFile

###
### NOTE / TO-DO
###
# Yet to implemet for Kaggle dataset using Kaggle API

## Configuration Start
## datasetInfo format - dataset_name, dataset_provider, dataset_url
datasetInfo = [
    ["NS_PM2.5_DATA.csv", "NovaScotia.ca", "https://data.novascotia.ca/api/views/ddk3-mz42/rows.csv?accessType=DOWNLOAD&bom=true&format=true"],
    ["Traffic_Volumes_Data.csv", "NovaScotia.ca", "https://data.novascotia.ca/api/views/8524-ec3n/rows.csv?accessType=DOWNLOAD&bom=true&format=true"],
]

datasetFolder = 'Traffic Pollution Analysis/assets'
## Configuration End


dir_path = os.path.join(os.getcwd(), datasetFolder)

def downloadDataset(dataset, fileType):
    print(f'Downloading Dataset - {dataset[0]}')

    if fileType == 'csv':
        response = requests.get(dataset[2], stream = True)
        if response.status_code == 200:
            with open(os.path.join(dir_path, dataset[0]), 'wb') as data: 
                for block in response.iter_content(chunk_size=1024):
                    data.write(block)
            print(f' Dataset {dataset[0]} downloaded.')
        else:
            print(f' Dataset {dataset[0]} download failed!!')

    # ZIP File
    else:
        response = requests.get(dataset[2])
        if response.status_code == 200:
            archive = ZipFile(BytesIO(response.content))
            archive.extractall(dir_path)
            print(f' Dataset {dataset[0]} downloaded.')
        else:
            print(f' Dataset {dataset[0]} download failed!!')
    

def main():

    if not os.path.exists(dir_path):
        os.mkdir(dir_path, 0o666)
    for dataset in datasetInfo:
        if dataset[0][-3:] == 'zip':
            downloadDataset(dataset, 'zip')
        else:
            downloadDataset(dataset, 'csv')
    print('Datasets download complete.')

main()