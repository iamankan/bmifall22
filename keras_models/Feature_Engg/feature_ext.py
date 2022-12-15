import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn import preprocessing
import pandas as pd




def extract_feature(model: tf.keras.Model, images: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    print(f'Extracting models')
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    features = intermediate_model.predict(images)
    # print(features.shape)
    # print(labels.shape)
    dataset = np.hstack([features, labels])
    cols = [f'feat_{i}' for i in range(features.shape[1])]
    cols.append('class')
    dfshape = len(cols)
    return pd.DataFrame(dataset, columns=cols), dfshape


def main():
    print('Hello')
    model_path = Path(f'../model_1670087179796380400/model_34.h5')
    model = tf.keras.models.load_model(filepath=model_path)
    # model.summary()
    # Load train images and labels
    group = 'train'
    train_images = np.load(Path(f'../images_{group}.npz.npy'), allow_pickle=True)
    train_labels = np.load(Path(f'../labels_{group}.npz.npy'), allow_pickle=True)
    encoder = preprocessing.LabelEncoder().fit(['COVID', 'non-COVID'])
    bin_train_labels = encoder.transform(train_labels[0])
    bin_train_labels = np.reshape(bin_train_labels, (-1, 1))
    
    traindataset, dfshape = extract_feature(model=model, images=train_images, labels=bin_train_labels)
    traindataset.to_csv(f'{group}_{dfshape-1}.csv', index=None)
    print(f'traindataset.shape: {traindataset.shape}')

    # Load test images and labels
    group = 'test'
    test_images = np.load(Path(f'../images_{group}.npz.npy'), allow_pickle=True)
    test_labels = np.load(Path(f'../labels_{group}.npz.npy'), allow_pickle=True)
    encoder = preprocessing.LabelEncoder().fit(['COVID', 'non-COVID'])
    bin_test_labels = encoder.transform(test_labels[0])
    bin_test_labels = np.reshape(bin_test_labels, (-1, 1))
    
    testdataset, dfshape = extract_feature(model=model, images=test_images, labels=bin_test_labels)
    testdataset.to_csv(f'{group}_{dfshape-1}.csv', index=None)
    print(f'testdataset.shape: {testdataset.shape}')

    # Load test images and labels
    group = 'val'
    val_images = np.load(Path(f'../images_{group}.npz.npy'), allow_pickle=True)
    val_labels = np.load(Path(f'../labels_{group}.npz.npy'), allow_pickle=True)
    encoder = preprocessing.LabelEncoder().fit(['COVID', 'non-COVID'])
    bin_val_labels = encoder.transform(val_labels[0])
    bin_val_labels = np.reshape(bin_val_labels, (-1, 1))
    
    valdataset, dfshape = extract_feature(model=model, images=val_images, labels=bin_val_labels)
    valdataset.to_csv(f'{group}_{dfshape-1}.csv', index=None)
    print(f'valdataset.shape: {valdataset.shape}')

    alldataset = pd.concat([traindataset, valdataset, testdataset], axis=0)
    print(f'alldataset.shape: {alldataset.shape}')
    alldataset.to_csv(f'all_{dfshape-1}.csv', index=None)
    




if __name__ == '__main__':
    main()