import tensorflow as tf
from sklearn import preprocessing
import kerasmodel1 as km1
import numpy as np
from pathlib import Path

def eval(fmodel, images_test, bin_test_labels):
    print(f'{fmodel}')
    model = tf.keras.models.load_model(fmodel)
    model.evaluate(x= images_test, y=bin_test_labels)


def main():
    print(f'This is main')
    # model = tf.keras.models.load_model(f'model_1670087179796380400/model_100.h5')
    # model.summary()

    images_test = np.load('images_test.npz.npy', allow_pickle=True)
    labels_test = np.load('labels_test.npz.npy', allow_pickle=True)

    encoder = preprocessing.LabelEncoder().fit(['COVID', 'non-COVID'])
    bin_test_labels = encoder.transform(labels_test[0])

    # model.evaluate(x= images_test, y=bin_test_labels)
    p= Path('model_1670087179796380400')
    for mod_sam in p.iterdir():
        if mod_sam.stem.startswith('model'):
            eval(fmodel=mod_sam, images_test=images_test, bin_test_labels=bin_test_labels)


if __name__ == "__main__":
    main()