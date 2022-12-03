import tensorflow as tf
from sklearn import preprocessing
import kerasmodel1 as km1

def main():
    print(f'This is main')
    model = tf.keras.models.load_model(f'model/model_10.h5')
    model.summary()

    folder_name = f'../../../data/sars-cov2-ct-scan'

    image_height = 255
    image_width = 255
    num_of_channel = 1
    data_format_map = {'channels_first':(num_of_channel,image_width, image_height), 
                    'channels_last': (image_width, image_height, num_of_channel)}

    data_format='channels_first'
    

    images_train, labels_train, images_test, labels_test, images_val, labels_val = km1.load_image_data(folder_name=folder_name, image_height= image_height, image_width=image_width)

    encoder = preprocessing.LabelEncoder().fit(['COVID', 'non-COVID'])
    bin_test_labels = encoder.transform(labels_test[0])

    model.evaluate(x= images_test, y=bin_test_labels)

if __name__ == "__main__":
    main()