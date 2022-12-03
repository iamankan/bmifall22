'''
This is a file where we make models using keras API. Not Sequential models.
'''

from pathlib import Path
import numpy as np
import argparse
import tensorflow as tf
import imageio.v2 as iio2
from sklearn import preprocessing
import cv2

from keras.utils.vis_utils import plot_model

def read_img(folder_name: Path, image_width: int, image_height: int, convert_to_single_channel: bool = True, channel_index_to_return: int = 0):
    print(f'Reading the images from {folder_name}')
    images = list()
    labels = list()
    for label in folder_name.iterdir():
        for sample in label.iterdir():
            iio_img = iio2.imread(sample)
            if convert_to_single_channel:
                iio_img = cv2.resize(iio_img[:,:,channel_index_to_return], (image_width, image_height))
            else:
                iio_img = cv2.resize(iio_img, (image_width, image_height))
            images.append([iio_img])
            labels.append(label.stem)
    return np.array(images), np.array([labels])

def load_image_data(folder_name: Path, image_width: int, image_height: int, convert_to_single_channel: bool = True, channel_index_to_return: int = 0):
    images_train, labels_train = read_img(folder_name=Path(folder_name) / 'train', image_width=image_width, image_height=image_height)
    print(f'images_train shape: {images_train.shape}, labels_train shape: {labels_train.shape}')
    
    images_val, labels_val = read_img(folder_name=Path(folder_name) / 'val', image_width=image_width, image_height=image_height)
    print(f'images_val shape: {images_val.shape}, labels_val shape: {labels_val.shape}')

    images_test, labels_test = read_img(folder_name=Path(folder_name) / 'test', image_width=image_width, image_height=image_height)
    print(f'images_test shape: {images_test.shape}, labels_test shape: {labels_test.shape}')

    return images_train, labels_train, images_test, labels_test, images_val, labels_val



def main():
    print(f'Hello this is main.')

    folder_name = f'../../../data/sars-cov2-ct-scan'

    image_height = 128
    image_width = 128
    num_of_channel = 1
    data_format_map = {'channels_first':(num_of_channel,image_width, image_height), 
                    'channels_last': (image_width, image_height, num_of_channel)}

    data_format='channels_first'
    

    images_train, labels_train, images_test, labels_test, images_val, labels_val = load_image_data(folder_name=folder_name, image_height= image_height, image_width=image_width)

    encoder = preprocessing.LabelEncoder().fit(['COVID', 'non-COVID'])
    bin_train_labels = encoder.transform(labels_train[0])
    bin_val_labels = encoder.transform(labels_val[0])
    bin_test_labels = encoder.transform(labels_test[0])


    modelobj = Model(input_shape=data_format_map[data_format])
    mymodel = modelobj.build(data_format=data_format)
    mymodel.summary()

    plot_model(mymodel, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    mymodel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

    history = modelobj.train(epochs=10, compiled_model=mymodel, x= images_train, y= bin_train_labels, batch_size=12, validation_data=
    (images_val, bin_val_labels))

    modelobj.visualize_history(history = history)

    



class Model(tf.keras.Model):
    def __init__(self, input_shape:tuple, API_FLAG:bool = True):
        super().__init__()
        self.API = API_FLAG
        self.INPUT_SHAPE = input_shape
        self.input_layer = tf.keras.Input(self.INPUT_SHAPE, name='input_layer')
        self.keras_layers = tf.keras.layers

    def build(self, mode: str = 'binary', data_format: str='channels_last') -> tf.keras.Model:
        print(f'Building model with keras API')
        # filters=512, kernel_size=3, strides=3, data_format=data_format, activation=tf.nn.relu, name='1st_convolution_layer'
        self.conv1 = self.keras_layers.Conv2D(64, kernel_size =(3,3), strides =(2,2), data_format=data_format, activation=tf.nn.relu, name='1st_convolution_layer')
        self.conv2 = self.keras_layers.Conv2D(32, kernel_size =(3,3), strides =(2,2), data_format=data_format, activation=tf.nn.relu, name='2nd_convolution_layer')
        
        '''
            Args:
                data_format: A string,
                one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
                keepdims: A boolean, whether to keep the spatial dimensions or not.
                If keepdims is False (default), the rank of the tensor is reduced for spatial dimensions. If keepdims is True, the spatial dimensions are retained with length 1. The behavior is the same as for tf.reduce_mean or np.mean.

            Input shape:

                If data_format='channels_last': 4D tensor with shape (batch_size, rows, cols, channels).
                If data_format='channels_first': 4D tensor with shape (batch_size, channels, rows, cols).
            Output shape:

                If keepdims=False: 2D tensor with shape (batch_size, channels).
                If keepdims=True:
                If data_format='channels_last': 4D tensor with shape (batch_size, 1, 1, channels)
                If data_format='channels_first': 4D tensor with shape (batch_size, channels, 1, 1)
        '''
        self.pool1 = self.keras_layers.MaxPooling2D(pool_size =(2,2), strides =(2, 2), data_format=data_format, name='1st_pooling_layer')
        self.gpool1 = self.keras_layers.GlobalAveragePooling2D(data_format=data_format, name='1st_global_pooling_layer')
        
        self.dense1 = self.keras_layers.Dense(4000, activation=tf.nn.relu, name='1st_dense_layer')
        self.dense2 = self.keras_layers.Dense(3000, activation=tf.nn.relu, name='2nd_dense_layer')
        self.dense3 = self.keras_layers.Dense(1000, activation=tf.nn.relu, name='3rd_dense_layer')
        self.output_block = self.keras_layers.Dense(1, activation=tf.nn.sigmoid, name='output_layer')

        # Starting to make the model
        x0 = self.conv1(self.input_layer)
        x0 = self.pool1(x0)
        x0 = self.conv2(x0)
        x0 = self.gpool1(x0)
        x0_flat = self.keras_layers.Flatten()(x0)
        x0_flat = self.dense1(x0_flat)
        x0_flat = self.keras_layers.Dropout(rate=0.3)(x0_flat)
        x0_flat = self.dense2(x0_flat)
        x0_flat = self.keras_layers.Dropout(rate=0.1)(x0_flat)
        # x0_flat = self.dense3(x0_flat)
        # x0_flat = self.keras_layers.Dropout(rate=0.1)(x0_flat)
        output_layer = self.output_block(x0_flat)

        fullmodel = tf.keras.Model(self.input_layer, output_layer)

        return fullmodel

    def train(self, epochs: int, compiled_model: tf.keras.Model, x: np.array, y: np.array, batch_size: int, validation_data: any,
    save_path: str= './model'):
        print(f'Starting to train.')
        Path.mkdir(Path(save_path), parents=True, exist_ok=True)
        save_model = tf.keras.callbacks.ModelCheckpoint(filepath= save_path+'/model_{epoch:02d}.h5')

        save_weight = tf.keras.callbacks.ModelCheckpoint(filepath=save_path+'/weight_{epoch:02d}.h5',
        save_weights_only=True)

        history = compiled_model.fit(x=x, y=y, batch_size=batch_size, validation_data=validation_data, epochs=epochs,
        callbacks=[save_model, save_weight])
        return history

    def visualize_history(self, history) -> None:
        # list all data in history
        print(history.history.keys())

        import matplotlib.pyplot as plt

        # summarize history for accuracy
        plt.plot(history.history['binary_accuracy'])
        plt.plot(history.history['val_binary_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('binary_accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


        

        






if __name__ == '__main__':
    main()