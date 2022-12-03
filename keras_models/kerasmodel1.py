'''
This is a file where we make models using keras API. Not Sequential models.
'''

from pathlib import Path
import numpy as np
import argparse
import tensorflow as tf
import imageio.v2 as iio2
import cv2

def read_img(folder_name: Path, image_width: int, image_height: int, convert_to_single_channel: bool = True, channel_index_to_return: int = 0) -> np.array:
    print(f'Reading the images from {folder_name}')
    images = list()
    for label in folder_name.iterdir():
        for sample in label.iterdir():
            iio_img = iio2.imread(sample)
            if convert_to_single_channel:
                iio_img = cv2.resize(iio_img[:,:,channel_index_to_return], (image_width, image_height))
            else:
                iio_img = cv2.resize(iio_img, (image_width, image_height))
            images.append(iio_img)
    return np.array(images)

def load_image_data(folder_name: Path, image_width: int, image_height: int, convert_to_single_channel: bool = True, channel_index_to_return: int = 0):
    images_train = read_img(folder_name=Path(folder_name) / 'train', image_width=image_width, image_height=image_height)
    print(f'images_train shape: {images_train.shape}')
    
    images_val = read_img(folder_name=Path(folder_name) / 'val', image_width=image_width, image_height=image_height)
    print(f'images_val shape: {images_val.shape}')

    images_test = read_img(folder_name=Path(folder_name) / 'test', image_width=image_width, image_height=image_height)
    print(f'images_test shape: {images_test.shape}')

    return images_train, images_test, images_val



def main():
    print(f'Hello this is main.')

    folder_name = f'../../../data/sars-cov2-ct-scan'

    image_height = 255
    image_width = 255
    num_of_channel = 1
    data_format_map = {'channels_first':(num_of_channel,image_width, image_height), 
                    'channels_last': (image_width, image_height, num_of_channel)}

    data_format='channels_last'
    

    # train, test, val = load_image_data(folder_name=folder_name, image_height= image_height, image_width=image_width)
    modelobj = Model(input_shape=data_format_map[data_format])
    mymodel = modelobj.build(data_format=data_format)
    mymodel.summary()

class Model(tf.keras.Model):
    def __init__(self, input_shape:tuple, API_FLAG:bool = True):
        super().__init__()
        self.API = API_FLAG
        self.INPUT_SHAPE = input_shape
        self.input_layer = tf.keras.Input(self.INPUT_SHAPE, name='input_layer')
        self.keras_layers = tf.keras.layers

    def build(self, mode: str = 'binary', data_format: str='channels_last') -> tf.keras.Model:
        print(f'Building model with keras API')
        self.conv1 = self.keras_layers.Conv2D(filters=32, kernel_size=3, strides=2, data_format=data_format, name='1st_convolution_layer')
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
        self.pool1 = self.keras_layers.GlobalAveragePooling2D(data_format=data_format, name='1st_pooling_layer')
        self.dense1 = self.keras_layers.Dense(400, activation=tf.nn.relu, name='1st_dense_layer')
        self.dense2 = self.keras_layers.Dense(100, activation=tf.nn.relu, name='2nd_dense_layer')
        self.dense3 = self.keras_layers.Dense(10, activation=tf.nn.relu, name='3rd_dense_layer')
        self.output_block = self.keras_layers.Dense(1, activation=tf.nn.sigmoid, name='output_layer')

        # Starting to make the model
        x0 = self.input_layer
        x0 = self.keras_layers.BatchNormalization()(x0)
        x0 = self.conv1(x0)
        x0 = self.pool1(x0)
        x0_flat = self.keras_layers.Flatten()(x0)
        x0_flat = self.dense1(x0_flat)
        x0_flat = self.keras_layers.Dropout(rate=0.1)(x0_flat)
        x0_flat = self.dense2(x0_flat)
        x0_flat = self.keras_layers.Dropout(rate=0.1)(x0_flat)
        x0_flat = self.dense3(x0_flat)
        x0_flat = self.keras_layers.Dropout(rate=0.1)(x0_flat)
        output_layer = self.output_block(x0_flat)

        fullmodel = tf.keras.Model(self.input_layer, output_layer)

        return fullmodel


        

        






if __name__ == '__main__':
    main()