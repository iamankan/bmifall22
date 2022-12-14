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
import time

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

    image_height = 227
    image_width = 227
    num_of_channel = 1
    data_format_map = {'channels_first':(num_of_channel,image_width, image_height), 
                    'channels_last': (image_width, image_height, num_of_channel)}

    data_format='channels_first'
    

    # images_train, labels_train, images_test, labels_test, images_val, labels_val = load_image_data(folder_name=folder_name, image_height= image_height, image_width=image_width)

    images_train = np.load('images_train.npz.npy', allow_pickle=True)
    labels_train = np.load('labels_train.npz.npy', allow_pickle=True)

    images_val = np.load('images_val.npz.npy', allow_pickle=True)
    labels_val = np.load('labels_val.npz.npy', allow_pickle=True)


    encoder = preprocessing.LabelEncoder().fit(['COVID', 'non-COVID'])
    bin_train_labels = encoder.transform(labels_train[0])
    bin_val_labels = encoder.transform(labels_val[0])


    modelobj = Model(input_shape=data_format_map[data_format])
    mymodel = modelobj.build2(data_format=data_format)
    mymodel.summary()

    slno = time.time_ns()

    plot_model(mymodel, to_file=f'model_plot_{slno}.png', show_shapes=True, show_layer_names=True)

    mymodel.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

    history = modelobj.train(epochs=100, compiled_model=mymodel, x= images_train, y= bin_train_labels, batch_size=12, validation_data=
    (images_val, bin_val_labels), save_path=f'model_{slno}')

    modelobj.visualize_history(history = history, save_path=f'my_history_{slno}.npy')

    



class Model(tf.keras.Model):
    def __init__(self, input_shape:tuple, API_FLAG:bool = True):
        super().__init__()
        self.API = API_FLAG
        self.INPUT_SHAPE = input_shape
        self.input_layer = tf.keras.Input(self.INPUT_SHAPE, name='input_layer')
        self.keras_layers = tf.keras.layers
    
    def fire_module(self, layer, data_format, fire_id=0, squeeze=16, expand=64):
        print(f'Fire module - {fire_id}')
        if data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        name = f'fire_{fire_id}'

        layer = self.keras_layers.Convolution2D(squeeze, (1,1), padding='valid', data_format=data_format, name=name+'_conv2d_1')(layer)
        layer = self.keras_layers.Activation('elu', name=name+'_activation_1')(layer)

        left = self.keras_layers.Convolution2D(expand, (1,1), padding='valid', data_format=data_format, name=name+'_conv2d_2_left')(layer)
        left = self.keras_layers.Activation('elu', name=name+'_activation_2left')(left)

        right = self.keras_layers.Convolution2D(expand, (1,1), padding='valid', data_format=data_format, name=name+'_conv2d_2_right')(layer)
        right = self.keras_layers.Activation('elu', name=name+'_activation_2right')(right)

        x = self.keras_layers.concatenate([left, right], axis=channel_axis, name=name+'_concar_fire')

        return x

    def build2(self, mode: str = 'binary', data_format: str='channels_last') -> tf.keras.Model:
        print(f'Building model with keras API-2')
        '''
        https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py
        '''
        
        x = self.keras_layers.Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', data_format=data_format, name='conv1')(self.input_layer)
        x = self.keras_layers.Activation('elu', name='elu_conv1')(x)
        x = self.keras_layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = self.fire_module(x, fire_id=2, squeeze=16, expand=64, data_format=data_format)
        x = self.fire_module(x, fire_id=3, squeeze=16, expand=64, data_format=data_format)
        x = self.keras_layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = self.fire_module(x, fire_id=4, squeeze=32, expand=128, data_format=data_format)
        x = self.fire_module(x, fire_id=5, squeeze=32, expand=128, data_format=data_format)
        x = self.keras_layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = self.fire_module(x, fire_id=6, squeeze=48, expand=192, data_format=data_format)
        x = self.fire_module(x, fire_id=7, squeeze=48, expand=192, data_format=data_format)
        x = self.fire_module(x, fire_id=8, squeeze=64, expand=256, data_format=data_format)
        x = self.fire_module(x, fire_id=9, squeeze=64, expand=256, data_format=data_format)

        x = self.keras_layers.Dropout(0.5, name='drop9')(x)

        x = self.keras_layers.Convolution2D(1000, (2,2), padding='valid', activation=tf.nn.elu, data_format=data_format, name='conv10')(x)
        x = self.keras_layers.Activation('elu', name='relu_conv10')(x)
        x = self.keras_layers.GlobalAveragePooling2D()(x)
        # x = self.keras_layers.Activation('sigmoid', name='loss')(x)
        x= self.keras_layers.Flatten(data_format=data_format)(x)
        # x = self.keras_layers.Dense(4000, activation=tf.nn.elu, name='dense_4000_1')(x)
        output = self.keras_layers.Dense(1, activation=tf.nn.sigmoid)(x)

        return tf.keras.Model(self.input_layer, output)

    def build(self, mode: str = 'binary', data_format: str='channels_last') -> tf.keras.Model:
        print(f'Building model with keras API')
        # filters=512, kernel_size=3, strides=3, data_format=data_format, activation=tf.nn.relu, name='1st_convolution_layer'
        self.conv1 = self.keras_layers.Conv2D(128, kernel_size =(7,7), strides =(2,2), data_format=data_format, activation=tf.nn.relu, name='1st_convolution_layer')
        self.conv2 = self.keras_layers.Conv2D(64, kernel_size =(7,7), strides =(2,2), data_format=data_format, activation=tf.nn.relu, name='2nd_convolution_layer')
        self.conv3 = self.keras_layers.Conv2D(32, kernel_size =(7,7), strides =(2,2), data_format=data_format, activation=tf.nn.relu, name='3rd_convolution_layer')
        self.conv4 = self.keras_layers.Conv2D(16, kernel_size =(7,7), strides =(2,2), data_format=data_format, activation=tf.nn.relu, name='4th_convolution_layer')
        self.conv5 = self.keras_layers.Conv2D(8, kernel_size =(7,7), strides =(2,2), data_format=data_format, activation=tf.nn.relu, name='5th_convolution_layer')
        
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
        self.gpool1 = self.keras_layers.GlobalAveragePooling2D(data_format=data_format, keepdims=True, name='1st_global_pooling_layer')
        
        self.dense1 = self.keras_layers.Dense(4000, activation=tf.nn.relu, name='1st_dense_layer')
        self.dense2 = self.keras_layers.Dense(3000, activation=tf.nn.relu, name='2nd_dense_layer')
        self.dense3 = self.keras_layers.Dense(1000, activation=tf.nn.relu, name='3rd_dense_layer')
        self.output_block = self.keras_layers.Dense(1, activation=tf.nn.sigmoid, name='output_layer')

        # Starting to make the model
        x0 = self.conv1(self.input_layer)
        # x0 = self.pool1(x0)
        x0 = self.conv2(x0)
        # x0 = self.pool1(x0)
        x0 = self.conv3(x0)
        # x0 = self.pool1(x0)
        x0 = self.conv4(x0)
        # x0 = self.pool1(x0)
        # x0 = self.conv5(x0)

        # x0 = self.gpool1(x0)
        x0_flat = self.keras_layers.Flatten()(x0)
        x0_flat = self.dense1(x0_flat)
        x0_flat = self.keras_layers.Dropout(rate=0.5)(x0_flat)
        x0_flat = self.dense2(x0_flat)
        x0_flat = self.keras_layers.Dropout(rate=0.3)(x0_flat)
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

        lr_sched = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * (0.80 ** np.floor(epoch / 2)))

        history = compiled_model.fit(x=x, y=y, batch_size=batch_size, validation_data=validation_data, epochs=epochs,
        callbacks=[save_model, save_weight])
        return history

    def visualize_history(self, history, save_path: str= 'my_history_0.npy') -> None:
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
        plt.savefig('accuracy.png')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig('loss.png')

        np.save(save_path, history.history)


        

        






if __name__ == '__main__':
    main()