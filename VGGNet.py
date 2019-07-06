from typing import Dict

import tensorflow as tf

from core import BaseDataSource, BaseModel
import util.gaze


class VGGNet(BaseModel):
    """An example neural network architecture."""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['left-eye']
        batch_size = 32
        # Trainable parameters should be specified within a known `tf.variable_scope`.
        # This tag is later used to specify the `learning_schedule` which describes when to train
        # which part of the network and with which learning rate.
        #
        # This network has two scopes, 'conv' and 'fc'. Though in practise it makes little sense to
        # train the two parts separately, this is possible.
        with tf.variable_scope('conv'):
            with tf.variable_scope('conv1'):
                x = tf.pad(x, [[0, 0], [0, 0], [0, 1], [0, 1]], "constant")
                x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2,
                                     padding='valid', data_format='channels_first')
               # self.summary.filters('filters', x)
                x = tf.nn.relu(x)
               # self.summary.feature_maps('features', x, data_format='channels_first')

                x = tf.pad(x, [[0, 0], [0, 0], [0, 1], [0, 1]], "constant")
                x = tf.layers.dropout(x, rate = 0.1, noise_shape = (batch_size, 128, 1, 1), training=False)
                x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1,
                                     padding='valid', data_format='channels_first')
                x = tf.nn.relu(x)
                x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='valid', data_format='channels_first')
               # self.summary.feature_maps('features', x, data_format='channels_first')

            with tf.variable_scope('conv2'):
                x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
                x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1,
                                     padding='valid', data_format='channels_first')
               # self.summary.feature_maps('features', x, data_format='channels_first')
                x = tf.nn.relu(x)
    
                #x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
               # x = tf.layers.dropout (x, rate=0.1, noise_shape=(batch_size, 512,1,1), training=False)
               # x = tf.layers.conv2d(x, filters=256, kernel_size=5, strides=2,
                #                     padding='same', data_format='channels_first')
               # self.summary.feature_maps('features', x, data_format='channels_first')
               # x = tf.nn.relu(x)
                x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='valid', data_format='channels_first')

            with tf.variable_scope('conv3'):
                x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
                x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1,
                                     padding='valid', data_format='channels_first')
               # self.summary.feature_maps('features', x, data_format='channels_first')
            
                x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
                x = tf.layers.dropout (x, rate=0.1, noise_shape=(batch_size, 512, 1, 1), training=False)
                x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1,
                                     padding='valid', data_format='channels_first')
               # self.summary.feature_maps('features', x, data_format='channels_first')
        
                x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
                x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1,
                                     padding='valid', data_format='channels_first')
               # self.summary.feature_maps('features', x, data_format='channels_first')
           
                x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
                x = tf.layers.dropout(x, rate=0.1, noise_shape=(batch_size, 512, 1, 1), training=False)
                x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='valid', data_format='channels_first')
            
                x = tf.layers.max_pooling2d(x, pool_size=3, strides=2,
                                        padding='same', data_format='channels_first')

            with tf.variable_scope('conv4'):
                x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
                x = tf.dropout(x, rate=0.1, noise_shape=(batch_size, 512, 1, 1), training=False)
                x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1,
                                 padding='valid', data_format='channels_first')
              
                x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
               # x = tf.layers.dropout(x, rate=0.1, noise_shape=(batch_size, 1024, 1, 1), training=False)
                x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2,
                                 padding='same', data_format='channels_first')
            
            
                 #x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
               # x = tf.layers.dropout(x, rate=0.1, noise_shape=(batch_size, 1024, 1, 1), training=False)
                x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2,
                                 padding='same', data_format='channels_first')
                x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same', data_format='channels_first')


            with tf.variable_scope('conv5'):
                 #x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
                x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2,
                                 padding='same', data_format='channels_first')

                x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
               # x = tf.layers.dropout(x, rate=0.1, noise_shape=(batch_size, 1024, 1, 1), training=False)
                x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2,
                                 padding='valid', data_format='channels_first')


                x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "constant")
                x = tf.layers.dropout(x, rate=0.1, noise_shape=(batch_size, 512, 1, 1), training=False)
                x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1,
                                 padding='valid', data_format='channels_first')
                x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same', data_format='channels_first')



        with tf.variable_scope('fc'):
            # Flatten the 50 feature maps down to one vector
            x = tf.contrib.layers.flatten(x)

            # FC layer
            x = tf.layers.dense(x, units=4096, activation='relu', name='fc5')
            x = tf.layers.dense(x, units=4096, activation='relu', name='fc6')
            x = tf.layers.dense(x, units=1024, activation='softmax', name='fc7')
            self.summary.histogram('fc7/activations', x)

            # Directly regress two polar angles for gaze direction
            x = tf.layers.dense(x, units=2, name='fc8')
            self.summary.histogram('fc8/activations', x)

        # Define outputs
        loss_terms = {}
        metrics = {}
        if 'gaze' in input_tensors:
            y = input_tensors['gaze']
            with tf.variable_scope('mse'):  # To optimize
                loss_terms['gaze_mse'] = tf.reduce_mean(tf.squared_difference(x, y))
            with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
                metrics['gaze_angular'] = util.gaze.tensorflow_angular_error_from_pitchyaw(x, y)
        return {'gaze': x}, loss_terms, metrics
