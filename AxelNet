from typing import Dict

import tensorflow as tf

from core import BaseDataSource, BaseModel
import util.gaze


class AlexNet(BaseModel):
    """An example neural network architecture."""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['head']

     
        with tf.variable_scope('conv'):
            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d(x, filters=96, kernel_size=3, strides=2,
                                     padding='same', data_format='channels_first')
                x = tf.nn.relu(x)
                #  self.summary.filters('filters', x)
                self.summary.feature_maps('features', x, data_format='channels_first')

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2,
                                     padding='same', data_format='channels_first')
                x = tf.layers.max_pooling2d(x, pool_size = 3, strides = 3, data_format='channels_first')
             #   self.summary.feature_maps('features', x, data_format='channels_first')

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, filters=384, kernel_size=3, strides=2,
                                     padding='same', data_format='channels_first')
                x = tf.layers.conv2d(x, filters=384, kernel_size=3, strides=2,
                                     padding='same', data_format='channels_first')
                x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1,
                                     padding='same', data_format='channels_first')
                x = tf.layers.max_pooling2d(x, pool_size =3, strides=2,
                                     padding='same', data_format='channels_first')
           

        with tf.variable_scope('fc'):
            # Flatten the 50 feature maps down to one vector
            x = tf.contrib.layers.flatten(x)

            # FC layer
            x = tf.layers.dense(x, units=4096, name='fc4')
            x = tf.layers.dense(x, units=4096, name='fc5')
            x = tf.layers.dense(x, units=42, name='fc6')
            #x = tf.layers.dense(x, units=64, name='fc7')
            self.summary.histogram('fc6/activations', x)

            # Directly regress two polar angles for gaze direction
            x = tf.layers.dense(x, units=2, name='fc7')
            self.summary.histogram('fc7/activations', x)

        # Define outputs
        loss_terms = {}
        metrics = {}
        if 'gaze' in input_tensors:
            y = input_tensors['gaze']
            with tf.variable_scope('mse'):  # To optimize
                loss_terms['gaze_mse'] = tf.reduce_mean(tf.squared_difference(x, y))
            with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
