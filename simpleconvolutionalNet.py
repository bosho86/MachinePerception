from typing import Dict

import tensorflow as tf

from core import BaseDataSource, BaseModel
import util.gaze


class NewModel(BaseModel):
    """An example neural network architecture."""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['left-eye']

        # Trainable parameters should be specified within a known `tf.variable_scope`.
        # This tag is later used to specify the `learning_schedule` which describes when to train
        # which part of the network and with which learning rate.
        #
        # This network has two scopes, 'conv' and 'fc'. Though in practise it makes little sense to
        # train the two parts separately, this is possible.
        with tf.variable_scope('conv'):
            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d(x, filters=128, kernel_size=7, strides=2,
                                     padding='same', data_format='channels_first')
               # x = tf.nn.relu(x)
                x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, data_format='channels_first')
               # self.summary.filters('filters', x)
                self.summary.feature_maps('features', x, data_format='channels_first')

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, filters=256, kernel_size=7, strides=2,
                                     padding='same', data_format='channels_first')
               
                x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, data_format='channels_first')
                self.summary.feature_maps('features', x, data_format='channels_first')


        with tf.variable_scope('fc'):
            # Flatten the 50 feature maps down to one vector
            x = tf.contrib.layers.flatten(x)

            # FC layer
            x = tf.layers.dense(x, units=512, name='fc3')
            x = tf.layers.dense(x, units=64, name='fc4')
            self.summary.histogram('fc4/activations', x)

            # Directly regress two polar angles for gaze direction
            x = tf.layers.dense(x, units=2, name='fc5')
            self.summary.histogram('fc5/activations', x)

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
