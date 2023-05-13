import tensorflow as tf 
from tensorflow.keras.layers import Conv2D

def bn_act(inputs, activation='swish'):
    
    x = tf.keras.layers.BatchNormalization()(inputs)
    if activation:
        x = tf.keras.layers.Activation(activation)(x)
    
    return x

def conv_bn_act(inputs, filters, kernel_size, strides=(1, 1), activation='relu', padding='same'):
    
    x = Conv2D(filters, kernel_size=kernel_size, padding=padding)(inputs)
    x = bn_act(x, activation=activation)
    
    return x

def merge(l, filters=None):
    if filters is None:
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = l[0].shape[channel_axis]
    
    x = tf.keras.layers.Add()([l[0],l[1]])
    
    # x = block(x, filters)
    
    return x