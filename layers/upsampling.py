import tensorflow as tf 
from tensorflow.keras.layers import Conv2D

def bn_act(inputs, activation='swish'):
    
    x = tf.keras.layers.BatchNormalization()(inputs)
    if activation:
        x = tf.keras.layers.Activation(activation)(x)
    
    return x

def decode(input_tensor, filters, scale = 2, activation = 'relu'):
    
    x1 = tf.keras.layers.Conv2D(filters, (1, 1), activation=activation, use_bias=False,
                                kernel_initializer='he_normal', padding = 'same')(input_tensor)
    
    x2 = tf.keras.layers.Conv2D(filters, (3, 3), activation=activation, 
                                use_bias=False, padding = 'same')(input_tensor)
    
    merge = tf.keras.layers.Add()([x1, x2])
    x = tf.keras.layers.UpSampling2D((scale, scale))(merge)

    skip_feature = tf.keras.layers.Conv2D(filters, (3, 3), activation=activation, use_bias=False,
                                        kernel_initializer='he_normal', padding = 'same')(merge)
    
    skip_feature = tf.keras.layers.Conv2D(filters, (1, 1), activation=activation, use_bias=False,
                                        kernel_initializer='he_normal', padding = 'same')(skip_feature)
    
    merge = tf.keras.layers.Add()([merge, skip_feature])

    x = bn_act(x, activation = activation)
    
    
    return x