import tensorflow as tf 

def convformer(input_tensor, filters, padding = "same"):
    
    x = tf.keras.layers.LayerNormalization()(input_tensor)
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size = (3,3), padding = padding)(x)
    # x = x1 + x2 + x3
    x = tf.keras.layers.Attention()([x, x, x])
    out = tf.keras.layers.Add()([x, input_tensor])
    
    x1 = tf.keras.layers.Dense(filters, activation = "gelu")(out)
    x1 = tf.keras.layers.Dense(filters)(x1)
    out_tensor = tf.keras.layers.Add()([out, x1])
    return out_tensor