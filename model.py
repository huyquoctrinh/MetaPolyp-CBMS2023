import tensorflow as tf
from keras_cv_attention_models import caformer
from layers.upsampling import decode
from layers.convformer import convformer
from layers.util_layers import merge, conv_bn_act
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
import tensorflow.keras.backend as K

def build_model(img_size = 256, num_classes = 1):
    backbone = caformer.CAFormerS18(input_shape=(256, 256, 3), pretrained="imagenet", num_classes = 0)

    layer_names = ['stack4_block3_mlp_Dense_1', 'stack3_block9_mlp_Dense_1', 'stack2_block3_mlp_Dense_1', 'stack1_block3_mlp_Dense_1']
    layers = [backbone.get_layer(x).output for x in layer_names]

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    x = layers[0]

    upscale_feature = decode(x, scale = 4, filters = x.shape[channel_axis])

    for i, layer in enumerate(layers[1:]):
        
        x = decode(x, scale = 2, filters = layer.shape[channel_axis])
        
        layer_fusion = convformer(layer, layer.shape[channel_axis])
        
        ## Doing multi-level concatenation
        if (i%2 == 1):
            upscale_feature = tf.keras.layers.Conv2D(layer.shape[channel_axis], (1, 1), activation = "relu", padding = "same")(upscale_feature)
            x = tf.keras.layers.Add()([x, upscale_feature])
            x = tf.keras.layers.Conv2D(x.shape[channel_axis], (1, 1), activation = "relu", padding = "same")(x)
        
        x = merge([x, layer_fusion], layer.shape[channel_axis])
        x = conv_bn_act(x, layer.shape[channel_axis], (1, 1))

        ## Upscale for next level feature
        if (i%2 == 1):
            upscale_feature = decode(x, scale = 8, filters = layer.shape[channel_axis])
        
    filters = x.shape[channel_axis] //2
    upscale_feature = conv_bn_act(upscale_feature, filters, 1)
    x = decode(x, filters, 4)
    x = tf.keras.layers.Add()([x, upscale_feature])
    x = conv_bn_act(x, filters, 1)
    x = Conv2D(num_classes, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = Model(backbone.input, x)

    return model
