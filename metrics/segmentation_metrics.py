import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
def dice_coeff(y_true, y_pred):
    
    _epsilon = 10 ** -7
    intersections = tf.reduce_sum(y_true * y_pred)
    unions = tf.reduce_sum(y_true + y_pred)
    dice_scores = (2.0 * intersections + _epsilon) / (unions + _epsilon)
    
    return dice_scores

def dice_loss(y_true, y_pred):
    
    loss = 1 - dice_coeff(y_true, y_pred)
    
    return loss

def total_loss(y_true, y_pred):
    return 0.5*binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def IoU(y_true, y_pred, eps=1e-6):
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    
    return K.mean( (intersection + eps) / (union + eps), axis=0)

def zero_IoU(y_true, y_pred):
    
    return IoU(1-y_true, 1-y_pred)

def bce_dice_loss(y_true, y_pred):
    
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    
    y_true_pos = tf.reshape(y_true,[-1])
    y_pred_pos = tf.reshape(y_pred,[-1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    
    tv = tversky(y_true, y_pred)
    
    return K.pow((1 - tv), gamma)
