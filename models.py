import tensorflow as tf
import numpy as np
from PIL import Image
import PIL

from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model

# ==================================================================================================================

def params_dict__2__conv_layer__w__pooling_n_dropout(params_dict):
    conv = tf.keras.layers.Conv2D(
        filters=params_dict['filters'],
        kernel_size=params_dict['kernel_size'],
        data_format='channels_last',
        activation='relu')
    pooling = tf.keras.layers.MaxPooling2D(
        pool_size=params_dict['pool_size'])
    dropout = tf.keras.layers.Dropout(
        params_dict['dropout_val'])
    return [conv, pooling, dropout]

def conv2D_stack__w__pooling_n_dropout(input_layer, conv_layer_paramss):
    layerss = map(
        params_dict__2__conv_layer__w__pooling_n_dropout,
        conv_layer_paramss)
    out = input_layer
    for layers in layerss:
        for layer in layers:
            out = layer(out)
    return out

# ==================================================================================================================

# def params_dict__2__conv_layer__w__pooling_n_dropout__residual(params_dict):
#     conv = tf.keras.layers.Conv2D(
#         filters=params_dict['filters'],
#         kernel_size=params_dict['kernel_size'],
#         data_format='channels_last',
#         activation='relu')
#     pooling = tf.keras.layers.MaxPooling2D(
#         pool_size=params_dict['pool_size'])
#     dropout = tf.keras.layers.Dropout(
#         params_dict['dropout_val'])
#     return [conv, pooling, dropout]

# def conv2D_stack__w__pooling_n_dropout__residual(input_layer, conv_layer_paramss):
#     layerss = map(
#         params_dict__2__conv_layer__w__pooling_n_dropout,
#         conv_layer_paramss)
#     out = input_layer
#     for layers in layerss:
#         for layer in layers:
#             out = layer(out)
#     return out

# ======================================================================================================

# img_input = tf.keras.layers.Input(shape=(None, None, 3)) # variable size images, RGB, not RGBA

# ------------------------------------------------------------------------------------------------------

# img_input = tf.keras.layers.Input(shape=(None, None, 3)) # variable size images, RGB, not RGBA

# conv = conv2D_stack__w__pooling_n_dropout__residual(
#     img_input,
#     [
#         {'filters': 32, 'kernel_size': 3, 'pool_size': (3, 3), 'dropout_val': 0.02},
#         {'filters': 64, 'kernel_size': 3, 'pool_size': (3, 3), 'dropout_val': 0.02},
#         {'filters': 128, 'kernel_size': 3, 'pool_size': (9, 9), 'dropout_val': 0.02},
#         {'filters': 256, 'kernel_size': 3, 'pool_size': (21, 21), 'dropout_val': 0.02},
#     ]
# )

# global_pooling = tf.keras.layers.GlobalAveragePooling2D()(conv)

# dense_1 = tf.keras.layers.Dense(64, activation='relu')(global_pooling)

# dropout_1 = tf.keras.layers.Dropout(0.25)(dense_1)

# dense_2 = tf.keras.layers.Dense(64, activation='relu')(dropout_1)

# encoder = tf.keras.models.Model(
#     inputs=img_input,
#     outputs=dense_2
# )


# ------------------------------------------------------------------------------------------------------
# simplest possible autoencoder

encoding_dim = 16

spa_input_img = Input(shape=(32, 32, 3))

spa_encoded = Dense(
    encoding_dim,
    input_shape=(32, 32, 3),
    activation='relu'
)(spa_input_img)

spa_decoded = Dense(1024, activation='sigmoid')(spa_encoded)

spa_autoencoder = Model(spa_input_img, spa_decoded)

spa_encoder = Model(spa_input_img, spa_encoded)

spa_encoded_input = Input(shape=(encoding_dim,))
spa_decoder_layer = spa_autoencoder.layers[-1]
spa_decoder = Model(spa_encoded_input, spa_decoder_layer(spa_encoded_input))


# ------------------------------------------------------------------------------------------------------



# conv_residual = conv2D_stack__w__pooling_n_dropout__residual(
#     img_input,
#     [
#         {'filters': 32, 'kernel_size': 3, 'pool_size': (3, 3), 'dropout_val': 0.02, 'has_residual_connection': False},
#         {'filters': 32, 'kernel_size': 3, 'pool_size': (3, 3), 'dropout_val': 0.02, 'has_residual_connection': True},
#         {'filters': 32, 'kernel_size': 3, 'pool_size': (3, 3), 'dropout_val': 0.02, 'has_residual_connection': False},
#         {'filters': 32, 'kernel_size': 3, 'pool_size': (3, 3), 'dropout_val': 0.02, 'has_residual_connection': True},
#         {'filters': 32, 'kernel_size': 3, 'pool_size': (3, 3), 'dropout_val': 0.02, 'has_residual_connection': False},
#         {'filters': 32, 'kernel_size': 3, 'pool_size': (3, 3), 'dropout_val': 0.02, 'has_residual_connection': True},
#         {'filters': 32, 'kernel_size': 3, 'pool_size': (3, 3), 'dropout_val': 0.02, 'has_residual_connection': False},
#         {'filters': 64, 'kernel_size': 3, 'pool_size': (3, 3), 'dropout_val': 0.02, 'has_residual_connection': True},
#         {'filters': 128, 'kernel_size': 3, 'pool_size': (9, 9), 'dropout_val': 0.02, 'has_residual_connection': True},
#         {'filters': 256, 'kernel_size': 3, 'pool_size': (21, 21), 'dropout_val': 0.02, 'has_residual_connection': True},
#     ]
# )

# global_pooling = tf.keras.layers.GlobalAveragePooling2D()(conv)

# dense_1 = tf.keras.layers.Dense(64, activation='relu')(global_pooling)

# dropout_1 = tf.keras.layers.Dropout(0.25)(dense_1)

# dense_2 = tf.keras.layers.Dense(64, activation='relu')(dropout_1)

# encoder = tf.keras.models.Model(
#     inputs=img_input,
#     outputs=dense_2
# )





if __name__ == '__main__':
    print('MODELS')



