import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import PIL

from word_synthesis__backup import normalize_image, denormalize_image, normalize_angle, denormalize_angle, normalize_bounding_boxes# , denormalize_bounding_boxes

import sys

from text import encode_character, decode_character
from image__backup import encode_image, decode_image
from letter_synthesis import encode_angle, decode_angle

from models import spa_autoencoder, spa_encoder, spa_decoder

#################################################################################################
## DATA AQUISITION

featdef = {
    'noisy_image__raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'clean_image__raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'character': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    'angle': tf.io.FixedLenFeature(shape=[], dtype=tf.float32) # i gotta check, maybe it's float 64
}


# def _parse_record(example_proto, clip=False):
#     example = tf.io.parse_single_example(
#         serialized=example_proto, features=featdef)
#     angle = example['angle']
#     character = example['character']
#     noisy_image = tf.reshape(
#         tf.io.decode_raw(
#             example['noisy_image__raw'], tf.float32
#         ),
#         (32, 32, 3)
#     )
#     clean_image = tf.reshape(
#         tf.io.decode_raw(
#             example['clean_image__raw'], tf.float32
#         ),
#         (32, 32)
#     )
#     features = {
#         'noisy_image': noisy_image,
#         'clean_image': clean_image,
#         'character': character,
#         'angle': angle
#     }
#     return features

def _parse_record(example_proto, clip=False):
    example = tf.io.parse_single_example(
        serialized=example_proto, features=featdef)
    angle = example['angle']
    character = example['character']
    noisy_image = tf.reshape(
        tf.io.decode_raw(
            example['noisy_image__raw'], tf.float32
        ),
        (32, 32, 3)
    )
    clean_image = tf.io.decode_raw(
        example['clean_image__raw'], tf.float32
    )
    features = {
        'noisy_image': noisy_image,
        'clean_image': clean_image,
        'character': character,
        'angle': angle
    }
    return noisy_image, clean_image


###############################################################################################

def recover_sample(parsed_record):
    noisy_image = decode_image(parsed_record['noisy_image'])
    clean_image = decode_image(parsed_record['clean_image'])
    character = decode_character(parsed_record['character'])
    angle = decode_angle(parsed_record['angle'])

    noisy_image.show()
    clean_image.show()
    print(character)
    print(angle)
    
    return {
        'noisy_image': noisy_image,
        'clean_image': clean_image,
        'character': character,
        'angle': angle
    }

###############################################################################################

filepath = sys.argv[1]

ds = tf.data.TFRecordDataset(filepath).map(_parse_record)
ds = ds.batch(512) # .shuffle(1024)

###############################################################################################

model = spa_autoencoder
model_name = 'spa_autoencoder'

model.compile(optimizer='adadelta', loss='binary_crossentropy')

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./checkpoints/' + model_name + '.h5', verbose=1), # conv2D__01.h5', verbose=1)
    tf.keras.callbacks.TensorBoard(log_dir='./tensorboard/' + model_name,
                                   write_images=True)
]

history = model.fit(ds, epochs=488, steps_per_epoch=2048)

###############################################################################################

if __name__ == '__main__':
    print('LOAD N LEARN')

    import time

    # some_samples = list(ds.take(100))

    # for i in range(10):
    #     recover_sample(some_samples[i])
