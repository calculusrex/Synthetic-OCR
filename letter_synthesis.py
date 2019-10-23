import tensorflow as tf

import PIL
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance, ImageFilter
import os
import math
import numpy as np
import random
import time
import functools as ft
import cv2

from word_synthesis__backup import random_ocr_datapoint__word

from text import encode_character, decode_character
from image__backup import encode_image, decode_image
from bounding_box import quad_box__2__corner_coords_box, corner_coords_box__2__quad_box

#################################################################################################
## ENCODING, DECODING, VECTORIZATION AND NORMALIZATION

def encode_angle(degrees):
    return math.radians(degrees + 180) / (2*np.pi)

def decode_angle(value):
    return (math.degrees(value * (2*np.pi))) - 180

#################################################################################################

def letter_data_tuple__2__letter_data_dict(tpl):
    noisy_image, clean_image, character, angle = tpl
    feature_dict = {
        'noisy_image': noisy_image,
        'clean_image': clean_image,
        'character': character,
        'angle': angle
    }
    return feature_dict



def character_data_generator__dict(batch_size):
    while True:
        samples = []
        batch_count = 0
        while batch_count < batch_size:
            features = random_ocr_datapoint__word()
            letter_images_noisy = map(
                lambda bbox: encode_image(features['image'].copy().crop(corner_coords_box__2__quad_box(bbox)).resize((32, 32), resample=PIL.Image.BICUBIC)),
                features['bounding_boxes']
            )
            letter_images_clean = map(
                lambda bbox: encode_image(features['autoencoder_target_image'].copy().crop(corner_coords_box__2__quad_box(bbox)).resize((32, 32), resample=PIL.Image.BICUBIC)),
                features['bounding_boxes']
            )
            characters = map(
                lambda c: encode_character(c),
                features['string']
            )
            encoded_angle = encode_angle(features['angle'])
            angles = [encoded_angle for chars in features['string']]
            letter_data = zip(
                letter_images_noisy,
                letter_images_clean,
                characters,
                angles
            )
            letter_data = map(
                letter_data_tuple__2__letter_data_dict,
                letter_data
            )
            for letter_datapoint in letter_data:
                samples.append(letter_datapoint)
                batch_count += 1
        yield samples[0:batch_size]



def character_data_generator__list_tuple(batch_size):
    while True:
        noisy_images = []
        clean_images = []
        characters = []
        angles = []
        batch_count = 0
        while batch_count < batch_size:
            features = random_ocr_datapoint__word()
            letter_images_noisy = list(map(
                lambda bbox: features['image'].copy().crop(corner_coords_box__2__quad_box(bbox)).resize((32, 32), resample=PIL.Image.BICUBIC),
                features['bounding_boxes']
            ))
            letter_images_clean = list(map(
                lambda bbox: features['autoencoder_target_image'].copy().crop(corner_coords_box__2__quad_box(bbox)).resize((32, 32), resample=PIL.Image.BICUBIC),
                features['bounding_boxes']
            ))
            encoded_angle = encode_angle(features['angle'])
            for i in range(len(features['string'])):
                noisy_images.append(
                    encode_image(letter_images_noisy[i])
                )
                clean_images.append(
                    encode_image(letter_images_clean[i])
                )
                characters.append(
                    encode_character(features['string'][i])
                )
                angles.append(
                    encoded_angle
                )
                batch_count += 1

        yield (
            noisy_images[0:batch_size],
            clean_images[0:batch_size],
            characters[0:batch_size],
            angles[0:batch_size]
        )


if __name__ == '__main__':
    print('LETTER SYNTHESIS')

    dgen = character_data_generator__dict(32)
    char_data = next(dgen)

