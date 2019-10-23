import PIL
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance, ImageFilter
import os
import math
import numpy as np
import random
import time
import functools as ft
import cv2

####################################################################################################

def letter_bounding_boxes__2__word_bounding_box(letter_bounding_boxes):
    return (
        (
            min(map(lambda bbox: bbox[0][0], letter_bounding_boxes)),
            min(map(lambda bbox: bbox[0][1], letter_bounding_boxes))
        ),
        (
            max(map(lambda bbox: bbox[0][0], letter_bounding_boxes)),
            max(map(lambda bbox: bbox[0][1], letter_bounding_boxes))
        )
    )

def centerpoint_n_size__2__bounding_box(centerpoint, letter_size):
    
    

####################################################################################################

def string_2_word_sample(string):
    data = string_2_word_data(string)
    labels['letter_bounding_boxes'] = list(map(
        lambda centerpoint: centerpoint_n_size__2__bounding_box(centerpoint, data['max_letter_size']),
        data['letter_centerpoints']
    ))
    labels['word_bounding_box'] = letter_bounding_boxes__2__word_bounding_box(data['letter_bounding_boxes'])
    labels['angle'] = data['angle']
    return (
        image,
        labels
    )


if __name__ == '__main__':
    print("word_synthesis_2")
