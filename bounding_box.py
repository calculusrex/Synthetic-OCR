import PIL
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance
import os
import math
import numpy as np
import random
import time
import functools as ft
import cv2


def draw_bounding_boxes(image, bounding_boxes, outline=(100, 0, 0)):
    image_draw = ImageDraw.Draw(image)
    for bbox in bounding_boxes:
        image_draw.rectangle(
            bbox, fill=None, outline=outline
        )
    return image


# def random_vector(min_radius, max_radius):
#     angle = np.random.random() * 2*math.pi
#     variability = max_radius - min_radius
#     offset = min_radius
#     radius = np.random.random() * variability + offset
#     x = radius * math.cos(angle)
#     y = radius * math.sin(angle)
#     return (x, y)



def random_vector__pixels(min_radius, max_radius):
    angle = np.random.random() * 2*math.pi
    variability = max_radius - min_radius
    offset = min_radius
    radius = np.random.random() * variability + offset
    x = np.ceil(radius * math.cos(angle))
    y = np.ceil(radius * math.sin(angle))
    return (x, y)

def random_vector__pixels__square(min_radius, max_radius):
    x = np.random.randint(min_radius, max_radius) * [-1, 1][np.random.randint(2)]
    y = np.random.randint(min_radius, max_radius) * [-1, 1][np.random.randint(2)]
    return (x, y)

def vector_add(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])

def vector_subtract(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1])

# : (x, y), (x, y), Float -> (x, y)
def rotate_point(point, origin, angle__degrees):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in degrees, it is being converted to radians here.

    I'm making the angle negative because experimentally, that's what rotates my image in the direction i want,
    The vertical axis is flipped in the PIL coordinate system.
    """
    angle = - math.radians(angle__degrees)
    
    ox, oy = origin
    px, py = point
    
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def point_2_squareBoundingBox(point, radius):
    upper_left_corner = (point[0]-radius, point[1]-radius)
    lower_right_corner = (point[0]+radius, point[1]+radius)
    return (upper_left_corner, lower_right_corner)


# This is a function which aims to replace the lines inside string_2_word_sample which create the dimensions of the image by iterating through the string and recording and letter sizes and bounding boxes.
# it takes a string and a font dictionary which has the font and the max character size of the font
# I don't know if it would need the word height too as parameter, we'll see
# (String, Fnt{}) -> {(,), {}}
def string_2_word_data(string, font):
    word_w, word_h = 0, font['max_char_size'][1]
    word_data = {}
    word_data['letter_data'] = []
    for letter in string:
        data_dict = {}
        random_offset = np.random.randint(-6, 10)
        word_w += random_offset
        letter_w, letter_h = font['font'].getsize(letter)
        letter_bbox = ((word_w, 0), (word_w + letter_w, letter_h)) # this should include some randomization in the future
        letter_center = (word_w + letter_w/2, word_h/2)
        data_dict['char'] = letter
        data_dict['bounding_box'] = letter_bbox
        data_dict['centerpoint'] = letter_center
        word_data['letter_data'].append(data_dict)
        word_w += letter_w
    word_data['bounding_box'] = ((0, 0), (word_w, word_h))
    word_data['centerpoint'] = (word_w/2, word_h/2)
    word_data['string'] = string
    return word_data

def bounding_box__2__w_n_h(bbox):
    return (
        bbox[1][0] - bbox[0][0],
        bbox[1][1] - bbox[0][1]
    )

def rotation_and_alignment_of_word_data_to_rotated_and_expanded_image(
        angle, word_data):

    new_word_data = {}
    new_word_data['string'] = word_data['string']
    
    og_image_centerpoint = (
        word_data['bounding_box'][1][0]/2,
        word_data['bounding_box'][1][1]/2
    )

    cosine = np.cos(math.radians(angle))
    sine = np.sin(math.radians(angle))
    radius = word_data['bounding_box'][1][0]/2

    char_height = max(map(
        lambda letter_data: bounding_box__2__w_n_h(letter_data['bounding_box'])[1],
        word_data['letter_data']
    ))

    new_word_w = (radius * cosine) * 2
    new_word_h = ((radius * sine) * 2) + char_height

    new_word_data['centerpoint'] = (new_word_w/2, new_word_h/2)
    
    new_word_data['bounding_box'] = (
        (0, 0),
        (new_word_w, new_word_h)
    )

    centerpoints_delta_vector = vector_subtract(
        new_word_data['centerpoint'], word_data['centerpoint'])

    # rotating the character centerpoints (positions)
    rotated_centerpoints = map(
        lambda char_data: rotate_point(char_data['centerpoint'], word_data['centerpoint'], angle),
        word_data['letter_data']
    )
    # aligning the character centerpoints (positions) to the rotated and expanded image
    rotated_centerpoints = map(
        lambda point: vector_add(point, centerpoints_delta_vector),
        rotated_centerpoints
    )
    # offsettng the character centerpoints (positions) randomly to break uniformity
    rotated_centerpoints = list(map(
        lambda point: vector_add(point, random_vector__pixels(0, 2)),
        rotated_centerpoints
    ))

    # create new bounding boxes from the new letter centers
    letter_bounding_boxes = list(map(
        lambda p: point_2_squareBoundingBox(p, (char_height/2) + np.random.randint(-2, 2)), # square radius is half the max char height plus some random padding (postive or negative)
        rotated_centerpoints
    ))

    new_word_data['letter_data'] = []
    for i in range(len(word_data['string'])):
        letter_data = {}
        letter_data['char'] = word_data['string'][i]
        letter_data['bounding_box'] = letter_bounding_boxes[i]
        letter_data['center_point'] = rotated_centerpoints[i]
        new_word_data['letter_data'].append(letter_data)

    return new_word_data

def word_data_2_bounding_boxes(word_data):
    return list(map(
        lambda letter_data: letter_data['bounding_box'],
        word_data['letter_data']
    ))

def integrate_bounding_boxes_into_word_data(bounding_boxes, word_data):
    for i in range(len(word_data['string'])):
        word_data['letter_data'][i]['bounding_box'] = bounding_boxes[i]
    return word_data

#################################################################################################

def set_bounding_box_values_to_int(bounding_box):
    return (
        (
            np.int32(np.round(bounding_box[0][0])),
            np.int32(np.round(bounding_box[0][1]))
        ),
        (
            np.int32(np.round(bounding_box[1][0])),
            np.int32(np.round(bounding_box[1][1]))
        )
    )

def corner_coords_box__2__quad_box(corner_box):
    return (
        corner_box[0][0],
        corner_box[0][1],
        corner_box[1][0],
        corner_box[1][1]
    )

def quad_box__2__corner_coords_box(quad_box):
    return (
        (
            quad_box[0],
            quad_box[1]
        ),
        (
            quad_box[2],
            quad_box[3]
        )
    )

#################################################################################################
#################################################################################################

if __name__ == '__main__':
    print("## BOUNDING BOX")
