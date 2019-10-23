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

from text import ascii_alphanumeric, ascii_characters, ascii_digits, ascii_symbols, ascii_alphabet, random_font, max_char_size, random_string

from image__backup import add_background, adjust_bitmap_n_bounding_box_margins, morphology_ops, noisey_background, noiseyRGBBitmap, adjust_textAlphaBitmap_alpha, add_line_noise, randcol, add_arc_noise

from bounding_box import point_2_squareBoundingBox, rotate_point, vector_add, vector_subtract, random_vector__pixels

#################################################################################################
## TOOLS

def normalize_image(xarr): # in this case, just divide by 255
    xarr = xarr / 255
    return xarr.astype(np.float32)

def denormalize_image(xarr):
    xarr = xarr * 255
    return xarr.astype(np.uint8)

###################
# The normalization functions below are aimed at targets

# the angles go from -180 to +180 and i want to convert them to a 0 to 1 float scale.
def normalize_angle(degrees):
    return math.radians(degrees + 180) / (2*np.pi)

def denormalize_angle(value):
    return (math.degrees(value * (2*np.pi))) - 180


# def normalize_bounding_boxes(bounding_boxes, image):
#     w, h = image.size
#     max_reach = max(w, h)
#     normalized_bounding_boxes = np.array(bounding_boxes) / max_reach
#     return normalized_bounding_boxes.astype(np.float32)

def normalize_bounding_boxes(bounding_boxes, image):
    normalized_bounding_boxes = np.array(bounding_boxes)
    return normalized_bounding_boxes.astype(np.int32)

# # : (Tensor, Tensor) -> NpArray
# def denormalize_bounding_boxes(bounding_boxes, hwd):
#     h, w, d = hwd
#     max_reach = max(w, h)
#     denormalized_bounding_boxes = np.array(bounding_boxes * max_reach)
#     return denormalized_bounding_boxes
    

#################################################################################################

def random_alphanumerical_string(length):
    indexes = np.random.random_integers(0, len(ascii_alphabet), length)
    string = "".join(
        map(lambda x: ascii_alphabet[x],
            indexes)
    )
    return string

def string_2_word_sample(string,
                         max_rotation_angle=5,
                         margin__left_x=np.random.randint(-4, 30),
                         margin__upper_y=np.random.randint(-4, 30),
                         margin__right_x=np.random.randint(-4, 30),
                         margin__lower_y=np.random.randint(-4, 30),
                         open_close__factor=random.randint(-3, 1),
                         dilate_erode__factor=random.randint(-1, 1),
                         white_bg=False):
    fnt = random_font()
    max_char_w, max_char_h = fnt['max_char_size']
    font = fnt['font']
    img_w, img_h = 0, max_char_h # we start with full height and zero width, the width is incremented letter by letter

    # Aquiring position data (letter center points and initial bounding boxes) for the string at hand
    letter_data = []
    for letter in string:
        data_dict = {}
        random_offset = np.random.randint(-6, 10)
        img_w += random_offset
        letter_w, letter_h = font.getsize(letter)
        letter_bbox = ((img_w, 0), (img_w + letter_w, letter_h)) # this should include some randomization in the future
        letter_center = (img_w + letter_w/2, img_h/2)
        data_dict['char'] = letter
        data_dict['bounding_box'] = letter_bbox
        data_dict['center_point'] = letter_center
        letter_data.append(data_dict)
        img_w += letter_w

    # Creating a canvas to write the letters to
    img = Image.new(
        mode='RGBA',
        size=(img_w, img_h),
        color=(0, 0, 0, 0) # transparent bg
    )
    img_draw = ImageDraw.Draw(img)

    # writing the characters to canvas
    for char_data in letter_data:
        img_draw.text(
            char_data['bounding_box'][0],
            char_data['char'],
            font=font,
            fill=(0, 0, 0, 255) # black
        )

    rotation_angle = np.random.random() * max_rotation_angle * [-1, 1][np.random.randint(2)]
    og_image_centerpoint = (img_w/2, img_h/2)
    # rotate the image
    img = img.rotate(
        angle=rotation_angle,
        center=og_image_centerpoint,
        expand=True,
        resample=PIL.Image.BICUBIC # I looked at nearest, bilinear and bicubic and bicubic looks accurate and sharp, i like it best
    )
    img_w__new, img_h__new = img.size
    new_centerpoint = (img_w__new/2, img_h__new/2) # after rotation
    centerpoints_delta_vector = vector_subtract(new_centerpoint, og_image_centerpoint)

    # rotating the character centerpoints (positions)
    rotated_centerpoints = map(
        lambda char_data: rotate_point(char_data['center_point'], og_image_centerpoint, rotation_angle),
        letter_data
    )
    # aligning the character centerpoints (positions) to the rotated and expanded image
    rotated_centerpoints = map(
        lambda point: vector_add(point, centerpoints_delta_vector),
        rotated_centerpoints
    )
    # offsettng the character centerpoints (positions) randomly to break uniformity
    rotated_centerpoints = map(
        lambda point: vector_add(point, random_vector__pixels(0, 2)),
        rotated_centerpoints
    )

    # create new bounding boxes from the new letter centers
    bounding_boxes = list(map(
        lambda p: point_2_squareBoundingBox(p, (max_char_h/2) + np.random.randint(-2, 2)), # square radius is half the max char height plus some random padding (postive or negative)
        rotated_centerpoints
    ))

    # Morphology adjustment (open, close, dilate and erode)
    img = morphology_ops(
        img,
        open_close__factor=open_close__factor,
        dilate_erode__factor=dilate_erode__factor
    )
    
    # Margins Adjustment
    img, bounding_boxes = adjust_bitmap_n_bounding_box_margins(
        img, bounding_boxes,
        margin__left_x, margin__upper_y, margin__right_x, margin__lower_y,
        fill=(0, 0, 0, 0))

    img = add_line_noise(img, np.random.randint(20, 60), max_width=3, min_length=2, max_length=60)
    img = add_line_noise(img, np.random.randint(2, 20), max_width=6, min_length=2, max_length=500, fill=(0, 0, 0, randcol()))

    # img = add_arc_noise(img, np.random.randint(60, 120), max_width=3, min_length=2, max_length=60)
    # img = add_arc_noise(img, np.random.randint(2, 60), max_width=6, min_length=2, max_length=500, fill=(0, 0, 0, randcol()))

    if white_bg:
        img = add_background(img, fill=(255, 255, 255, 255))
    else:
        bg_noise = noisey_background(bitmap_size=img.size, steps=12,
                                     noise_saturation=random.uniform(0.05, 0.93),
                                     noise_brightness=random.uniform(0.7, 1.8),
                                     noise_contrast=random.uniform(0.01, 0.6))

        grain = noiseyRGBBitmap(bitmap_size=img.size,
                                noise_saturation=random.uniform(0.05, 0.93),
                                noise_brightness=random.uniform(0.7, 1.8),
                                noise_contrast=random.uniform(0.01, 0.6),
                                alpha_value=255, noise_size=1)

        img = PIL.Image.alpha_composite(bg_noise, img)
        img = PIL.Image.blend(img, grain, np.random.uniform(0.1, 0.8))
        
    image = img.convert('RGB') # remove the alpha channel

    
    
    # Blur
    image = image.filter(ImageFilter.GaussianBlur(radius=np.random.randint(3)))

    # Contrast
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(random.uniform(0.2, 1))

    # Brightness
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(random.uniform(0.2, 1))

    # Saturation
    saturation = ImageEnhance.Color(image)
    image = saturation.enhance(random.uniform(0, 1))

    
    angle = rotation_angle
    
    return image, bounding_boxes, angle



def string_2_word_sample_n_autoencoder_target(
        string,
        max_rotation_angle=5,
        margin__left_x=np.random.randint(-4, 30),
        margin__upper_y=np.random.randint(-4, 30),
        margin__right_x=np.random.randint(-4, 30),
        margin__lower_y=np.random.randint(-4, 30),
        open_close__factor=random.randint(-3, 1),
        dilate_erode__factor=random.randint(-1, 1),
        white_bg=False):
    
    fnt = random_font()
    max_char_w, max_char_h = fnt['max_char_size']
    font = fnt['font']
    img_w, img_h = 0, max_char_h # we start with full height and zero width, the width is incremented letter by letter

    # Aquiring position data (letter center points and initial bounding boxes) for the string at hand
    letter_data = []
    for letter in string:
        data_dict = {}
        random_offset = np.random.randint(-6, 10)
        img_w += random_offset
        letter_w, letter_h = font.getsize(letter)
        letter_bbox = ((img_w, 0), (img_w + letter_w, letter_h)) # this should include some randomization in the future
        letter_center = (img_w + letter_w/2, img_h/2)
        data_dict['char'] = letter
        data_dict['bounding_box'] = letter_bbox
        data_dict['center_point'] = letter_center
        letter_data.append(data_dict)
        img_w += letter_w

    # Creating a canvas to write the letters to
    img = Image.new(
        mode='RGBA',
        size=(img_w, img_h),
        color=(0, 0, 0, 0) # transparent bg
    )
    img_draw = ImageDraw.Draw(img)

    # writing the characters to canvas
    for char_data in letter_data:
        img_draw.text(
            char_data['bounding_box'][0],
            char_data['char'],
            font=font,
            fill=(0, 0, 0, 255) # black
        )

    rotation_angle = np.random.random() * max_rotation_angle * [-1, 1][np.random.randint(2)]
    og_image_centerpoint = (img_w/2, img_h/2)
    # rotate the image
    img = img.rotate(
        angle=rotation_angle,
        center=og_image_centerpoint,
        expand=True,
        resample=PIL.Image.BICUBIC # I looked at nearest, bilinear and bicubic and bicubic looks accurate and sharp, i like it best
    )

    # Autoencoder_target image
    autoencoder_target_image = img.copy()
    
    img_w__new, img_h__new = img.size
    new_centerpoint = (img_w__new/2, img_h__new/2) # after rotation
    centerpoints_delta_vector = vector_subtract(new_centerpoint, og_image_centerpoint)

    # rotating the character centerpoints (positions)
    rotated_centerpoints = map(
        lambda char_data: rotate_point(char_data['center_point'], og_image_centerpoint, rotation_angle),
        letter_data
    )
    # aligning the character centerpoints (positions) to the rotated and expanded image
    rotated_centerpoints = map(
        lambda point: vector_add(point, centerpoints_delta_vector),
        rotated_centerpoints
    )
    # offsettng the character centerpoints (positions) randomly to break uniformity
    rotated_centerpoints = map(
        lambda point: vector_add(point, random_vector__pixels(0, 2)),
        rotated_centerpoints
    )

    # create new bounding boxes from the new letter centers
    bounding_boxes = list(map(
        lambda p: point_2_squareBoundingBox(p, (max_char_h/2) + np.random.randint(-2, 2)), # square radius is half the max char height plus some random padding (postive or negative)
        rotated_centerpoints
    ))

    # Morphology adjustment (open, close, dilate and erode)
    img = morphology_ops(
        img,
        open_close__factor=open_close__factor,
        dilate_erode__factor=dilate_erode__factor
    )
    
    # Margins Adjustment, it takes the autoencoder_target_image too and applies the same operation as to the main image
    img, autoencoder_target_image, bounding_boxes = adjust_bitmap_n_bounding_box_margins(
        img, autoencoder_target_image,
        bounding_boxes,
        margin__left_x, margin__upper_y, margin__right_x, margin__lower_y,
        fill=(0, 0, 0, 0))

    img = add_line_noise(img, np.random.randint(20, 60), max_width=3, min_length=2, max_length=60)
    img = add_line_noise(img, np.random.randint(2, 20), max_width=6, min_length=2, max_length=500, fill=(0, 0, 0, randcol()))

    # img = add_arc_noise(img, np.random.randint(60, 120), max_width=3, min_length=2, max_length=60)
    # img = add_arc_noise(img, np.random.randint(2, 60), max_width=6, min_length=2, max_length=500, fill=(0, 0, 0, randcol()))

    if white_bg:
        img = add_background(img, fill=(255, 255, 255, 255))
    else:
        bg_noise = noisey_background(bitmap_size=img.size, steps=12,
                                     noise_saturation=random.uniform(0.05, 0.93),
                                     noise_brightness=random.uniform(0.7, 1.8),
                                     noise_contrast=random.uniform(0.01, 0.6))

        grain = noiseyRGBBitmap(bitmap_size=img.size,
                                noise_saturation=random.uniform(0.05, 0.93),
                                noise_brightness=random.uniform(0.7, 1.8),
                                noise_contrast=random.uniform(0.01, 0.6),
                                alpha_value=255, noise_size=1)

        img = PIL.Image.alpha_composite(bg_noise, img)
        img = PIL.Image.blend(img, grain, np.random.uniform(0.1, 0.8))
        
    image = img.convert('RGB') # remove the alpha channel

    # Add a white bg to the autoencoder_target_image
    autoencoder_target_image = add_background(autoencoder_target_image, fill=(255, 255, 255, 255))
    # converting the autoencoder_target_image to grayscale
    autoencoder_target_image = autoencoder_target_image.convert('L')
    
    # Blur
    image = image.filter(ImageFilter.GaussianBlur(radius=np.random.randint(3)))

    # Contrast
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(random.uniform(0.2, 1))

    # Brightness
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(random.uniform(0.2, 1))

    # Saturation
    saturation = ImageEnhance.Color(image)
    image = saturation.enhance(random.uniform(0, 1))

    
    angle = rotation_angle
    
    return image, autoencoder_target_image, bounding_boxes, angle




# this function returns a feature dict, ready to be plugged to tf.train.Example
def random_ocr_datapoint__word():
    string = random_string(ascii_alphanumeric, np.random.randint(1, 64))
    feature_dict = {}
    feature_dict['string'] = string
    image, autoencoder_target_image, bounding_boxes, angle = string_2_word_sample_n_autoencoder_target(string)
    feature_dict['image'] = image
    feature_dict['autoencoder_target_image'] = autoencoder_target_image
    feature_dict['bounding_boxes'] = bounding_boxes
    feature_dict['angle'] = angle
    return feature_dict

def random_ocr_datapoint__word____tensors():
    features = random_ocr_datapoint__word()
    ftrs = {}
    ftrs['image'] = tf.convert_to_tensor(
        normalize_image(
            np.array(
                features['image'])))
    ftrs['angle'] = normalize_angle(
        features['angle'])
    ftrs['bounding_boxes'] = tf.convert_to_tensor(
        normalize_bounding_boxes(
            features['bounding_boxes'],
            features['image']))
    return ftrs


if __name__ == '__main__':
    print("## WORD SYNTHESIS")

    string = random_string(ascii_alphanumeric, np.random.randint(8, 16))
    # string = 'BACK AND ABS Punishment'

    image, autoencoder_target_image, bounding_boxes, angle = string_2_word_sample_n_autoencoder_target(
        string, # random string
        max_rotation_angle=5) # maximum rotation angle (a random one will be computed)

    def show_sample():
        feature_dict = random_ocr_datapoint__word()

        image = feature_dict['image']
        autoencoder_target_image = feature_dict['autoencoder_target_image']
        bounding_boxes = feature_dict['bounding_boxes']
        angle = feature_dict['angle']
        string = feature_dict['string']

        # show bounding boxes in green:
        image_draw = ImageDraw.Draw(image)
        for bbox in bounding_boxes:
            image_draw.rectangle(
                bbox, fill=None, outline=(100, 0, 0)
            )

        # show bounding boxes in green:
        autoencoder_target_image_draw = ImageDraw.Draw(autoencoder_target_image)
        for bbox in bounding_boxes:
            autoencoder_target_image_draw.rectangle(
                bbox, fill=None, outline=(100)
            )

        image.show()
        autoencoder_target_image.show()
            
        print('angle: ', angle)
        print('string: ', string)

    show_sample()
