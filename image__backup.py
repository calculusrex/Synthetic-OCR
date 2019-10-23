import PIL
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance
import os
import math
import numpy as np
import random
import time
import functools as ft
import cv2

from bounding_box import vector_add, random_vector__pixels

#################################################################################################
## ENCODING, DECODING, VECTORIZATION AND NORMALIZATION

def normalize_image_array(image_array):
    return (image_array / 255).astype(np.float32)

def denormalize_image_array(image_array):
    return np.round(
        (image_array * 255)).astype(np.uint8)

def encode_image(image):
    return normalize_image_array(
        np.array(image)
    )

def decode_image(image_array):
    return Image.fromarray(
        denormalize_image_array(
            image_array
        )
    )

def encode_angle(degrees):
    return math.radians(degrees + 180) / (2*np.pi)

def decode_angle(value):
    return (math.degrees(value * (2*np.pi))) - 180

#################################################################################################
## BACKGROUND

def add_background(img, fill=(255, 255, 255, 255)):
    bg = PIL.Image.new('RGBA', img.size, color=fill)
    img = PIL.Image.alpha_composite(bg, img)
    return img

#################################################################################################
## ADD SOME NOISE

def noiseyRGBBitmap(bitmap_size, noise_saturation, noise_brightness, noise_contrast, alpha_value, noise_size):

    if noise_size > 1:
        size = (bitmap_size[0] // noise_size, bitmap_size[1] // noise_size)
    else:
        size = bitmap_size

    noise_array = np.random.rand(size[1], size[0], 3) # generate array of random values on 3 chanels
    noise_array = noise_array * noise_brightness
    noisey_image = Image.fromarray(noise_array, 'RGB') # create a noise image from a numpy array of random values

    # Adjust saturation
    converter = ImageEnhance.Color(noisey_image)
    noisey_image = converter.enhance(noise_saturation)

    # Adjust contrast
    converter = ImageEnhance.Contrast(noisey_image)
    noisey_image = converter.enhance(noise_contrast)

    # Adjust brightness
    converter = ImageEnhance.Brightness(noisey_image)
    noisey_image = converter.enhance(noise_brightness)

    noisey_image.putalpha(alpha_value)

    if noise_size > 1:
        noisey_image = noisey_image.resize(bitmap_size, resample=PIL.Image.BICUBIC)
    
    return noisey_image

def noisey_background(bitmap_size,
                      steps=16,
                      noise_saturation=0.2,
                      noise_brightness=1.3,
                      noise_contrast=0.12,
                      alpha_value=255):

    nse = noiseyRGBBitmap(bitmap_size=bitmap_size, noise_saturation=noise_saturation,
                          noise_brightness=noise_brightness, noise_contrast=noise_contrast,
                          alpha_value=alpha_value, noise_size=1)

    for i in range(1, steps):
        ns2 = noiseyRGBBitmap(bitmap_size=bitmap_size, noise_saturation=noise_saturation,
                              noise_brightness=noise_brightness, noise_contrast=noise_contrast,
                              alpha_value=alpha_value, noise_size=i)
        nse = PIL.Image.blend(nse, ns2, 1/i)
        ns2.close()

    # Adjust saturation
    converter = ImageEnhance.Color(nse)
    nse = converter.enhance(noise_saturation)

    # Adjust contrast
    converter = ImageEnhance.Contrast(nse)
    nse = converter.enhance(noise_contrast)

    # Adjust brightness
    converter = ImageEnhance.Brightness(nse)
    nse = converter.enhance(noise_brightness)
    
    return nse

#################################################################################################
## ERRODE AND DILATE THE TEXT

def point_distance(a, b):
    x_distance = a[0] - b[0]
    y_distance = a[1] - b[1]
    return np.sqrt(x_distance**2 + y_distance**2)

def circularMorphOpKernel(diameter):
    center = ((0 + (diameter-1)/2),(0 + (diameter-1)/2))
    kernel = np.ones((diameter, diameter), np.uint8)
    for i in range(diameter):
        for j in range(diameter):
            if point_distance((i, j), center) > diameter/2:
                kernel[i][j] = 0
    return kernel

def dilate(image, kernel_diameter):
    kernel = circularMorphOpKernel(kernel_diameter)
    r, g, b, alpha = image.split()
    image.close()
    image_alpha__arr = np.array(alpha) # the image alpha channel converted to opencv format (aka numpy array)
    image_alpha__arr = cv2.dilate(image_alpha__arr, kernel, iterations=1)
    alpha_image = PIL.Image.fromarray(image_alpha__arr)
    outimage = PIL.Image.new(mode='RGB', size=image.size, color=(0, 0, 0))
    outimage.putalpha(alpha_image)
    alpha_image.close()

    return outimage

def erode(image, kernel_diameter):
    kernel = circularMorphOpKernel(kernel_diameter)
    r, g, b, alpha = image.split()
    image.close()
    image_alpha__arr = np.array(alpha) # the image alpha channel converted to opencv format (aka numpy array)
    image_alpha__arr = cv2.erode(image_alpha__arr, kernel, iterations=1)
    alpha_image = PIL.Image.fromarray(image_alpha__arr)
    outimage = PIL.Image.new(mode='RGB', size=image.size, color=(0, 0, 0))
    outimage.putalpha(alpha_image)
    alpha_image.close()

    return outimage

# negative open_close__factor close, positive ones open
# negative dilate_erode__factor erodes, positive ones dilate
# Here, the open_close__factor and dilate_erode__factor act as kernel diameter too, The thing is that for the sake of making the variation linear, i'm taking 0 as a kernel size of 1, -1 as a kernel size of -2 and 1 as a kernel size of 2 and so on, offseting them inwards, as a kernel size of 0 and a kernel size of 1 would have the same effect and would skew the distribution if i use a random number for it.
def morphology_ops(image, open_close__factor, dilate_erode__factor):
    r, g, b, alpha = image.split()
    image.close()
    image_alpha__arr = np.array(alpha) # the image alpha channel converted to opencv format (aka numpy array)
    if open_close__factor > 0:
        open_close__kernel = circularMorphOpKernel(open_close__factor + 1)
        image_alpha__arr = cv2.morphologyEx(image_alpha__arr, cv2.MORPH_OPEN, open_close__kernel)

    if open_close__factor < 0:
        open_close__kernel = circularMorphOpKernel(-open_close__factor + 1)
        image_alpha__arr = cv2.morphologyEx(image_alpha__arr, cv2.MORPH_CLOSE, open_close__kernel)

    if dilate_erode__factor > 0:
        dilate_erode__kernel = circularMorphOpKernel(dilate_erode__factor + 1)
        image_alpha__arr = cv2.dilate(image_alpha__arr, dilate_erode__kernel, iterations=1)

    if dilate_erode__factor < 0:
        dilate_erode__kernel = circularMorphOpKernel(-dilate_erode__factor + 1)
        image_alpha__arr = cv2.erode(image_alpha__arr, dilate_erode__kernel, iterations=1)

    alpha_image = PIL.Image.fromarray(image_alpha__arr)
    outimage = PIL.Image.merge(mode='RGB', bands=[r, g, b])
    outimage.putalpha(alpha_image)
    alpha_image.close()

    return outimage

def adjust_textAlphaBitmap_alpha(image, alpha):
    r, g, b, a = image.split()
    a = a.point(lambda p: np.floor(p*alpha))
    image = PIL.Image.merge(mode="RGBA", bands=(r, g, b, a))
    return image

#################################################################################################
## ADJUST ALPHABITMAP MARGINS

def adjust_bitmap_n_bounding_box_margins(
        bitmap,
        autoencoder_target_bitmap,
        bounding_boxes,
        left_x, upper_y, right_x, lower_y,
        fill=(0, 0, 0, 0)):
    
    # Image
    initial_size = bitmap.size
    margins = [left_x,  upper_y, right_x,lower_y]
    expansion_border_width = max(margins)
    # expand image equally in all directions by the largest margin, preparing it for cropping
    bitmap = ImageOps.expand(
        bitmap,
        border=expansion_border_width, # expand it with a border with the width of the largest margin
        fill=fill) # the fill of the border is transparent, i've used transparent black throughout the script
    expanded_size = bitmap.size
    # I won't take the time to explain the arithmetic below
    crop_coordinate_box = (
        expansion_border_width - left_x, # upperLeftCorner_x
        expansion_border_width - upper_y, # upperLeftCorner_y
        expanded_size[0] - (expansion_border_width - right_x), # lowerRightCorner_x
        expanded_size[1] - (expansion_border_width - lower_y)) # lowerRightCorner_y
    # crop the image with it's own class method
    bitmap = bitmap.crop(box=crop_coordinate_box)

    # Autoencoder target Image processed by the same parameters as the main image
    autoencoder_target_bitmap = ImageOps.expand(
        autoencoder_target_bitmap,
        border=expansion_border_width, # expand it with a border with the width of the largest margin
        fill=fill) # the fill of the border is transparent, i've used transparent black throughout the script
    autoencoder_target_bitmap = autoencoder_target_bitmap.crop(box=crop_coordinate_box)

    # Bounding Boxes
    bounding_boxes = map(
        lambda bbox: (vector_add(bbox[0], (left_x, 0)), vector_add(bbox[1], (left_x, 0))),
        bounding_boxes
    )
    bounding_boxes = list(map(
        lambda bbox: (vector_add(bbox[0], (0, upper_y)), vector_add(bbox[1], (0, upper_y))),
        bounding_boxes
    ))
    
    return bitmap, autoencoder_target_bitmap, bounding_boxes

#################################################################################################
## SQUIGGLES

def randcol():
    return np.random.randint(255)

def randang():
    return np.random.randint(360)

def random_point_on_image(lower_right_corner_coordinates):
    return (np.random.randint(0, lower_right_corner_coordinates[0]),
            np.random.randint(0, lower_right_corner_coordinates[1]))

def add_line_noise(image, n,
                   fill=None,
                   max_width=5,
                   min_length=4, max_length=60):
    w, h = image.size
    noise = PIL.Image.new('RGBA', (w, h), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(noise)
    for i in range(n):
        origin_point = random_point_on_image((w, h))
        endpoint = vector_add(
            random_vector__pixels(min_length, max_length),
            origin_point
        )
        if fill == None:
            draw.line((origin_point, endpoint),
                      fill=(randcol(), randcol(), randcol(), randcol()),
                      width=np.random.randint(1, max_width))
        else:
            draw.line((origin_point, endpoint),
                      fill=fill,
                      width=np.random.randint(1, max_width))
    image = PIL.Image.alpha_composite(image, noise)
    return image

def add_arc_noise(image, n,
                   fill=None,
                   max_width=5,
                   min_length=4, max_length=60):
    w, h = image.size
    noise = PIL.Image.new('RGBA', (w, h), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(noise)
    for i in range(n):
        origin_point = random_point_on_image((w, h))
        endpoint = vector_add(
            random_vector__pixels(min_length, max_length),
            origin_point
        )
        bbox = (origin_point, endpoint)
        if fill == None:
            draw.arc(bbox,
                     start=randang(), end=randang(),
                     fill=(randcol(), randcol(), randcol(), randcol()))
        else:
            draw.arc(bbox,
                     start=randang(), end=randang(),
                     fill=fill)
    image = PIL.Image.alpha_composite(image, noise)
    return image


#################################################################################################
#################################################################################################

if __name__ == '__main__':
    print("## IMAGE")
