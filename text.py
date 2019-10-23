import PIL
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance
import os
import math
import numpy as np
import random
import time
import functools as ft

#################################################################################################
## FONTS

ascii_characters = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

ascii_alphanumeric = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

ascii_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

ascii_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

ascii_symbols = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~']


#################################################################################################
## ENCODING, DECODING, VECTORIZATION AND NORMALIZATION

def encode_character(char):
    return ascii_characters.index(char)

def decode_character(i):
    return ascii_characters[i]

#################################################################################################
## UTILITARY FUNCTIONS

stringContains = lambda subject_string, object_string: subject_string.find(object_string) > -1

#################################################################################################
## FONTS

ttf_path = '/usr/share/fonts/truetype/' # Font root folder
ttf_folder_names = ['abyssinica', 'padauk', 'ubuntu', 'dejavu', 'liberation2', 'Nakula'] # font folders
# ttf_folder_names = ['liberation2'] # font folders, I temporarily restricted them to a single one.

ttf_folder_paths = list(map(lambda name: ttf_path + name,
                            ttf_folder_names))

def max_char_size(font, char_set):
    w = ft.reduce(
        max,
        map(lambda char: font.getsize(char)[0],
            char_set)
    )
    h = ft.reduce(
        max,
        map(lambda char: font.getsize(char)[1],
            char_set)
    )
    return w, h

# Aquire fonts from disk
fonts = []
for path in ttf_folder_paths:
    font_names = os.listdir(path) # list font files in folder
    font_names = filter(lambda string: stringContains(string, ".ttf"),
                        font_names) # filter ttf files
    font_paths = map(lambda name: path + '/' + name,
                     font_names) # compose full filepaths for each font
    for fontpath in font_paths:
        font_dict = {}
        font_dict['font'] = ImageFont.truetype(font=fontpath, size=40) # i've been using 40 for size, but let's try smaller
        max_w, max_h = max_char_size(
            font_dict['font'],
            ascii_characters)
        font_dict['max_char_size'] = (max_w, max_h)
        fonts.append(
            font_dict
        ) # instantiate and append

len_fonts = len(fonts)

def random_font():
    return fonts[np.random.randint(len_fonts)]

def random_string(char_set, length):
    len_char_set = len(char_set)
    indexes = [np.random.randint(len_char_set) for i in range(length)]
    string = "".join(
        map(lambda i: char_set[i],
            indexes)
    )
    return string

# just a string
def random_word(max_len):
    return random_string(ascii_alphanumeric,
                         np.random.randint(1, max_len))

# multiline text
def random_text(max_line_char_count, n_of_lines):
    lines = []
    for l in range(n_of_lines):
        line = ''
        if np.random.random() < 0.1:
            line += '\t'
        line_char_count = 0
        while True:
            if max_line_char_count < 64:
                randword = random_word(max_line_char_count/2)
            else:
                randword = random_word(32)
            if len(line) + 1 + len(randword) > max_line_char_count:
                break
            line += randword + " "
        lines.append(line[:-1] + '\n')
    string = "".join(lines)[0:-1]
    return string

# preprocessing the ascii string for use in the image generator
# takng into account a tab character
def split_line(line):
    if line[0] == '\t':
        dalist = ['\t'] + line[1:].split(' ')
        return dalist
    else:
        return line.split(' ')

# takes a string and splits it by space and newline, into a list of lists of words
# : String -> [[String]]
def split_text(string):
    return list(map(
        split_line,
        string.split('\n')
    ))


if __name__ == '__main__':
    print("## TEXT")

