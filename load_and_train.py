import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import PIL

from word_synthesis__backup import normalize_image, denormalize_image, normalize_angle, denormalize_angle, normalize_bounding_boxes# , denormalize_bounding_boxes

from bounding_box import draw_bounding_boxes

import sys

#################################################################################################
## DATA AQUISITION

featdef = {'image__raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
           'bounding_boxes__raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
           'image_height': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
           'image_width': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
           'image_depth': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
           'bounding_box_count': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
           'angle': tf.io.FixedLenFeature(shape=[], dtype=tf.float32)}


def _parse_record(example_proto, clip=False):
    example = tf.io.parse_single_example(
        serialized=example_proto, features=featdef)
    image_height = example['image_height']
    image_width = example['image_width']
    image_depth = example['image_depth']
    bounding_box_count = example['bounding_box_count']
    angle = example['angle']
    image = tf.io.decode_raw(
        example['image__raw'], tf.float32
    )
    bounding_boxes = tf.io.decode_raw(
        example['bounding_boxes__raw'], tf.int32 # tf.float32
    )
    features = {
        'unshaped_image': image,
        'unshaped_bounding_boxes': bounding_boxes,
        'image_height': image_height,
        'image_width': image_width,
        'image_depth': image_depth,
        'bounding_box_count': bounding_box_count,
        'angle': angle
    }
    return features

###############################################################################################

def array_2_bounding_boxes(bbox_array):
    return list(map(
        lambda bbox: tuple(map(tuple, bbox)),
        bbox_array
    ))

def recover_image(parsed_feature_dict):
    h, w, d = parsed_feature_dict['image_height'], parsed_feature_dict['image_width'], parsed_feature_dict['image_depth']
    bbx_c = parsed_feature_dict['bounding_box_count']
    angle = denormalize_angle(
        np.float32(parsed_feature_dict['angle'])
    )
    image = Image.fromarray(
        denormalize_image(
            np.array(
                parsed_feature_dict['unshaped_image']
            ).reshape((h, w, d))
        )
    )
    bounding_boxes = array_2_bounding_boxes(
        np.array(
            parsed_feature_dict['unshaped_bounding_boxes']
        ).reshape((bbx_c, 2, 2)),
    )

    features = {
        'image': image,
        'bounding_boxes': bounding_boxes,
        'angle': angle
    }

    return features

def recover_sample(parsed_feature_dict):
    h, w, d = parsed_feature_dict['image_height'], parsed_feature_dict['image_width'], parsed_feature_dict['image_depth']
    bbx_c = parsed_feature_dict['bounding_box_count']
    angle = np.float32(parsed_feature_dict['angle']) # not normalized
    image = np.array(
        parsed_feature_dict['unshaped_image']
    ).reshape((h, w, d)) # not normalized
    bounding_boxes = array_2_bounding_boxes(
        np.array(
            parsed_feature_dict['unshaped_bounding_boxes']
        ).reshape((bbx_c, 2, 2)),
    )

    features = {
        'image': image,
        'bounding_boxes': bounding_boxes,
        'angle': angle
    }

    return features


def prettyShow(recovered_sample):
    image = draw_bounding_boxes(recovered_sample['image'], recovered_sample['bounding_boxes'])
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('ANGLE: ')
    print(recovered_sample['angle'])
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    image.show()


###############################################################################################


if __name__ == '__main__':
    print('LOAD N LEARN')

    import time

    filepath = sys.argv[1]
    
    ds = tf.data.TFRecordDataset(filepath).map(_parse_record)

    subsample = ds.take(800)

    samples = list(subsample)

    sample_datapoint = samples[np.random.randint(0, 200)]

    features = recover_image(sample_datapoint)
    
    recovered_images = []
    for dp in subsample:
        recovered_images.append(
            recover_image(dp))

    prettyShow(recovered_images[0])

