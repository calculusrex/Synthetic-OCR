import numpy as np
import tensorflow as tf

from word_synthesis__backup import random_ocr_datapoint__word, random_ocr_datapoint__word____tensors

import sys

from word_synthesis__backup import normalize_image, denormalize_image, normalize_angle, denormalize_angle, normalize_bounding_boxes #, denormalize_bounding_boxes


#################################################################################################
## DATA SERIALIZING

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# def feature():
#     gnr = random_ocr_datapoint__word()
#     feature = {}

#     image = tf.convert_to_tensor(normalize(
#         np.array(gnr['image'])
#     ))
#     image_shape = image.shape

#     bounding_boxes = tf.convert_to_tensor(
#         np.array(gnr['bounding_boxes'])
#     )
#     bounding_boxes_shape = bounding_boxes.shape

#     feature['image'] = _bytes_feature(
#         tf.io.serialize_tensor(image)
#     )
#     feature['image_shape'] = _bytes_feature(
#         np.array(image_shape).tostring()
#     )
#     feature['target_string'] = _bytes_feature(
#         gnr['string'].encode('utf-8')
#     )
#     feature['bounding_boxes'] = _bytes_feature(
#         tf.io.serialize_tensor(bounding_boxes)
#     )
#     feature['bounding_boxes_shape'] = _bytes_feature(
#         np.array(bounding_boxes_shape).tostring()
#     )
#     feature['angle'] = _float_feature(
#         np.float32(gnr['angle']))
#     return feature

def feature():
    gnr = random_ocr_datapoint__word()
    w, h = gnr['image'].size
    image_height = h
    image_width = w
    image_depth = 3
    bounding_box_count = len(gnr['bounding_boxes'])
    string = gnr['string']
    
    image__raw = _bytes_feature(
        normalize_image(np.array(
            gnr['image']
        )).tostring()
    )
    autoencoder_target_image = _bytes_feature(
        normalize_image(np.array(
            gnr['autoencoder_target_image']
        )).tostring()
    )
    bounding_boxes__raw = _bytes_feature(
        normalize_bounding_boxes(
            gnr['bounding_boxes'],
            gnr['image']
        ).tostring()
    )
    angle = normalize_angle(
        np.float32(
            gnr['angle'])
    )
    feature = {
        'image__raw': image__raw,
        'autoencoder_target_image': autoencoder_target_image,
        'bounding_boxes__raw': bounding_boxes__raw,
        'image_height': _int64_feature(image_height),
        'image_width': _int64_feature(image_width),
        'image_depth': _int64_feature(image_depth),
        'bounding_box_count': _int64_feature(bounding_box_count),
        'angle': _float_feature(angle)
#        'string': _bytes_feature(string)
    }
    return feature


def write_record(filename, datapoint_count):
    writer = tf.io.TFRecordWriter(filename)
    for i in range(datapoint_count):
        if i%100 == 0:
            print('Written datapoints: ', i)
        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature()
            )
        )
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    print('GENERATE WORD DATA\n')

    filepath = sys.argv[1]

    write_record(filepath, 100000)



    # gnr = random_ocr_datapoint__word____tensors()

