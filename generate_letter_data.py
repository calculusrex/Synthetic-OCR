import numpy as np
import tensorflow as tf

import sys

from letter_synthesis import character_data_generator__dict


###########################################################################################


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


###########################################################################################


def write_record(filename, n):
    
    writer = tf.io.TFRecordWriter(filename)
    
    datagen = character_data_generator__dict(512)
    batch_number = n // 512

    for batch_n in range(batch_number):
        batch = next(datagen)

        for sample in batch:
            feature = {
                'noisy_image__raw': _bytes_feature(sample['noisy_image'].tostring()),
                'clean_image__raw': _bytes_feature(sample['clean_image'].tostring()),
                'character': _int64_feature(np.int64(sample['character'])),
                'angle': _float_feature(sample['angle'])
            }

            example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature
                )
            )
            writer.write(example.SerializeToString())


        print('written_datapoints: ', batch_n * 512)

    writer.close()



if __name__ == '__main__':
    print('GENERATE WORD DATA')

    filename = sys.argv[1]

    write_record(filename, 1000000)
