"""A function to create TFRecord datasets from source folder containing photos.
The directory should contain subdirectories named with class names. The photos should
be contained in subdirectories and have .jpg extension. For more info about
TFRecords please visit https://www.tensorflow.org/tutorials/load_data/tfrecord"""

import os
import argparse
import tensorflow as tf
import numpy as np


## functions creating features of certain types required to write a tfrecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, label, image_shape):
    """Function to serialize image into TFRecord example using defined
    feature description"""
    feature = {
        "image": _bytes_feature(image),
        "label": _int64_feature(label),
        "height": _int64_feature(image_shape[0]),
        "width": _int64_feature(image_shape[1]),
        "depth": _int64_feature(image_shape[2]),
    }
    #  Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script converting image dataset into TFRecord one."
    )
    parser.add_argument(
        "dir_path",
        type=str,
        help="Path to file containing image dataset. Images should be in subfolders accordingly to the classes.",
    )
    parser.add_argument(
        "tfr_path", type=str, help="Path to folder in which to save TFRecord files."
    )
    parser.add_argument(
        "imgs_per_tfrecord",
        type=int,
        help="Number of images to write in a single TFRecord file.",
    )

    args = parser.parse_args()
    # path to source photos
    DIR_PATH = args.dir_path
    # path to folder to save tfrecords
    RECORD_PATH = args.tfr_path
    if not os.path.exists(RECORD_PATH):
        os.mkdir(RECORD_PATH)
    # number of pictures included in single record
    IMGS_PER_TFRECORD = args.imgs_per_tfrecord
    # the number of images in current TFRecord file
    NUM = 0
    # the index of the tfrecord file
    record_file_num = 0
    # dictionary with labels encoded as numbers
    keys = []
    values = []
    for folder in os.listdir(DIR_PATH):
        keys.append(folder)
    for i in range(0, len(keys)):
        values.append(i)
    classes = dict(zip(keys, values))
    writer = tf.io.TFRecordWriter(RECORD_PATH + str(record_file_num))

    for name, label in classes.items():
        class_path = os.path.join(DIR_PATH, name)
        for img_name in os.listdir(class_path):
            num += 1
            if num > IMGS_PER_TFRECORD:
                num = 1
                record_file_num += 1
                writer = tf.io.TFRecordWriter(
                    RECORD_PATH + str(record_file_num) + str(name)
                )
                print(f"Creating the {record_file_num} tfrecord file")
            img_path = os.path.join(class_path, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path)  ## load image
            img_array = tf.keras.preprocessing.image.img_to_array(
                img
            )  ## convert to array
            img_array = img_array.astype(np.float32)  ## convert values to float
            img_raw = tf.io.serialize_tensor(
                img_array
            )  ## serialize tensor into string of bytes
            example = serialize_example(img_raw, label, img_array.shape)
            ## serailize example and prepare it for writing to tfrecord
            writer.write(example)
    writer.close()
    print("Creating dataset finished.")
