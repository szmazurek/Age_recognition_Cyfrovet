"""A function to create TFRecord datasets from source folder containing photos.
The directory should contain subdirectories named with class names. The photos should
be contained in subdirectories and have .jpg extension. For more info about
TFRecords please visit https://www.tensorflow.org/tutorials/load_data/tfrecord"""
from pathlib import Path
import os
import tensorflow as tf
import numpy as np

# path to source photos
DIR_PATH = "/net/scratch/people/plgmazurekagh/dogs_datasets/FaceDetector_cropped/Huge_dataset"
# path to folder to save tfrecords
RECORD_PATH = "/net/scratch/people/plgmazurekagh/TFRecords_ds/Huge_ds_unprocessed"
if not Path.exists(RECORD_PATH):
    os.mkdir(RECORD_PATH)
# number of pictures included in single record
BEST_NUM = 3000
# the number of images in current TFRecord file
NUM = 0
# the index of the tfrecord file
RECORD_FILE_NUM = 0
# dictionary with labels encoded as numbers
keys = []
values = []
for folder in os.listdir(DIR_PATH):
    keys.append(folder)
for i in range(0,len(keys)):
    values.append(i)
classes = dict(zip(keys, values))
writer = tf.io.TFRecordWriter(RECORD_PATH + str(RECORD_FILE_NUM))

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
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
    }
#  Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

for name, label in classes.items():
    class_path = os.path.join(DIR_PATH, name)
    for img_name in os.listdir(class_path):
        num += 1
        if num > BEST_NUM:
            num = 1
            RECORD_FILE_NUM += 1
            writer = tf.io.TFRecordWriter(RECORD_PATH + str(RECORD_FILE_NUM)+str(name))
            print("Creating the %.3d tfrecord file" % RECORD_FILE_NUM)
        img_path = os.path.join(class_path, img_name)
        img = tf.keras.preprocessing.image.load_img(img_path) ## load image
        img_array = tf.keras.preprocessing.image.img_to_array(img) ## convert to array
        img_array = img_array.astype(np.float32) ## convert values to float
        img_raw = tf.io.serialize_tensor(img_array) ## serialize tensor into string of bytes
        img_shape = img_array.shape ## recover shape of image
        example = serialize_example(img_raw, label, img_shape)
        ## serailize example and prepare it for writing to tfrecord
        writer.write(example)
writer.close()
print("Creating dataset finished.")
