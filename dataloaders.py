""" Support functions for loading the datasets."""

import os
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def create_image_dataset(dirpath,
batch_size : int = 32,
img_w : int = 100, 
img_h : int = 100,
val_split : float = 0.2,
one_hot : bool = True,
shuffle : bool = True,
seed : int = None
):
    """Function to create train, test, valid and full datasets from folder containing
    labeled photos."""
    
    if one_hot:
        label_mode = 'categorical'
    else:
        label_mode = 'int'
    
    dataset_train = tf.keras.utils.image_dataset_from_directory(
        dirpath,
        labels='inferred',
        label_mode=label_mode,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_h, img_w),
        shuffle=shuffle,
        seed=seed,
        validation_split=val_split,
        subset='training',
        interpolation='bilinear',

    )
    dataset_valid = tf.keras.utils.image_dataset_from_directory(
        dirpath,
        labels='inferred',
        label_mode=label_mode,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_h, img_w),
        shuffle=False,
        seed=seed,
        validation_split=val_split,
        subset='validation',
        interpolation='bilinear',

    )

    test_len = round(len(list(dataset_valid))/2)
    dataset_test = dataset_valid.take(test_len)
    dataset_valid = dataset_valid.skip(test_len)
    
    dataset_train = dataset_train.prefetch(buffer_size=AUTOTUNE)
    dataset_valid = dataset_valid.prefetch(buffer_size=AUTOTUNE)
    dataset_test = dataset_test.prefetch(buffer_size=AUTOTUNE)
    
    return dataset_train, dataset_valid, dataset_test


def create_tfr_dataset(
    dirpath : str,
    batch_size : int = 32,
    img_w : int = 100, 
    img_h : int = 100,
    val_split : float = 0.2,
    shuffle : bool = True,
    one_hot : bool = True,
    num_classes : int = 3,
    custom_feature_description : dict = None

):
    """Function to create train, test, valid and full datasets from TFRecord dataset."""
    filenames = os.listdir(dirpath)
    for i in  range(len(filenames)):
        filenames[i] = os.path.join(dirpath,filenames[i])
    
    
    if custom_feature_description is None:
        feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64)
        }
        
        
    dataset_train = read_dataset(filenames)

    ds_length = len(list(dataset_train))
    val_len = round(val_split*ds_length)
    test_len = round(val_len/2)

    dataset_valid = dataset_train.take(val_len)
    dataset_train = dataset_train.skip(val_len)
    dataset_test = dataset_valid.take(test_len)
    dataset_valid = dataset_valid.skip(test_len)

    if shuffle:
        dataset_train = dataset_train.shuffle(ds_length-val_len,reshuffle_each_iteration=False)

    dataset_train = dataset_train.batch(batch_size,
    num_parallel_calls=AUTOTUNE,
    drop_remainder=False).prefetch(buffer_size=AUTOTUNE)

    dataset_valid = dataset_valid.batch(batch_size,
    num_parallel_calls=AUTOTUNE,
    drop_remainder=False).prefetch(buffer_size=AUTOTUNE)

    dataset_test = dataset_test.batch(batch_size,
    num_parallel_calls=AUTOTUNE,
    drop_remainder=False).prefetch(buffer_size=AUTOTUNE)

   
    def parse_image_function(example_proto):
        """Function parsing TFRecrod files into image and label"""
        features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.parse_tensor(features['image'], tf.float32)
        image.set_shape([img_w*img_h*3])
        image = tf.reshape(image, [img_w, img_h, 3])
        image = tf.image.resize(image,(img_w,img_h))
        label = tf.cast(features['label'], tf.int32)
        if one_hot:
            label = tf.one_hot(label, num_classes)
        return image, label

    def read_dataset(filename_list):
        """Function to read and decode TFRecord files into TRecordDataset object."""
        dataset = tf.data.TFRecordDataset(filename_list)
        dataset = dataset.map(parse_image_function, num_parallel_calls=AUTOTUNE)
        return dataset

    return dataset_train, dataset_valid, dataset_test