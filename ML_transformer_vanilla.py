from random import shuffle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import os
import wandb
from wandb.keras import WandbCallback
from pathlib import Path

os.environ["WANDB_NOTEBOOK_NAME"] = "ML_Veterinary_DataLoader.ipynb" 

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


strategy  = tf.distribute.MirroredStrategy()

SEED = 1234
AUTOTUNE = tf.data.AUTOTUNE
num_classes = 3
input_shape = (100, 100, 3)

learning_rate = 0.0001
weight_decay = 0.0001
batch_size = 128
num_epochs = 300
image_size = 100  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]

CONFIG = dict (
    seed = SEED,
    train_val_split = 0.1,
    train_test_split = 0.1,
    model_name = 'Transformer_Small',
    img_size = image_size,
    num_classes = num_classes,
    num_epochs = num_epochs,
    batch_size = batch_size,
)

experiment_name = wandb.util.generate_id()
wandb.init(project = "Age_transformer",
group = "Transformer_vanilla_small",
name = "Big lr",
job_type = "BigDs_on_Small",
config = CONFIG
)



data_augmentation = keras.Sequential(
    [
        layers.Resizing(image_size, image_size),
        layers.Lambda(lambda x: x/255.0),
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(factor=0.3),

    ],
    name="data_augmentation",
)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    def get_config(self):
        config = super(Patches,self).get_config()
        config.update({
            'patch_size' : self.patch_size
        })
        return config
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.projection = layers.Dense(units=self.projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=self.projection_dim
        )
    def get_config(self):
        config = super(PatchEncoder,self).get_config()
        config.update({
            'patch_size' : self.num_patches,
            'projection_dim' : self.projection_dim
        })
        return config 

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifier(augment=True):
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    if augment == True:
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = Patches(patch_size)(augmented)
    else:
        patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes,activation = 'softmax')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
# path to record files - give your own path to the dataset you want to use
# cwd is a folder with raw images, recordPath with images written as TFRecord

cwd = Path('...')
recordPath =  Path('...')


keys = []
values = []
for folder in os.listdir(cwd):
    keys.append(folder)
for i in range(0,len(keys)):
    values.append(i)
classes = dict(zip(keys, values))

filenames = os.listdir(recordPath)
for i in  range(len(filenames)):
    filenames[i] = recordPath / filenames[i] ## extending filenames for readability for dataset

feature_description = {
    'image': tf.io.FixedLenFeature((), tf.string),
    'label': tf.io.FixedLenFeature((), tf.int64),
    'height': tf.io.FixedLenFeature((), tf.int64),
    'width': tf.io.FixedLenFeature((), tf.int64),
    'depth': tf.io.FixedLenFeature((), tf.int64)
}
def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

def _parse_image_function(example_proto):
    features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.parse_tensor(features['image'], tf.float32)
    image.set_shape([100*100*3])
    image = tf.reshape(image, [100, 100, 3])
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, num_classes)

    return image, label

def read_dataset(filename_list):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  
    dataset = tf.data.TFRecordDataset(filename_list)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(len(list(dataset)),reshuffle_each_iteration=False)
    return dataset

dataset_train = read_dataset(filenames)
dataset_full = dataset_train
ds_length = len(list(dataset_train))
val_len = round(0.2*ds_length)
test_len = round(val_len/2)
dataset_valid = dataset_train.take(val_len)
dataset_train = dataset_train.skip(val_len)
dataset_test = dataset_valid.take(test_len)
dataset_valid = dataset_valid.skip(test_len)
dataset_train = dataset_train.batch(batch_size, drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()
dataset_valid = dataset_valid.batch(batch_size, drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()
dataset_test = dataset_test.batch(batch_size, drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()
dataset_full = dataset_full.batch(batch_size, drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    restore_best_weights=True,
)


with strategy.scope():
    model = create_vit_classifier(augment = False)
    optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            tfa.metrics.CohenKappa(num_classes),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            tfa.metrics.F1Score(num_classes)
        ],
    )

model.summary()
history = model.fit(
    x=dataset_train,
    validation_data=dataset_valid,
    epochs=num_epochs,
    callbacks = [WandbCallback(),
    early_stop
    ]
)

model.save_weights('../My_weights.h5',save_format='h5')




print("Test dataset:")
model.evaluate(dataset_test)
print("Valid dataset:")
model.evaluate(dataset_valid)
print("Train dataset:")
model.evaluate(dataset_train)
print("Full dataset:")
model.evaluate(dataset_full)