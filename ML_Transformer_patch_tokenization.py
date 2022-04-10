import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers
import wandb
import os
from random import shuffle
from wandb.keras import WandbCallback
from pathlib import Path
os.environ["WANDB_NOTEBOOK_NAME"] = "ML_Veterinary_DataLoader.ipynb" 

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



strategy  = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
AUTOTUNE = tf.data.AUTOTUNE
# Setting seed for reproducibiltiy
SEED = 42
keras.utils.set_random_seed(SEED)
NUM_CLASSES = 3
INPUT_SHAPE = (100, 100, 3)
BATCH_SIZE = 64
IMG_LEN = 100
PATCH_SIZE = 6
NUM_PATCHES = (IMG_LEN // PATCH_SIZE) ** 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 150

# ARCHITECTURE
LAYER_NORM_EPS = 1e-6
TRANSFORMER_LAYERS = 8
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]
MLP_HEAD_UNITS = [2048, 1024]

CONFIG = dict (
    seed = SEED,
    train_val_split = 0.1,
    train_test_split = 0.1,
    model_name = 'Transformer_Small',
    img_size = IMG_LEN,
    num_classes = 4,
    num_epochs = EPOCHS,
    batch_size = BATCH_SIZE,
)

experiment_name = wandb.util.generate_id()
wandb.init(project = "Age_transformer",
group = "Transformer_small",
name = experiment_name,
job_type = "Small_dataset",
config = CONFIG
)

data_augmentation = keras.Sequential(
    [
        layers.Resizing(IMG_LEN, IMG_LEN),
        layers.Lambda(lambda x: x/255.0),
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(factor=0.3),
    ],
    name="data_augmentation",
)

class ShiftedPatchTokenization(layers.Layer):
    def __init__(
        self,
        image_size=IMG_LEN,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        projection_dim=PROJECTION_DIM,
        vanilla=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = vanilla  # Flag to swtich to vanilla patch extractor
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.flatten_patches = layers.Reshape((num_patches, -1))
        self.projection = layers.Dense(units=projection_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
    def get_config(self):
        config = super(ShiftedPatchTokenization,self).get_config()
        config.update({
            'vanilla' : self.vanilla,
            'image_size' : self.image_size,
            'patch_size' : self.patch_size,
            'num_patches' : NUM_PATCHES,
            'projection_dim' : PROJECTION_DIM
        })
        return config
    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)
        return (tokens, patches)

class PatchEncoder(layers.Layer):
    def __init__(
        self, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches


class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The trainable temperature term. The initial value is
        # the square root of the key dimension.
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# Build the diagonal attention mask
diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

def create_vit_classifier(vanilla=False):
    inputs = layers.Input(shape=INPUT_SHAPE)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    (tokens, _) = ShiftedPatchTokenization(vanilla=vanilla)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder()(tokens)

    # Create multiple layers of the Transformer block.
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        if not vanilla:
            attention_output = MultiHeadAttentionLSA(
                num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1, attention_mask=diag_attn_mask)
        else:
            attention_output = layers.MultiHeadAttention(
                num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    logits = layers.Dense(512)(features)
    logits = layers.Dropout(0.2)(logits)
    logits = layers.Dense(256)(logits)
    logits = layers.Dropout(0.2)(logits)
    # Classify outputs.
    logits = layers.Dense(NUM_CLASSES,activation='softmax',)(logits)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )
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
    filenames[i] = recordPath + filenames[i] ## extending filenames for readability for dataset

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
    label = tf.one_hot(label, NUM_CLASSES)

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
dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()
dataset_valid = dataset_valid.batch(BATCH_SIZE, drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()
dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()
dataset_full = dataset_full.batch(BATCH_SIZE, drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()



early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=13,
    mode="min",
    restore_best_weights=True,
)


with strategy.scope():
    model = create_vit_classifier(vanilla=False)
    optimizer = tfa.optimizers.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            tfa.metrics.CohenKappa(NUM_CLASSES),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            tfa.metrics.F1Score(NUM_CLASSES)
        ],
    )
model.summary()
total_steps = int((len(list(dataset_train)) / BATCH_SIZE) * EPOCHS)
warmup_epoch_percentage = 0.10
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=warmup_steps,
)



history = model.fit(
    x=dataset_train,
    validation_data = dataset_valid,
    epochs=150,
    callbacks = [WandbCallback(),
    early_stop]
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

