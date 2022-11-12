"""A script to train the network for dog breed classification task using
Stanford Dogs Dataset. Inegration with Weights and biases provided."""
import wandb
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from models import create_cnn_network
from wandb.keras import WandbCallback

SEED = 1234
tf.random.set_seed(SEED)
STATELESS_SEED = (SEED, SEED)

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

strategy = tf.distribute.MirroredStrategy()

AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 120
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 1024
NUM_EPOCHS = 50
VAL_SIZE = 0.2
FINETUNING = False

CONFIG = dict(
    seed=SEED,
    img_size=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
)

wandb.init(
    project="my_proj",
    group="my_group",
    name="my_name",
    job_type="my_job",
    config=CONFIG,
)


# load and preprocess the data

training_dataset, test_dataset = tfds.load(
    "stanford_dogs", as_supervised=True, split=["train", "test"], shuffle_files=True
)


training_dataset = (
    training_dataset.map(lambda x, y: (tf.image.resize(x,[IMAGE_WIDTH,IMAGE_HEIGHT]), y))
    .batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE, drop_remainder=False)
    .prefetch(buffer_size=AUTOTUNE)
)


valid_dataset = (
    test_dataset.map(lambda x, y: (tf.image.resize(x,[IMAGE_WIDTH,IMAGE_HEIGHT]), y))
    .batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE, drop_remainder=False)
    .prefetch(buffer_size=AUTOTUNE)
)


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
    restore_best_weights=True,
)

with strategy.scope():

    optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model_breed = create_cnn_network(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model_breed.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )



## train the model
history = model_breed.fit(
    training_dataset,
    validation_data=valid_dataset,
    epochs=NUM_EPOCHS,
    callbacks=[WandbCallback(), early_stop],
)
if FINETUNING:
    model_breed.trainable = True
    
    early_stopping_tuning = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        restore_best_weights=True,
    )
    
    with strategy.scope():
        optimizer = tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=WEIGHT_DECAY)
        model_breed.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
     )
    history_finetuning = model_breed.fit(
    training_dataset,
    validation_data=valid_dataset,
    epochs=NUM_EPOCHS,
    callbacks=[WandbCallback(), early_stop],
    )
    
model_breed.evaluate(training_dataset)
model_breed.evaluate(valid_dataset)

## save the model
model_breed.save("my_model_breeds")
