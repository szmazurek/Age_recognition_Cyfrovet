"""A script to train the network for dog breed classification task using
Stanford Dogs Dataset. Inegration with Weights and biases provided."""
import wandb
import tensorflow as tf
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

training_dataset, valid_dataset = tfds.load(
    "stanford_dogs", as_supervised=True, split=["train", "test"], shuffle_files=True
)

training_dataset = (
    training_dataset.map(lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)))
    .batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE, drop_remainder=False)
    .prefetch(buffer_size=AUTOTUNE)
)


valid_dataset = (
    valid_dataset.map(lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)))
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

    adamw = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = adamw
    model_breed = create_cnn_network(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model_breed.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tfa.metrics.CohenKappa(NUM_CLASSES),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tfa.metrics.F1Score(NUM_CLASSES),
        ],
    )

model_breed.summary()

## train the model
history = model_breed.fit(
    training_dataset,
    validation_data=valid_dataset,
    epochs=NUM_EPOCHS,
    callbacks=[WandbCallback(), early_stop],
)

## save the model
model_breed.save("my_model_breeds")
