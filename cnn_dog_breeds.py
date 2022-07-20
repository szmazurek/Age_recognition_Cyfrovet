"""A script to train the network for dog breed classification  task using
Stanford Dogs Dataset. Inegration with Weights and biases provided."""
import wandb
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers,regularizers
import tensorflow_addons as tfa
from wandb.keras import WandbCallback

SEED = 1234
tf.random.set_seed(SEED)
STATELESS_SEED = (SEED,SEED)

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

strategy  = tf.distribute.MirroredStrategy()

AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 120
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 1024
NUM_EPOCHS = 50
PROJECTION_DIM = 64
NUM_HEADS = 6
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]  # Size of the transformer layers
TRANSFORMER_LAYERS = 4
MLP_HEAD_UNITS = [2048, 1024]
REG = regularizers.L1L2(l1=1e-4, l2=1e-4)

CONFIG = dict (
    seed = SEED,
    model_name = 'inception_resnet_v2',
    img_size = INPUT_SHAPE,
    num_classes = NUM_CLASSES,
    num_epochs = NUM_EPOCHS,
    batch_size = BATCH_SIZE,
)

EXPERIMENT_NAME = "inception_resnet_v2"
wandb.init(project = "Breed_classifier",
group = "Initial_run",
name = EXPERIMENT_NAME,
job_type = "inception_resnet_v2",
config = CONFIG
)

def mlp(x_0, hidden_units, dropout_rate):
    """Function to implement MLP in ViT."""
    for units in hidden_units:
        x_0 = layers.Dense(
            units,
            activation=tf.nn.gelu,
            kernel_regularizer=REG,
        )(x_0)
        x_0 = layers.Dropout(dropout_rate)(x_0)
    return x_0


class PatchEncoder(layers.Layer):
    """A layer to create patch embdeddings from input image when using ViT."""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.projection = layers.Dense(units=self.projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=self.projection_dim
        )

    def call(self, patch):
        """Call method for Patch encoder"""
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded



def data_augmentation(img_in):
    """Data augmentation fucntion"""
    img = tf.image.stateless_random_flip_up_down(img_in,seed = STATELESS_SEED)
    img = tf.image.stateless_random_flip_left_right(img,seed = STATELESS_SEED)
    img = tf.image.stateless_random_brightness(img,max_delta = 0.15,seed = STATELESS_SEED)
    img = tf.image.stateless_random_contrast(img,lower=0.1, upper=0.9,seed = STATELESS_SEED)
    return img


def create_network1():
    """Function to create model using the combined achitecture of pre-trained CNN with
    standard dense layer classifier"""
    base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
        )

    base_model.trainable = False

    model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, kernel_regularizer = REG),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512,kernel_regularizer = REG),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128,kernel_regularizer = REG),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def create_network2():
    """Function to create model using the combined achitecture of pre-trained CNN as
    feature extractor and ViT as classifier"""
    base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
        )

    base_model.trainable = False

    inputs = layers.Input(shape=INPUT_SHAPE)
    feature_extractor = base_model(inputs)
    shape_to_encode = feature_extractor.shape[-1]
    patches_num = feature_extractor.shape[1]**2
    reshaped_features = layers.Reshape((patches_num,shape_to_encode))(feature_extractor)
    encoded_patches = PatchEncoder(patches_num, PROJECTION_DIM)(reshaped_features)
    for _ in range(TRANSFORMER_LAYERS):
        x_1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x_1, x_1)
        # Skip connection 1.
        x_2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x_3 = layers.LayerNormalization(epsilon=1e-6)(x_2)
        # MLP.
        x_3 = mlp(x_3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x_3, x_2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    output = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    output = layers.Dense(NUM_CLASSES,activation = 'softmax')(output)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

#load and preprocess the data

training_dataset,valid_dataset = tfds.load(
    'stanford_dogs',
    as_supervised=True,
    split=['train','test'],
    shuffle_files=True
    )
training_dataset = training_dataset.map(
    lambda x,y : (tf.image.resize(x,[IMAGE_HEIGHT,IMAGE_WIDTH]),tf.one_hot(y,NUM_CLASSES))
    )
valid_dataset = valid_dataset.map(
    lambda x,y : (tf.image.resize(x,[IMAGE_HEIGHT,IMAGE_WIDTH]),tf.one_hot(y,NUM_CLASSES))
    )

training_dataset = training_dataset.map(
    lambda x,y : (tf.keras.applications.inception_resnet_v2.preprocess_input(x),y)
    )
valid_dataset = valid_dataset.map(
    lambda x,y : (tf.keras.applications.inception_resnet_v2.preprocess_input(x),y)
    )

training_dataset = training_dataset.batch(
    BATCH_SIZE,
    num_parallel_calls=AUTOTUNE,
    drop_remainder=False
).prefetch(buffer_size=AUTOTUNE).cache()

valid_dataset = valid_dataset.batch(
    BATCH_SIZE,
    num_parallel_calls=AUTOTUNE,
    drop_remainder=False
).prefetch(buffer_size=AUTOTUNE).cache()


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
    restore_best_weights=True,
)

with strategy.scope():

    adamw = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
        )
    optimizer = adamw
    model_breed = create_network1()
    model_breed.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tfa.metrics.CohenKappa(NUM_CLASSES),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tfa.metrics.F1Score(NUM_CLASSES)
    ],
    )

model_breed.summary()

## train the model
history = model_breed.fit(
    training_dataset,
    validation_data=valid_dataset,
    epochs=NUM_EPOCHS,
    callbacks = [
    WandbCallback(),
    early_stop
    ]
)

## save the model
model_breed.save('models\my_model')
