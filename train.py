import os
from cnn_dog_breeds import EXPERIMENT_NAME
import wandb
import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback
import tensorflow_addons as tfa
from tensorflow.keras import layers,regularizers,backend as K


SEED = 1234
tf.random.set_seed(SEED)
STATELESS_SEED = (SEED,SEED)

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


strategy  = tf.distribute.MirroredStrategy()

AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 3
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 256
NUM_EPOCHS = 150
PROJECTION_DIM = 64
NUM_HEADS = 6
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]  # Size of the transformer layers
TRANSFORMER_LAYERS = 4
MLP_HEAD_UNITS = [2048, 1024]
REG = regularizers.L1L2(l1=1e-4, l2=1e-4)

## define your own logging names for Wandb
CONFIG = dict (
    seed = SEED,
    model_name = 'EfficientNetL2',
    img_size = INPUT_SHAPE,
    num_classes = NUM_CLASSES,
    num_epochs = NUM_EPOCHS,
    batch_size = BATCH_SIZE,
)

EXPERIMENT_NAME = "No_aug"
wandb.init(project = "CNN_tests",
group = "Chihuahua_only",
name = EXPERIMENT_NAME,
job_type = "EfficientNetL2",
config = CONFIG
)
## Path do image containing the image dataset
DIRPATH = r''

# path to TFRrecord files
RECORD_PATH = r''

filenames = os.listdir(RECORD_PATH)
for i in  range(len(filenames)):
    filenames[i] = os.path.join(RECORD_PATH,filenames[i])

feature_description = {
    'image': tf.io.FixedLenFeature((), tf.string),
    'label': tf.io.FixedLenFeature((), tf.int64),
    'height': tf.io.FixedLenFeature((), tf.int64),
    'width': tf.io.FixedLenFeature((), tf.int64),
    'depth': tf.io.FixedLenFeature((), tf.int64)
}

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

def _parse_image_function(example_proto):
    """Function parsing TFRecrod files into image and label"""
    features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.parse_tensor(features['image'], tf.float32)
    image.set_shape([IMAGE_HEIGHT*IMAGE_WIDTH*3])
    image = tf.reshape(image, [IMAGE_WIDTH, IMAGE_HEIGHT, 3])
    image = tf.image.resize(image,(IMAGE_WIDTH,IMAGE_HEIGHT))
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def read_dataset(filename_list):
    """Function to read and decode TFRecord files into TRecordDataset object."""
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  
    dataset = tf.data.TFRecordDataset(filename_list)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(len(list(dataset)),reshuffle_each_iteration=False)
    return dataset

def create_tfr_dataset(filenames,
val_size=0.2,
batch_size = 32,
preprocess = False,
preprocess_fn = None
):
    """Function to create train, test, valid and full datasets from TFRecord dataset."""
    dataset_train = read_dataset(filenames)
    if preprocess:
        dataset_train = dataset_train.map(lambda x,y :(preprocess_fn(x),y))
    dataset_full = dataset_train
    ds_length = len(list(dataset_train))
    val_len = round(val_size*ds_length)
    test_len = round(val_len/2)

    dataset_valid = dataset_train.take(val_len)
    dataset_train = dataset_train.skip(val_len)
    dataset_test = dataset_valid.take(test_len)
    dataset_valid = dataset_valid.skip(test_len)


    dataset_train = dataset_train.shuffle(ds_length,reshuffle_each_iteration=True)
    dataset_test = dataset_valid.shuffle(len(list(dataset_test)),reshuffle_each_iteration=True)
    dataset_valid = dataset_valid.shuffle(len(list(dataset_valid)),reshuffle_each_iteration=True)


    dataset_train = dataset_train.batch(batch_size,
    num_parallel_calls=AUTOTUNE,
    drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()

    dataset_valid = dataset_valid.batch(batch_size,
    num_parallel_calls=AUTOTUNE,
    drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()

    dataset_test = dataset_test.batch(batch_size,
    num_parallel_calls=AUTOTUNE,
    drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()

    dataset_full = dataset_full.batch(batch_size,
    num_parallel_calls=AUTOTUNE,
    drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()
   
    return dataset_train, dataset_valid, dataset_test, dataset_full



def create_image_dataset(dirpath,
batch_size = 32,
img_w = 100, 
img_h = 100,
val_split = 0.2,
shuffle = True,
preprocess = False,
preprocess_fn = None
):
    """Function to create train, test, valid and full datasets from folder containing
    labeled photos."""
    dataset_train = tf.keras.utils.image_dataset_from_directory(
        dirpath,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_h, img_w),
        shuffle=shuffle,
        seed=SEED,
        validation_split=val_split,
        subset='training',
        interpolation='bilinear',

    )
    dataset_valid = tf.keras.utils.image_dataset_from_directory(
        dirpath,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_h, img_w),
        shuffle=shuffle,
        seed=SEED,
        validation_split=val_split,
        subset='validation',
        interpolation='bilinear',

    )

    dataset_full = tf.keras.utils.image_dataset_from_directory(
        dirpath,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_h, img_w),
        shuffle=shuffle,
        seed=SEED,
        validation_split=None,
        subset=None,
        interpolation='bilinear')
    test_len = np.floor(0.5*len(list(dataset_valid)))
    dataset_test = dataset_valid.take(test_len)
    dataset_valid = dataset_valid.skip(test_len)
    if preprocess:
        dataset_train = dataset_train.map(lambda x,y :(preprocess_fn(x),y))
        dataset_valid = dataset_valid.map(lambda x,y :(preprocess_fn(x),y))
        dataset_test = dataset_test.map(lambda x,y :(preprocess_fn(x),y))
        dataset_full = dataset_full.map(lambda x,y :(preprocess_fn(x),y))
    dataset_train = dataset_train.prefetch(buffer_size=AUTOTUNE).cache()
    dataset_valid = dataset_valid.prefetch(buffer_size=AUTOTUNE).cache()
    dataset_test = dataset_test.prefetch(buffer_size=AUTOTUNE).cache()
    dataset_full = dataset_full.prefetch(buffer_size=AUTOTUNE).cache()
    return dataset_train, dataset_valid, dataset_test, dataset_full

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        """Define the scheduler behaviour at the end of a batch."""
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        """Define the scheduler behaviour at the beggining of a batch."""
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning rate to %s.' % (self.global_step , lr))

# uncomment if using Cosine Warmup
# total_steps = int((len(list(dataset_train)) / batch_size) * num_epochs)
# warmup_epoch = 10
# warmup_steps = int(warmup_epoch * len(list(dataset_train)) / batch_size)
# warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate,
#                                             total_steps=total_steps,
#                                             warmup_learning_rate=0.0,
#                                             warmup_steps=warmup_steps,
#                                             hold_base_rate_steps=0,
#                                             verbose=1)



def create_network1():
    """Function to create model using the combined achitecture of pre-trained CNN with
    standard dense layer classifier"""
    base_model = tf.keras.applications.efficientnet.EfficientNetB7(input_shape=INPUT_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
 
    base_model.trainable = False

    model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(), 
    tf.keras.layers.Dense(256, kernel_regularizer = REG),
    tf.keras.layers.Dense(128,kernel_regularizer = REG),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def create_network2():
    """Function to create model using the combined achitecture of pre-trained CNN as
    feature extractor and ViT as classifier"""
    base_model = tf.keras.applications.efficientnet.EfficientNetB7(input_shape=INPUT_SHAPE,
                                            include_top=False,
                                            weights='imagenet')


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


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
    restore_best_weights=True,
)

dataset_train, dataset_valid, dataset_test, dataset_full = create_image_dataset(
    DIRPATH,
    batch_size=BATCH_SIZE,
    img_w=IMAGE_WIDTH,
    img_h=IMAGE_HEIGHT
)


with strategy.scope():

    adamw = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
    )
    optimizer = adamw
    model_age = create_network1()
    model_age.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tfa.metrics.CohenKappa(NUM_CLASSES),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tfa.metrics.F1Score(NUM_CLASSES)
    ]
    )

history = model_age.fit(
    dataset_train,
    validation_data=dataset_valid,
    epochs=NUM_EPOCHS,
    callbacks = [
    WandbCallback(),
    early_stop
    ]
)
model_age.save('models\my_model')
print("Test dataset:")
eval_test = model_age.evaluate(dataset_test)
print("Valid dataset:")
eval_valid = model_age.evaluate(dataset_valid)
print("Train dataset:")
eval_train = model_age.evaluate(dataset_train)
print("Full dataset:")
eval_full = model_age.evaluate(dataset_full)