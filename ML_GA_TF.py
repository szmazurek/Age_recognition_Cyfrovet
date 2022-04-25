import random
from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa
import os


import pygad
import cv2
import pygad.kerasga
from matplotlib import pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input 
import wandb
from wandb.keras import WandbCallback

## basic configuration

os.environ["WANDB_NOTEBOOK_NAME"] = "ML_Veterinary_DataLoader.ipynb" 
os.environ["WANDB_API_KEY"] = "37db6daf3fa79fefaa88342b2be82f52bd0ad699" 
DIRPATH = '/net/scratch/people/plgmazurekagh/Downloaded_imgs_Petfinder_copy'
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


SEED = 1234
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(SEED)
preprocess = False
strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = 32
EPOCHS = 100
IMG_SIZE =100

## helper function if the data has to be loaded from folder of images
def createTrainingData(path_test):
    cat = os.listdir(DIRPATH)
    CATEGORIES = sorted(cat)
    labels = []
    imgs_array = np.empty((0,IMG_SIZE,IMG_SIZE,3))
    for folder in os.listdir(path_test):
        class_num  = CATEGORIES.index(folder)
        path = os.path.join(path_test,folder)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            imgs = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
            imgs = cv2.resize(imgs, (IMG_SIZE, IMG_SIZE))
            imgs = imgs.astype('float')
            imgs = np.expand_dims(imgs, axis=0)
            labels.append(class_num)
            imgs_array = np.append(imgs_array,imgs,axis=0)
            #imgs_array = np.stack((imgs_array,imgs))

    return imgs_array,labels

## blocks for transformer
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

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

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.projection = layers.Dense(units=self.projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=self.projection_dim
        )


    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


## function to create desired model
def create_model(model_name : str = 'vit', num_classes : int = 3 ,img_size : int = 100, one_hot : bool = False, 
trainable : bool = True, pretrained : bool = False):
    learning_rate = 0.0001
    weight_decay = 0.0001
    num_classes = num_classes
    input_shape = (img_size, img_size, 3)
    
    if model_name == 'vit':
        patch_size = 6  # Size of the patches to be extract from the input images
        num_patches = (img_size // patch_size) ** 2
        projection_dim = 64
        num_heads = 6
        transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]  # Size of the transformer layers
        transformer_layers = 8
        mlp_head_units = [2048, 1024]


        #with strategy.scope():
        inputs = layers.Input(shape=input_shape)

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


        optimizer = tfa.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            )
        if one_hot ==  True:
            loss = tf.keras.losses.CategoricalCrossentropy()
            accuracy = tf.keras.metrics.CategoricalAccuracy()
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
            accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                accuracy,
                tfa.metrics.CohenKappa(num_classes),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                tfa.metrics.F1Score(num_classes)
            ],
        )
        return model
    elif model_name == 'ResNet50':

        global preprocess 
        preprocess = True
        with strategy.scope():
            if pretrained == True:
                wt_name = 'Imagenet'
            else:
                wt_name = None
            ## I can not pass the shape argument for resnet creation, it has to be called from tensor that is prepared earlier
            base_model = tf.keras.applications.ResNet50(
                                                    input_tensor = tf.keras.Input(shape=input_shape),
                                                    include_top=False,
                                                    weights=wt_name)
            # base_model.load_weights("/net/people/plgmazurekagh/CyfroVet/ML_project/saved_model/breed_weights.h5",
            # by_name=True,skip_mismatch = True)
            if trainable == True:
                base_model.trainable = True
            else:
                base_model.trainable = False

            model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
        return model
    else:
        print('Model not found. Valid models are: \n'
        'vit\n'
        'ResNet50')


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    restore_best_weights=True,
)
## function to compile model (created for code readability, as the compilation has to be
# executed every generation)
def compile_function(model,optimizer,num_classes : int = 3,one_hot : bool = False):
    #with strategy.scope():
        if one_hot ==  True:
            loss = tf.keras.losses.CategoricalCrossentropy()
            accuracy = tf.keras.metrics.CategoricalAccuracy()
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
            accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                accuracy,
                tfa.metrics.CohenKappa(num_classes),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                tfa.metrics.F1Score(num_classes)
            ],
        )

learning_rate = 0.0001
weight_decay = 0.0001
optimizer = tfa.optimizers.AdamW(
                    learning_rate=learning_rate, weight_decay=weight_decay
                )   
## creating model and saving initial weights (we have to train the model from the same initial
# values every generation  )
model = create_model('vit')
INIT_WEIGHTS = model.get_weights()

def normalize(img):
    img =  img/255.0
    return img
## creating training dataset from numpy arrays (created for code readability, as the dataset has to be
# created every generation)
def prepare_dataset_train(X_inputs, y_inputs):

    dataset = tf.data.Dataset.from_tensor_slices((X_inputs, y_inputs))
    if preprocess == True:
        dataset = dataset.map(lambda x,y : (preprocess_input(x),y))
    else:
        dataset = dataset.map(lambda x,y : (normalize(x),y))
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=False,num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE).cache()
    return dataset


## loading data from npy files (faster than reading form a file everytime we run expeiments)
data_inputs = np.load(r'/net/people/plgmazurekagh/CyfroVet/ML_project/npy_datasets/Original_mix_dataset/Images.npy')
labels = np.load(r'/net/people/plgmazurekagh/CyfroVet/ML_project/npy_datasets/Original_mix_dataset/Labels.npy')
X_valid = np.load(r'/net/people/plgmazurekagh/CyfroVet/ML_project/npy_datasets/Expert_dataset/Images.npy')
y_valid = np.load(r'/net/people/plgmazurekagh/CyfroVet/ML_project/npy_datasets/Expert_dataset/Labels.npy')
y_valid = tf.keras.utils.to_categorical(y_valid)

## creating golden dataset to validate the fitness of every generation of training data
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
if preprocess == True:
    valid_dataset = valid_dataset.map(lambda x,y : (preprocess_input(x),y))
else:
    valid_dataset = valid_dataset.map(lambda x,y : (normalize(x),y))
valid_dataset = valid_dataset.shuffle(buffer_size=len(valid_dataset)).batch(BATCH_SIZE,drop_remainder=False,num_parallel_calls=AUTOTUNE)
valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE).cache()



## fitness function
## the workflow is as follows: 
# 1. create solution (indexes of samples from training dataset)
# 2. train the network
# 3. validate the results on golden dataset
# 4. use evaluation accuracy as solution fitness 
# 5. start new generation
# There is still a need to use integration with Wandb to monitor the trainings and fitness of every generation
def fitness_func(solution, sol_idx):
    global data_inputs,labels, model, INIT_WEIGHTS,solution_dict
    print(solution)
    t_labels = tf.keras.utils.to_categorical(labels[solution])
    train_dataset = prepare_dataset_train(data_inputs[solution],t_labels)
    model.set_weights(INIT_WEIGHTS)
    compile_function(model,optimizer,one_hot=True)
    model.fit(
        x=train_dataset,
        verbose = 1,
        validation_data=valid_dataset,
        epochs = 1,
        callbacks = [early_stop]
    )
    metrics = model.evaluate(valid_dataset,verbose =1 , return_dict = True)
    solution_fitness = metrics['categorical_accuracy']
    print(solution_fitness)


    return solution_fitness


    


## creating GA instance and running it
## 
num_generations = 200 
num_parents_mating = 5
def on_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop =10,
                       num_genes = 5000,
                       init_range_low=0,
                       gene_type=int,
                       init_range_high=len(data_inputs)-1,
                       fitness_func=fitness_func,
                       on_generation = on_generation)

ga_instance.run()
