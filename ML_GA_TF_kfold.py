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
from sklearn.model_selection import StratifiedKFold
## basic configuration


DIRPATH = '/net/scratch/people/plgmazurekagh/Downloaded_imgs_Petfinder_copy'
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

num_classes = 3
image_height = 100
image_width = 100
input_shape = (image_height, image_width, 3)
SEED = 1234
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(SEED)

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = 256
EPOCHS = 100

N_SPLIT = 5
## helper function if the data has to be loaded from folder of images



early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
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

def normalize(img):
    img =  img/255.0
    return img
## creating training dataset from numpy arrays (created for code readability, as the dataset has to be
# created every generation)
def prepare_dataset(X_inputs, y_inputs,preprocess_fn = None,preprocess = False):

    dataset = tf.data.Dataset.from_tensor_slices((X_inputs, y_inputs))
    if preprocess == True:
        dataset = dataset.map(lambda x,y : (preprocess_fn(x),y))
    else:
        dataset = dataset.map(lambda x,y : (normalize(x),y))
    dataset = dataset.shuffle(len(list(dataset)))
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=False,num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE).cache()
    return dataset


## loading data from npy files (faster than reading form a file everytime we run expeiments)


def create_network1():
    base_model = tf.keras.applications.efficientnet.EfficientNetB7(input_shape=input_shape,
                                                include_top=False,
                                                weights='imagenet')

    base_model.trainable = False

    model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(), 
    tf.keras.layers.Dense(256
    ),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

## creating golden dataset to validate the fitness of every generation of training data



## fitness function
## the workflow is as follows: 
# 1. create solution (indexes of samples from training dataset)
# 2. train the network
# 3. validate the results on golden dataset
# 4. use evaluation accuracy as solution fitness 
# 5. start new generation
# There is still a need to use integration with Wandb to monitor the trainings and fitness of every generation


def fitness_func(solution, sol_idx):
    global data_inputs,labels, model, INIT_WEIGHTS,solution_dict,eval_dataset
    print(solution)
    y_train = labels[solution]
    X_train  = data_inputs[solution]
    model.set_weights(INIT_WEIGHTS)
    compile_function(model,optimizer,one_hot=True)
    skf = StratifiedKFold(n_splits=N_SPLIT, shuffle=True,random_state=SEED)
    for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        X_train_fold = X_train[train_index]
        y_train_fold = tf.keras.utils.to_categorical(y_train[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = tf.keras.utils.to_categorical(y_train[test_index])
        train_dataset = prepare_dataset(X_train_fold,y_train_fold)
        valid_dataset = prepare_dataset(X_test_fold,y_test_fold)
        print(f'============= Fold#{fold_number+1} =============')
        model.fit(
            x=train_dataset,
            verbose = 1,
            validation_data=valid_dataset,
            epochs = 50,
            callbacks = [early_stop]
        )

    metrics = model.evaluate(eval_dataset,verbose =1 , return_dict = True)
    solution_fitness = np.mean(metrics['f1_score'])
    print(solution_fitness)


    return solution_fitness

eval_dataset = tf.keras.utils.image_dataset_from_directory(
    r'/net/scratch/people/plgmazurekagh/dogs_datasets/FaceDetector_cropped/Expert_ds',
    labels='inferred',
    label_mode='categorical',
    class_names=['Adult','Senior','Young'],
    color_mode='rgb',
    batch_size=256,
    image_size=(100, 100),
    shuffle=True,
    seed=SEED,
    validation_split=None,

)
eval_dataset = eval_dataset.prefetch(buffer_size=AUTOTUNE).cache()

model = create_network1()
INIT_WEIGHTS = model.get_weights()
data_inputs = np.load(r'/net/people/plgmazurekagh/CyfroVet/ML_project/npy_datasets/Big_ds_new/Images.npy')
labels = np.load(r'/net/people/plgmazurekagh/CyfroVet/ML_project/npy_datasets/Big_ds_new/Labels.npy')  


## creating GA instance and running it
## 
num_generations = 200 
num_parents_mating = 5
def on_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop = 10,
                       num_genes = 1500,
                       init_range_low=0,
                       gene_type=int,
                       init_range_high=len(data_inputs)-1,
                       fitness_func=fitness_func,
                       on_generation = on_generation)

ga_instance.run()
