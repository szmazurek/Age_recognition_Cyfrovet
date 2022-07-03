"""Experimental implementation of GA to clear the image dataset from noisy samples.
Samples have to be loaded as numpy array (it is recommended to save dataset to .npy file as
loading the data every time takes a long time."""
import os
import random
import pygad
import pygad.kerasga
import wandb
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import StratifiedKFold

SEED = 1234
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
random.seed(SEED)
tf.random.set_seed(SEED)
strategy = tf.distribute.MirroredStrategy()
AUTOTUNE = tf.data.AUTOTUNE

NUM_CLASSES = 10
IMAGE_HEIGHT = 75
IMAGE_WIDTH = 75
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
BATCH_SIZE = 64
EPOCHS = 50
NUM_GENERATIONS = 200
NUM_PARENTS_MATING = 3
SOL_PER_POP = 5
NUM_GENES = 1000
PARENT_SELECTION_TYPE = "rank"
MUTATION_TYPE = "adaptive"
MUTATION_PROBABILITY = [0.5,0.2]
N_SPLIT = 5

CONFIG = dict (
    seed = SEED,
    model_name = "Inception_resnet_v2",
    img_size = INPUT_SHAPE,
    num_classes = NUM_CLASSES,
    epochs_per_sol = EPOCHS,
    batch_size = BATCH_SIZE,
    num_genes = NUM_GENES,
    num_generations = NUM_GENERATIONS,
    num_parents_mating = NUM_PARENTS_MATING,
    sol_per_pop = SOL_PER_POP,
    parent_selection_type = PARENT_SELECTION_TYPE,
    mutation_type = MUTATION_TYPE,
    mutation_probability = MUTATION_PROBABILITY
)

EXPERIMENT_NAME = "GA_run_rank_adaptive_1"
wandb.init(project = "GA_tests",
group = "CIFAR_tests",
name = EXPERIMENT_NAME,
job_type = "Inception_resnet_v2",
config = CONFIG
)

##directory to save best solutions after each generation
solutions_dir = ''
SOLUTION_PATH = os.path.join(solutions_dir,EXPERIMENT_NAME)
if not os.path.exists(SOLUTION_PATH):
    os.mkdir(SOLUTION_PATH)

def compile_function(model,optimizer,num_classes : int = 3,one_hot : bool = False):
    """Function to compile model (created for code readability, as the compilation has to be
    executed every generation)."""
    if one_hot:
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
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tfa.metrics.F1Score(num_classes)
        ],
    )

def prepare_dataset(x_inputs, y_inputs,preprocess_fn = None,preprocess = False,one_hot = False):
    """Function to create dataset from numpy array"""
    if preprocess:
        image = preprocess_fn(x_inputs)
        label = tf.cast(y_inputs,dtype=tf.float32)
        image = tf.cast(image, dtype=tf.float32)
    else:
        label = tf.cast(y_inputs,dtype=tf.float32)
        image = tf.cast(x_inputs, dtype=tf.float32)
    if one_hot:
        label = tf.keras.utils.to_categorical(label, NUM_CLASSES)
        dataset = tf.data.Dataset.from_tensor_slices((image, label))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((image, label))
    dataset = dataset.map(lambda x,y : (tf.image.resize(x,[IMAGE_HEIGHT,IMAGE_WIDTH]),y))
    dataset = dataset.shuffle(len(list(dataset)))
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=False,num_parallel_calls=AUTOTUNE
    ).prefetch(buffer_size=AUTOTUNE).cache()
    return dataset

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
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def fitness_func_kfold(solution, sol_idx):
    """This fitness function is based on trying to train the model wth given samples chosen by GA.
    The fitness is measured by mean F1 score achieved on evaluation dataset, which does not change.
    Training is done using Stratified Kfold cross validation.
    """
    global DATA_INPUTS,LABELS, EVALUATED_MODEL,INIT_WEIGHTS,EVAL_DATASET
    y_train = LABELS[solution]
    x_train  = DATA_INPUTS[solution]
    EVALUATED_MODEL.set_weights(INIT_WEIGHTS)
    optimizer = tf.keras.optimizers.Adam()
    compile_function(EVALUATED_MODEL,optimizer,one_hot=True,num_classes=NUM_CLASSES)
    skf = StratifiedKFold(n_splits=N_SPLIT, shuffle=True,random_state=SEED)
    for fold_number, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train[test_index]
        train_dataset = prepare_dataset(x_train_fold,
        y_train_fold,
        one_hot=True,
        preprocess_fn=resnet_preprocess,
        preprocess=True
        )
        valid_dataset = prepare_dataset(x_test_fold,
        y_test_fold,
        one_hot=True,
        preprocess_fn=resnet_preprocess,
        preprocess=True
        )
        print(f'============= Fold#{fold_number+1} =============')
        EVALUATED_MODEL.fit(
            x=train_dataset,
            verbose = 1,
            validation_data=valid_dataset,
            epochs = EPOCHS,
            callbacks = [early_stop]
        )

    metrics = EVALUATED_MODEL.evaluate(EVAL_DATASET,verbose = 1,return_dict = True)
    solution_fitness = np.mean(metrics['f1_score'])
    return solution_fitness

def fitness_func_normal(solution, sol_idx):
    """This fitness function is based on trying to overfit the model during the training.
    With noisy images in the sample present, the model will not be able to overfit. Therefore
    highest accuracy should correspond to highest quality photos that are relevant for the problem.
    Use when validation dataset is unavaliable for the problem."""
    global DATA_INPUTS,LABELS, EVALUATED_MODEL, INIT_WEIGHTS
    y_train = LABELS[solution]
    x_train  = DATA_INPUTS[solution]
    EVALUATED_MODEL.set_weights(INIT_WEIGHTS)
    optimizer = tf.keras.optimizers.Adam()
    compile_function(EVALUATED_MODEL,optimizer,one_hot=True,num_classes=NUM_CLASSES)
    train_dataset = prepare_dataset(
        x_train,
        y_train,
        one_hot=True,
        preprocess=True,
        preprocess_fn=resnet_preprocess
        )
    history = EVALUATED_MODEL.fit(
        x=train_dataset,
        verbose = 1,
        epochs = EPOCHS,
    )
    solution_fitness = np.max(history.history['categorical_accuracy'])
    return solution_fitness

def on_generation(ga_instance):
    """Function called on every generation finish. Logs the fitness into Wandb and
    saves best solution of every generation in a file"""
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Best solution fitness = {best_fitness}"
    .format(best_fitness = ga_instance.best_solutions_fitness[-1]))
    best_solution_gen = ga_instance.best_solutions[-1]
    wandb.log(
        {
        'fitness (accuracy)' : ga_instance.best_solutions_fitness[-1],
        'generation' : ga_instance.generations_completed
        },
        step = ga_instance.generations_completed,
        commit = True
    )
    np.save(os.path.join(SOLUTION_PATH,
    f"generation_{ga_instance.generations_completed}"),
    best_solution_gen
    )


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
    restore_best_weights=True,
)


resnet_preprocess =  tf.keras.applications.inception_resnet_v2.preprocess_input
EVALUATED_MODEL = create_network1()
INIT_WEIGHTS = EVALUATED_MODEL.get_weights()
data_test_cifar = np.load(r'')
label_test_cifar = np.load(r'')
EVAL_DATASET = prepare_dataset(data_test_cifar,
label_test_cifar,
one_hot=True,
preprocess_fn=resnet_preprocess,
preprocess=True)

data_inputs_cifar = np.load(r'')
labels_cifar = np.load(r'')
data_inputs_noise = np.load(r'')
labels_noise = np.load(r'')
DATA_INPUTS = np.concatenate((data_inputs_cifar,data_inputs_noise),axis = 0)
labels = np.concatenate((labels_cifar,labels_noise), axis=0)

## shuffling the input data before starting using GA algorithm
## to properly recover the data one should save the shuffled files before running GA
## to be able to indentify the best samples determined by GA
## the GA determines indexes of the best samples withing data_inputs array
## the GA will save the solution if the result is better than previous best one

shuffled_data = list(zip(DATA_INPUTS,labels))
random.shuffle(shuffled_data)
DATA_INPUTS, labels = zip(*shuffled_data)
DATA_INPUTS = np.asarray(DATA_INPUTS,dtype=np.float32)
LABELS = np.asarray(labels,dtype=np.float32)
np.save(os.path.join(SOLUTION_PATH,'shuffled_photos'),DATA_INPUTS)
np.save(os.path.join(SOLUTION_PATH,'shuffled_labels'),LABELS)

## creating GA instance and running it


genetic_algorithm = pygad.GA(num_generations=NUM_GENERATIONS,
                       num_parents_mating=NUM_PARENTS_MATING,
                       sol_per_pop=SOL_PER_POP,
                       num_genes=NUM_GENES,
                       init_range_low=0,
                       init_range_high=len(DATA_INPUTS)-1,
                       gene_type=int,
                       parent_selection_type=PARENT_SELECTION_TYPE,
                       mutation_type=MUTATION_TYPE,
                       mutation_probability=MUTATION_PROBABILITY,
                       fitness_func=fitness_func_normal,
                       on_generation=on_generation,
                       save_best_solutions=True
                       )

genetic_algorithm.run()
