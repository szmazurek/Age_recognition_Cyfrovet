# Dog age assessment project led by CyfroVet, an initiative for veterinary medicine created by ACC Cyfronet AGH in Cracow.

The main goal of this work is to utilize machine learning models to improve the wellbeing of dogs.
The system shows improvement beyond state-of-the-art, but the results are still not satisfying. CyfroVet is working on improving this along with approaching the problem with examination of videos showing the dogs. This repository contains the following files:
- train.py - pipeline and model that can be trained to classify dogs into three age groups. 
- ga_dataset_assesment.py - a tool based on GA to exclude the least informative images from the dataset
- cnn_dog_breeds.py - a script with pipeline and model for dog breed classification task based on stanford dogs dataset
- tfrecords_writer.py - a script for saving image dataset in form of TFRecord files
- function_library.py - base of functions developed for some repeating tasks in this project. **WARNING** Some of them may not be compatibile with current scripts as they were developed in the early stages of the project

**USAGE GUIDE**

1. Install packages from requirements.txt (requirements file to be introduced soon)
2. [train.py](https://github.com/SzymonMazurekAGH/Age_recognition_Cyfrovet/blob/main/train.py)
   1. Open file in the editor.
   2. Insert path to image folder into `DIRPATH` constant or path to TFRecord files into `RECORD_PATH` constant.
   3. Create datasets using `create_image_dataset` function if using image dataset or `create_tfr_dataset` if using TFRecord files dataset.
   4. Create CNN model using `create_network1` function or create CNN + ViT model using `create_network2` function.
   5. Configure `wand.init` call if using [Weights and Biases](https://wandb.ai/site) to track your experiments (if no tracking is used `wandb.init` call and `WandbCallback` in `fit` method call can be removed).
   6. Run the script.
3. [cnn_dog_breeds.py](https://github.com/SzymonMazurekAGH/Age_recognition_Cyfrovet/blob/main/cnn_dog_breeds.py)
   1. Open file in the editor.
   2. Create CNN model using `create_network1` function or create CNN + ViT model using `create_network2` function.
   3. Dataset is automatically loaded from [Tensorflow Datasets](https://www.tensorflow.org/datasets/api_docs/python/tfds) . If different data loading is needed please refer to the documentation under the link above.
   4. Configure `wand.init` call if using [Weights and Biases](https://wandb.ai/site) to track your experiments (if no tracking is used `wandb.init` call and `WandbCallback` in `fit` method call can be removed).
   5. Run the script.
4. [tfrecords_writer.py](https://github.com/SzymonMazurekAGH/Age_recognition_Cyfrovet/blob/main/tfrecords_writer.py)
   1. Open file in the editor.
   2. Insert path to folder with source images into `DIR_PATH` constant and path to save TFRecord files into `RECORD_PATH` constant.
   3. Change `BEST_NUM` constant to specify how many photos should be included into single TFRecord file.
   4. Run the script.
5. [ga_dataset_assesment](https://github.com/SzymonMazurekAGH/Age_recognition_Cyfrovet/blob/main/ga_dataset_assesment.py) (This algorithm is currently being tested on CIFAR dataset)
   1. Open file in the editor.
   2. Insert paths to:
      1. Images - into `np.load` function creating `data_inputs_cifar` variable
      2. Labels - into `np.load` function creating `labels_cifar` variable
      3. Invalid images - into `np.load` function creating `data_inputs_noise` variable
      4. Invalid labels - into `np.load` function creating `labels_noise` variable
      5. Initial data order (indexes of images after shuffling) - into `solutions_dir` variable
   3. If needed, configure the algorithm parameters specified with constants in the beginning of the file.
   4. Configure `wand.init` call if using [Weights and Biases](https://wandb.ai/site) to track your experiments (if no tracking is used `wandb.init` call and `WandbCallback` in `fit` method call can be removed).
   5. Run the script.
6. [function_library.py](https://github.com/SzymonMazurekAGH/Age_recognition_Cyfrovet/blob/main/function_library.py)
   1. Import the file to your script and call the methods as needed.
   2. Be sure to use correct paths for `DETECTOR` and `PREDICTOR` - use files provided in [Utils](https://github.com/SzymonMazurekAGH/Age_recognition_Cyfrovet/tree/main/Utlis) folder.

**DATASETS**
Datasets, as well as conference paper describing research done with these tools will be published soon.
Currently published datasets are avaliable:
https://tinyurl.com/dog-age-datasets
For more information about this project please contact the owner via email: s.mazurek@cyfronet.pl
