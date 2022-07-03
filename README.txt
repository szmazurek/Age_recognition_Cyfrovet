Repository for dog age assessment project led by CyfroVet, an initiative for veterinary medicine created by
ACC Cyfronet AGH in Cracow. The main goal of this work is to utilize machine learning models to improve the 
wellbeing of dogs. This repository contains the following files:
    - train.py - pipeline and model that can be trained to classify dogs into three age groups. The system shows
improvement beyon state-of-the-art, but the results are still not satisfying. CyfroVet is working on improving this
along with approaching the problem with examination of videos showing the dogs.
    - ga_dataset_assesment.py - a tool based on GA to exclude the least informative images from the dataset
    - cnn_dog_breeds.py - a script with pipeline and model for dog breed classification task based on stanford dogs dataset
    - tfrecords_writer.py - a script for saving image dataset in form of TFRecord files
    - function_library.py - base of functions developed for some repeating tasks in this project. Some of them may
not be compatibile with current scripts as they were developed in the early stages of the project