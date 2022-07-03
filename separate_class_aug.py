# A_PATH = '/net/scratch/people/plgmazurekagh/TFRecords_ds/TFR_Huge_dataset/Adult'
# Y_PATH = '/net/scratch/people/plgmazurekagh/TFRecords_ds/TFR_Huge_dataset/Young'
# S_PATH = '/net/scratch/people/plgmazurekagh/TFRecords_ds/TFR_Huge_dataset/Senior'


# adult_filenames = os.listdir(A_PATH)
# young_filenames = os.listdir(Y_PATH)
# senior_filenames = os.listdir(S_PATH)
# for i in range(len(adult_filenames)):
#     adult_filenames[i] = os.path.join(A_PATH,adult_filenames[i])

# for i in range(len(young_filenames)):
#     young_filenames[i] = os.path.join(Y_PATH,young_filenames[i])

# for i in range(len(senior_filenames)):
#     senior_filenames[i] = os.path.join(S_PATH,senior_filenames[i])

# start = time.time()

# a_path = '/net/scratch/people/plgmazurekagh/dogs_datasets/FaceDetector_cropped/Huge_A/Adult'
# y_path = '/net/scratch/people/plgmazurekagh/dogs_datasets/FaceDetector_cropped/Huge_Y/Young'
# s_path = '/net/scratch/people/plgmazurekagh/dogs_datasets/FaceDetector_cropped/Huge_S/Senior'

# a_labels = []
# s_labels = []
# y_labels = []


# for _ in range(len(os.listdir(a_path))):
#     a_labels.append(0)
# for _ in range(len(os.listdir(s_path))):
#     s_labels.append(1)
# for _ in range(len(os.listdir(y_path))):
#     y_labels.append(2)



# adult_dataset =  tf.keras.utils.image_dataset_from_directory(
#         A_PATH,
#         labels=a_labels,
#         label_mode='int',
#         class_names=None,
#         color_mode='rgb',
#         batch_size=None,
#         image_size=(224, 224),
#         shuffle=True,
#         seed=SEED,
#         validation_split=None,
#         subset=None,
#         interpolation='bilinear')

# senior_dataset = tf.keras.utils.image_dataset_from_directory(
#         S_PATH,
#         labels=s_labels,
#         label_mode='int',
#         class_names=None,
#         color_mode='rgb',
#         batch_size=None,
#         image_size=(224, 224),
#         shuffle=True,
#         seed=SEED,
#         validation_split=None,
#         subset=None,
#         interpolation='bilinear')

# young_dataset  = tf.keras.utils.image_dataset_from_directory(
#         Y_PATH,
#         labels=y_labels,
#         label_mode='int',
#         class_names=None,
#         color_mode='rgb',
#         batch_size=None,
#         image_size=(224, 224),
#         shuffle=True,
#         seed=SEED,
#         validation_split=None,
#         subset=None,
#         interpolation='bilinear')


# adult_dataset = read_dataset(adult_filenames)
# young_dataset = read_dataset(young_filenames)
# senior_dataset = read_dataset(senior_filenames)
# print(f'Created initial datasets in {time.time() - start} seconds.')




# start = time.time()

# print(f'Len of adult dataset: {len(list(adult_dataset))}')
# print(f'Len of senior dataset: {len(list(senior_dataset))}')
# print(f'Len of young dataset: {len(list(young_dataset))}')

# valid_a = adult_dataset.take(5000)
# adult_dataset = adult_dataset.skip(25000)

# valid_s = senior_dataset.take(5000)
# senior_dataset = senior_dataset.skip(5000)

# valid_y = young_dataset.take(5000)
# young_dataset = young_dataset.skip(5000)

# valid_dataset = valid_a.concatenate(valid_s).concatenate(valid_y)
# print(f'Length of valid dataset before split: {len(list(valid_dataset))}')
# test_len = round(0.5*len(list(valid_dataset)))
# test_dataset = valid_dataset.take(test_len)
# valid_dataset = valid_dataset.skip(test_len)

# print(f'Length of valid dataset after split: {len(list(valid_dataset))}')
# print(f'Length of test dataset : {test_len}')

# senior_dataset = senior_dataset.repeat(6).map(lambda x,y :(data_augmentation(x),y))
# #young_dataset = young_dataset.repeat(2).map(lambda x,y :(data_augmentation(x),y))

# print(f'Len of adult dataset after augmentation: {len(list(adult_dataset))}')
# print(f'Len of senior after augmentation: {len(list(senior_dataset))}')
# print(f'Len of young after augmentation: {len(list(young_dataset))}')

# training_dataset = adult_dataset.concatenate(senior_dataset).concatenate(young_dataset)

# print(f'Length of training dataset: {len(list(training_dataset))}')


# training_dataset = training_dataset.shuffle(len(list(training_dataset)),reshuffle_each_iteration=True)
# valid_dataset = valid_dataset.shuffle(len(list(valid_dataset)),reshuffle_each_iteration=True)
# test_dataset = test_dataset.shuffle(len(list(test_dataset)),reshuffle_each_iteration=True)
# training_dataset = training_dataset.batch(256,num_parallel_calls=AUTOTUNE,drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()
# valid_dataset = valid_dataset.batch(256,num_parallel_calls=AUTOTUNE,drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()
# test_dataset = test_dataset.batch(256,num_parallel_calls=AUTOTUNE,drop_remainder=False).prefetch(buffer_size=AUTOTUNE).cache()

#print(f'Processed all datasets in {time.time() - start} seconds.')