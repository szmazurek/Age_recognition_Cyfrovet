from CyfroVet.ML_Veterinray_Metrics_Func import Face_detector

from tensorflow.keras.preprocessing import image
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
import cv2


def simplify_image(image: np.ndarray, slice_shape: tuple[int, int], pooling_func: callable = np.max):
    sliced_image = slice_img(image, slice_shape)
    if len(image.shape) == 2:
        arr = np.array([pooling_func(sl) for sl in sliced_image]).reshape((image.shape[0] // slice_shape[0],
                                                                           image.shape[1] // slice_shape[1]))
    else:
        shape_0 = shape_1 = int(np.sqrt(image.shape[0]))
        arr = np.array([pooling_func(sl) for sl in sliced_image]).reshape((shape_0 // slice_shape[0],
                                                                           shape_1 // slice_shape[1]))
    return arr


def image_preprocess(filename: str, img_size: tuple[int, int]):
    img = cv2.imread(filename)
    x, x1, x2, y1, y2 = Face_detector(img)
    x = img[y1:y2, x1:x2]
    rgb_image = cv2.resize(x, img_size)
    image = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
    standardized_image = (image - np.mean(image)) / np.std(image)
    return standardized_image


def training_preprocessor(input_path: str, res: tuple = (100, 100)):
    img = image.load_img(input_path, target_size=res)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    img /= 255
    return img


def get_heatmap(img_path: str, res=(100, 100)):
    model = tf.keras.models.load_model("")
    img = training_preprocessor(img_path)
    conv_layer = model.get_layer(index=0)
    heatmap_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img)
        argmax = tf.argmax(predictions[0])
        loss = predictions[:, argmax]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    heatmap *= 255
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    heatmap = heatmap.squeeze()
    heatmap = cv2.resize(heatmap, res)
    return heatmap


def get_age_category(age: int) -> str:
    if age < 4:
        return 'junior'
    elif age < 7:
        return 'adult'
    else:
        return 'senior'


def read_data(path: str, only_dirs: bool = True) -> list[(str, str, str)]:
    # .DS_store is some kind of thrash
    disease_dirs = os.listdir(path)
    try:
        disease_dirs.remove('.DS_Store')
    except:
        pass
    final_paths = []
    for disease_dir in disease_dirs:
        current_diseases = []
        for disease in disease_dir.split(';'):
            if disease != '.DS_Store':
                current_diseases.append(disease)
        for year_dir in os.listdir(f'{path}/{disease_dir}'):
            if year_dir != '.DS_Store':
                if only_dirs:
                    final_paths.append((f'{path}/{disease_dir}/{year_dir}', current_diseases,
                                        get_age_category(int(year_dir[:-1]))))
                else:
                    for photo in os.listdir(f'{path}/{disease_dir}/{year_dir}'):
                        if photo != '.DS_Store':
                            final_paths.append((f'{path}/{disease_dir}/{year_dir}/{photo}', current_diseases,
                                                get_age_category(int(year_dir[:-1]))))

    return final_paths


def slice_img(image: np.ndarray, kernel_size: tuple):
    tile_height, tile_width = kernel_size
    if len(image.shape) == 3:
        img_height, img_width, channels = image.shape
        tiled_array = image.reshape(img_height // tile_height,
                                    tile_height,
                                    img_width // tile_width,
                                    tile_width,
                                    channels)
        tiled_array = tiled_array.swapaxes(1, 2)
        tiled_array = tiled_array.reshape(-1, kernel_size[0], kernel_size[1], 3)

    elif len(image.shape) == 2:
        img_height, img_width = image.shape
        tiled_array = image.reshape(img_height // tile_height,
                                    tile_height,
                                    img_width // tile_width,
                                    tile_width)

        tiled_array = tiled_array.swapaxes(1, 2)
        tiled_array = tiled_array.reshape(-1, kernel_size[0], kernel_size[1])

    elif len(image.shape) == 1:
        img_height = int(np.sqrt(image.shape[0]))
        img_width = int(np.sqrt(image.shape[0]))
        tiled_array = image.reshape(img_height // tile_height,
                                    tile_height,
                                    img_width // tile_width,
                                    tile_width)
        tiled_array = tiled_array.swapaxes(1, 2)
        tiled_array = tiled_array.reshape(-1, kernel_size[0], kernel_size[1])
    return tiled_array


def sum_mistakes(predictions: list, real_diseases: list):
    final_vector = dict()
    for prediction, real_disease in zip(predictions, real_diseases):
        for k, v in prediction.items():
            final_vector[k] = final_vector.get(k, {})
            if set(real_disease) == set(k):
                final_vector[k]['True positive'] = final_vector[k].get('True positive', 0) + v
                final_vector[k]['False negative'] = final_vector[k].get('False negative', 0) + 1 - v
                final_vector[k]['True negative'] = final_vector[k].get('True negative', 0)
                final_vector[k]['False positive'] = final_vector[k].get('False positive', 0)
                final_vector[k]['Positive'] = final_vector[k].get('Positive', 0) + 1
            else:
                final_vector[k]['True positive'] = final_vector[k].get('True positive', 0)
                final_vector[k]['False negative'] = final_vector[k].get('False negative', 0)
                final_vector[k]['True negative'] = final_vector[k].get('True negative', 0) + 1 - v
                final_vector[k]['False positive'] = final_vector[k].get('False positive', 0) + v
                final_vector[k]['Negative'] = final_vector[k].get('Negative', 0) + 1
    return final_vector


def get_precision_recall(vector):
    precisions = {}
    recalls = {}
    for disease_combo, values in vector.items():
        try:
            if values['Positive'] > 0:
                precisions[disease_combo] = values['True positive'] / (
                            values['True positive'] + values['False positive'])
                recalls[disease_combo] = values['True positive'] / (values['True positive'] + values['False negative'])
        except KeyError:
            pass
    return precisions, recalls


def weighted_f_score(predictions: list, real_diseases: list, beta: float = 1):
    """
    example input data:
        predictions = [{('blind',): 0.3, ('bone_cancer',): 0.4, ('blind', 'bone_cancer'): 0},
                   {('blind',): 0.9, ('bone_cancer',): 0.1, ('blind', 'bone_cancer'): 0.2},
                   {('blind',): 0.1, ('bone_cancer',): 0.7, ('blind', 'bone_cancer'): 0.2}]

        real_diseases = [('blind',), ('blind',), ('bone_cancer',)]
    """
    scores = {}
    vector = sum_mistakes(predictions, real_diseases)
    precisions, recalls = get_precision_recall(vector)

    total_positive = 0
    total_score_sum = 0
    for disease in precisions:
        scores[disease] = (1 + beta ** 2) * (precisions[disease] * recalls[disease]) / (
                (beta ** 2) * precisions[disease] + recalls[disease])
        amount_of_positive_samples = vector[disease]['Positive']
        total_positive += amount_of_positive_samples
        total_score_sum += scores[disease] * amount_of_positive_samples

    return total_score_sum / total_positive
