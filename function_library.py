"""This script contains various functions to automate repeating task encountered during
the development of the project. Note that this script may be not compatibile with present models
as some of the functions were developed for the early stages of the project.
"""
import cv2
import string
import os
import shutil
import dlib
import tensorflow as tf
import numpy as np
from skimage import io
from pathlib import Path
from imutils import face_utils
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import correlate as corr
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error as MSE

## mean and std values for RGB channels obtained from a given dataset

mean_vals = {
    "Red" : 123,
    "Green" : 135,
    "Blue" : 146
}

std_vals = {
    "Red" : 68,
    "Green" : 69,
    "Blue" : 69
}

IMG_SIZE = 224
## Load detector libraries
DETECTOR = dlib.cnn_face_detection_model_v1(Path('utils\dogHeadDetector.dat'))
PREDICTOR = dlib.shape_predictor(Path('utils\landmarkDetector.dat'))

## age categories used for training the model, always in the alphabetical order
CATEGORIES = ["Adult", "Senior", "Young"]


def disp_hist(path : string, iteration_number : int = 0):

    """ Experimental - display histogram of image's rgb channels
    """
    if iteration_number == 0:
        iteration_number=len(os.listdir(path))
    else:
        iteration_number =  iteration_number
    piclist=os.listdir(path)
    for i in range(iteration_number):
        img_path=path+piclist[i]
        img = io.imread(img_path)
        _ = plt.hist(img.ravel(), bins = 256, color = 'orange', )
        _ = plt.hist(img[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
        _ = plt.hist(img[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
        _ = plt.hist(img[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
        _ = plt.xlabel('Intensity Value')
        _ = plt.ylabel('Count')
        _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
        plt.show()
        _ = plt.imshow(img)
        plt.show()

def pred_decoder(prediction):

    """ Decodes network's predictions
    """
    x = np.argmax(prediction)
    print(x)
    print("Prediction: " + CATEGORIES[x])
    prob = prediction[x]
    print("Confidence: "+ str(prob))
    return (CATEGORIES[x])


def feature_norm(input_img : np.ndarray):
    """ Normalizes RGB channels of input photo with respect to the
    dataset that network was trained on
    Args:
        input_img (np.ndarray): RGB image to be normalized
    Returns:
        (np.ndarray): normalized RGB image
    """
    r,g,b = cv2.split(input_img)
    r = (r - mean_vals["Red"])/std_vals["Red"]
    g = (g - mean_vals["Green"])/std_vals["Green"]
    b = (b - mean_vals["Blue"])/std_vals["Blue"]
    output_img = cv2.merge([r,g,b])

    return output_img



def training_preprocessor(input_path : string, res : tuple =(IMG_SIZE,IMG_SIZE)):
    """ Preprocesses the images from the input folder for neural network input

    Args:
        input_path (string): path to folder containing photographs
        res (tuple, optional): size of of output images

    Returns:
        (np.ndarray): preprocessed image with added batch dimesnion (4D ndarray)
    """
    img = cv2.imread(input_path)
    img = cv2.resize(img, (res[0],res[1]))
    img = img.astype(np.float32)
    img = feature_norm(img)
    img = np.expand_dims(img, axis=0)
    return img

def slice_img(image: np.ndarray, kernel_size: tuple):
    """ Divides input image to smaller parts (slices).
        Takes input as np.ndarray and tuple describing dimension of slices.
        Return 4D np.ndarray, containing subsequent slices of the image.

    Args:
        image (np.ndarray): image to be sliced
        kernel_size (tuple): size of slices

    Returns:
        (np.ndarray): 4D array containing slices 
    """

    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width,
                                channels)
    tiled_array = tiled_array.swapaxes(1, 2)
    tiled_array = tiled_array.reshape(-1,10,10,3)
    return tiled_array


def slice_visualizer(sliced_img : np.ndarray, slice_size : tuple):
    """ Method to visualize sliced image by combining together all slices

    Args:
        sliced_img (np.ndarray): sliced image
        slice_size (tuple): size of slices
    """
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=slice_size,  # creates 2x2 grid of axes
                    #axes_pad=0.1,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, sliced_img):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()

def equalize_image(img):
    """
    A funtion for histogram equqlization of image

    Parameters:
    img (np.ndarray): input image in numpay array format

    Returns:
    np.ndarray: equqlized image
    """
    b, g, r = cv2.split(img)

    b_equalize = cv2.equalizeHist(b)
    g_equalize = cv2.equalizeHist(g)
    r_equalize = cv2.equalizeHist(r)

    return cv2.merge((b_equalize, g_equalize, r_equalize))     


def predict_on_examples(dirpath : string,
model,
last_conv_index : int,
ground_truth : string ='Unknown',
display_image : bool = False, 
display_heatmap : bool = False, 
display_template : bool = False,
iteration_number : int = 0, 
res : tuple = (IMG_SIZE,IMG_SIZE), 
intensity : float = 0.8):
    """ method using trained model to asses age of a dog in input image
        can be configured to display the image, heatmap or heatmap template
        of many inputs

    Args:
        dirpath (string): path to directory containing photos.
        model: an tf.keras.Model or tf.keras.Sequential instance to examine.
        last_conv_index(int): index of last convolutional layer in the model.
        ground_truth (string, optional): label for input image. Defaults to 'Unknown'.
        display_image (bool, optional): show input image. Defaults to False.
        display_heatmap (bool, optional): show image heatmap. Defaults to False.
        display_template (bool, optional): show template image. Defaults to False.
        iteration_number (int, optional): [description]. Defaults to 0.
        res (tuple, optional): sets resolution of input image. Defaults to (100,100).
        intensity (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """

    if iteration_number == 0:
        iteration_number=len(os.listdir(dirpath))
    else:
        iteration_number =  iteration_number
    piclist=os.listdir(dirpath)
    hm_sum = np.zeros((res[0], res[1], 3, 1))
    for i in range(iteration_number):
        img_path=dirpath+piclist[i]
        if display_image:
            img = io.imread(img_path)
            plt.imshow(img)
            plt.show()
        img = training_preprocessor(img_path)
        preds = model.predict(img)[0]
        pred_decoder(preds)
        print("Ground_truth: "+ ground_truth)
        if display_heatmap or display_template :
            conv_layer = model.get_layer(index=last_conv_index)
            heatmap_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])
            with tf.GradientTape() as gtape:
                conv_output, predictions = heatmap_model(img)
                argmax=tf.argmax(predictions[0])
                loss = predictions[:, argmax]
                grads = gtape.gradient(loss, conv_output)
                pooled_grads = K.mean(grads, axis=(0,1,2))

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            max_heat = np.max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat

            img = cv2.imread(img_path)
            img = cv2.resize(img, res)
            heatmap = heatmap.squeeze()
            
            heatmap = cv2.resize(heatmap, res)
            heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_HOT)
            
            heatmap = equalize_image(heatmap)

            img_hm = heatmap * intensity + img    
            img_hm = cv2.cvtColor(img_hm.astype('float32'), cv2.COLOR_BGR2RGB)
            img_hm = img_hm/np.amax(img_hm)

            if display_heatmap:
                fig, ax = plt.subplots(1,3, figsize=(20, 20))
                ax[0].imshow(img)
                ax[0].set_title("Input image")
                ax[0].axis('off')
                ax[1].imshow(img_hm)
                ax[1].set_title("Image with heatmap")
                ax[1].axis('off')
                ax[2].matshow(heatmap)
                ax[2].axis('off')
                ax[2].set_title("Heatmap")
                plt.axis('off')
                plt.show()
            if display_template:
                features = features_detector(img)
                if len(features):
                    img = img_rotate(img, features, res=res)
                    heatmap = img_rotate(heatmap, features, res=res)

                features = features_detector(img)
                if len(features):
                    img = translate_image(img, (50, 50), features,  res=res)
                    heatmap = translate_image(heatmap, (50, 50), features,  res=res)
                    hm_sum = np.append(hm_sum, heatmap.reshape((res[0], res[1], 3, 1)), axis=3)
        
                    if i == iteration_number-1:
                        hm_sum_disp = hm_sum.sum(axis=3)
                        hm_mean = hm_sum.mean(axis=3)
                        hm_median = np.median(hm_sum, axis=3)
                        fig, ax = plt.subplots(1, 3, figsize=(20, 20))
                        ax[0].matshow(hm_sum_disp)
                        ax[0].title.set_text('heatmaps sum')
                        ax[1].matshow(hm_mean)
                        ax[1].title.set_text('heatmaps mean')
                        ax[2].matshow(hm_median)
                        ax[2].title.set_text('heatmaps median')
                        plt.show()
    if display_template:
        return hm_sum, hm_sum_disp, hm_mean, hm_median


def hm_loader(path : string):
    """ loads the heatmap from numpy array saved as txt file
        and reshapes it to original shape takes path to saved .txt as input
        returns reshaped template image

    Args:
        path (string): string to saved heatmap

    Returns:
        (np.ndarray): reshaped template image
    """
    
    template_pre =np.loadtxt(path)
    template = template_pre.reshape(template_pre.shape[0], template_pre.shape[1] // 3, 3)
    return template


def photo_detector(src_image: np.ndarray, res : tuple = (IMG_SIZE,IMG_SIZE)):
    """ detects an image on white background and crops it
        takes an image as input

    Args:
        src_image (np.ndarray): input image
        res (tuple, optional): output resolution of croped image. Defaults to (100,100).

    Returns:
        (np.ndarray): croped image from whit background
    """

    img_in = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    ret, thresh = cv2.threshold(gray, 254, 255, 1)

    photo_width = np.count_nonzero(thresh == 255, axis=0).max()
    photo_height = np.count_nonzero(thresh == 0, axis=1).min()

    column, row, photo_width, photo_height = cv2.boundingRect(thresh)

    result_image = src_image[row:row + photo_height, column: column + photo_width]

    return cv2.resize(result_image, res)


def features_detector(src_image: np.ndarray, equalize : bool = True):
    """ Locates facial features on the dog's face image
        takes an image as an input, can be configured to 
        equalize histogram before eature extraction.
        Returns coordinates of the features if detected

    Args:
        src_image (np.ndarray): source image 
        equalize (bool, optional): enables histogram equalization before feature localization. Defaults to True.

    Returns:
        (tuple): tuple of cordinations of facial features in order forehead, right_ear, right_eye, nose, left_ear, left_eye
    """
    if equalize:
        ycrvb_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2YCrCb)
        ycrvb_img[:, :, 0] = cv2.equalizeHist(ycrvb_img[:, :, 0])

        yuv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2YUV)
        yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])

        equlized_img1 = cv2.cvtColor(ycrvb_img, cv2.COLOR_YCrCb2BGR)
        dets = DETECTOR(equlized_img1, upsample_num_times=1)
    else:
        dets = DETECTOR(src_image, upsample_num_times=1)
    if len(dets):
        shape = PREDICTOR(equlized_img1, dets.pop().rect)
        (forehead, right_ear, right_eye, nose, left_ear, left_eye) = face_utils.shape_to_np(shape)

        return (forehead, right_ear, right_eye, nose, left_ear, left_eye)
    else:
        return ()


def img_rotate(src_image: np.ndarray, face_features: tuple, res : tuple =(IMG_SIZE, IMG_SIZE)):
    """ rotates the dog's face to make eyeline horizontal.
        Uses nose's cordinates as center point.
        Takes image nad list of face feature coordinates as input.
        Returns rotated image

    Args:
        src_image (np.ndarray): input image to be rotated
        face_features (tuple): tuple of face features
        res (tuple, optional): output image size. Defaults to (100, 100).

    Returns:
        (np.ndarray): rotated input image
    """

    (forehead, right_ear, (x2, y2), nose, left_ear, (x1, y1)) = face_features
    
    slope = (y2 - y1) / (x2 - x1)
    rad = np.arctan2(y2 - y1, x2 - x1)
    angle = np.rad2deg(rad)

    cX = int(nose[0])
    cY = int(nose[1])
    rotationMatrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotatedImage = cv2.warpAffine(src_image, rotationMatrix, res)

    return rotatedImage

## method 

def translate_image(src_image: np.ndarray, translate_destination: tuple, feautres: tuple, feature_id : int = 3, res : tuple =(IMG_SIZE, IMG_SIZE)):
    """[summary]

    Args:
        src_image (np.ndarray): image to be transleted
        translate_destination (tuple): tuple of destination cordinates of translation
        feautres (tuple): Facial features of dog;s face
        feature_id (int, optional): Face feature to be used as beging of translation. Defaults to 3.
        res (tuple, optional): Size of output image. Defaults to (100, 100).

    Returns:
        [type]: [description]
    """
    tX, tY = translate_destination
    bX, bY = feautres[feature_id]

    M = np.float32([[1, 0, tX - bX],[0, 1,  tY - bY]])

    return_image = cv2.warpAffine(src_image, M, res)
    return return_image


def check_triangle_angles(face_features: tuple, max_angle : float = 70.0):
    """ checks if dog's face features have proper geometrical features to be used in tamplate creation

    Args:
        face_features (tuple): tuple of dog's face features
        max_angle (float, optional): maximum angle allowed in dog's face features.
                                     Angle is checked between triangle formed by eyes and nose.
                                     Defaults to 70.0.

    Returns:
        (bool): True if maxium angle isn't excedeed
    """
    (forehead, right_ear, (x1, y1), (x2, y2), left_ear, (x3, y3)) = face_features

    # calculate distance between dog eyes 
    a = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    # calculate distance between dog left eye and nose 
    b = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    # calculate distance between dog right eye and nose 
    c = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    cos_a = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    cos_b = (c ** 2 + a ** 2 - b ** 2) / (2 * a * c)
    cos_c = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

    angle_a = np.rad2deg(np.arccos(cos_a))
    angle_b = np.rad2deg(np.arccos(cos_b))
    angle_c = np.rad2deg(np.arccos(cos_c))

    # print(angle_a, angle_b, angle_c)
    if angle_a < max_angle and angle_b < max_angle and angle_c < max_angle:
        return True

    return False


def face_detector(img: np.ndarray):
    """ detects and return posiion of dog face. Takes np.ndarray image and retuns positions for rectangle

    Args:
        img (np.ndarray): [description]

    Returns:
        (np.ndarray, int, int, int, int) or (np.ndarray): return input image with postions for dog face
        if no dog face wa detected than input image is returned
          
    """
    img_in = img

    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    x,y,w,h = cv2.boundingRect(cnt)
    dst = img_in[y:y+h, x:x+w]

    plt.imshow(dst)

    dets = DETECTOR(img_in, upsample_num_times =1)

    img_result = img_in.copy()

    for i, d in enumerate(dets):
        x1, y1 = d.rect.left(), d.rect.top()
 
        x2, y2 = d.rect.right(), d.rect.bottom()
        if y1 < 0:
            y1 = 0
        if x1 < 0:
            x1 = 0
        if x2 < 0:
            x2 = 0
        if y2 < 0:
            y2 = 0 
        
    shapes = []

    for i, d in enumerate(dets):
        shape = PREDICTOR(img_in, d.rect)
        shape = face_utils.shape_to_np(shape)
        shapes.append(shape)
    try:
        return x1, x2, y1, y2
    except:
        return img_result


def img_preprocess(src : string, thrash : string):
    """ method to crop images with face detector before inputting to network
        takes source folder and thrash folder paths as an input incorrect 
        photos will be copied to thrash correct will be cropped and saved 
        in the same folder

        run only ONCE on given folder

    Args:
        src (string): source folder containing images to be procesed 
        thrash (string): path to trash folder
    """
    for filename in os.listdir(src):
        string_e=src+filename
        img = cv2.imread(string_e)
        IMG_SIZE =224
        try:
            x1, x2, y1, y2 = face_detector(img)
            img = img[y1:y2,x1:x2]
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            resized_image = img
            # save same img
            cv2.imwrite(string_e, resized_image)
        except: ## if file not detected, copy to trash folder and remove from the source
            wrong_address = thrash
            shutil.copy(string_e,wrong_address) ## move incorrect files to other folder
            os.remove(string_e)
            continue


def max_val_ssim (img1 : tf.Tensor, img2 : tf.Tensor = None, single_arg : bool = False):
    """ Method returning max pixel value occuring in a pair of images passed as 4D tf.tensors.
        Takes pair of images as an input can be configured to one picture mode.
        Returns max pixel value in given images

    Args:
        img1 (tf.Tensor): first input image as tf.Tensor 
        img2 (tf.Tensor, optional): second input image as tf.Tensor. Defaults to None.
        single_arg (bool, optional): Cahnges mode to one picture. Defaults to False.

    Returns:
        (float): maxiumum value
    """
    if single_arg==True:
        max1 = tf.math.reduce_max(img1,axis=(0, 1))
        max1 = tf.math.reduce_max(max1,axis=(0, 1))
        max_val = max1.numpy()
        return max_val    
    max1 = tf.math.reduce_max(img1,axis=(0, 1))
    max1 = tf.math.reduce_max(max1,axis=(0, 1))
    max2 = tf.math.reduce_max(img2,axis=(0, 1))
    max2 = tf.math.reduce_max(max2,axis=(0, 1))
    max_val = tf.math.maximum(max1,max2).numpy()
    return max_val


def metrics_calculator(input_img : np.ndarray, template_img : np.ndarray):
    """ Calculates MSE, SSIM and correlation between 
    the input image and template image.

    Args:
        input_img (np.ndarray): input image or array
        template_img (np.ndarray): image to compare the input image with

    Returns:
        (list): list containing the MSE, SSIM and correlation values
    """
    mse_metric = tf.convert_to_tensor(MSE(template_img.numpy().flatten(),input_img.numpy().flatten()))
    ssim_metric = tf.image.ssim(input_img,template_img,
    max_val=max_val_ssim(input_img,template_img)).numpy()[0]
    correlation_metric = tf.convert_to_tensor(corr(input_img,template_img))
    return  [mse_metric, ssim_metric, correlation_metric]

