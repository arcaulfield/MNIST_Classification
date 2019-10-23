import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_pkl_file(pkl_file_path: str):
    """
    Loads the pkl file into a 3D numpy array (num_samples, img_width, img_height)
    :param pkl_file_path: The file path of th pkl file
    :return: The numpy representation of all images
    """
    if not os.path.isfile(pkl_file_path):
        raise Exception("The pkl file " + pkl_file_path + " you are trying to load does not exist.")

    return pd.read_pickle(pkl_file_path).astype(np.uint8)


def save_ndarray(file_path: str, images: np.ndarray):
    """
    Saves the numpy array as a pkl file
    :param file_path: File path to save to
    :param images: Images as a numpy array
    """
    if not os.path.isdir(os.path.dirname(file_path)):
        raise Exception("The directory in which you want to save the file " + file_path + " does not exist.")

    np.save(file_path, images)


def load_ndarray(file_path: str):
    """
    Loads a numpy array from file
    :param file_path: File path of numpy array
    :return: The numpy array
    """
    if not os.path.isfile(file_path):
        raise Exception("The numpy array file " + file_path + " you are trying to load does not exist.")

    return np.load(file_path).astype(np.uint8)


def load_training_labels(file_path: str):
    """
    Loads the csv file containing the true labels to the training set
    :param file_path: The file path of the file containing true labels of training data
    :return: A numpy array containing all training labels
    """
    if not os.path.isfile(file_path):
        raise Exception("The training labels file " + file_path + " you are trying to load does not exist.")

    df = pd.read_csv(file_path)
    return np.array(df['Label'])


def show_image(image: np.ndarray, title: str = "", grayscale: bool = True):
    """
    Shows an image represented as a np array
    :param image: numpy array representing image
    :param title: Title of image to be displayed
    :param grayscale: True if the image is grayscale
    """
    plt.title(title)
    if grayscale:
        plt.imshow(image, cmap=plt.get_cmap('gray'))
    else:
        plt.imshow(image)
    plt.show()
