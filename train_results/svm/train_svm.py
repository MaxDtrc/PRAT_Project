import pyarrow.parquet as pa
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
import skimage
from ultralytics import YOLO
from skimage.feature import hog, local_binary_pattern
import io
import os
import sys
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle


# Packages = pyarrow pillow matplotlib numpy ultralytics datasets skimages sklearn

"""
Script used to train the SVM based on a manual classification of photos/other illustrations
"""

N_BINS = 15 # Number of bins used in the histograms

def load_photos():
    """
    Load all photos from train_svm/photo
    """
    photos = []
    images_folder = os.path.join("train_svm", "photo")
    for img_name in os.listdir(images_folder): 
        img_path = os.path.join(images_folder, img_name) # Path
        img = Image.open(img_path) # Image file
        photos.append(img)
    return photos

def load_illustrations():
    """
    Load all illustrations from train_svm/other
    """
    illustrations = []
    images_folder = os.path.join("train_svm", "other")
    for img_name in os.listdir(images_folder): 
        img_path = os.path.join(images_folder, img_name) # Path
        img = Image.open(img_path) # Image file
        illustrations.append(img)
    return illustrations

def compute_hog(images):
    """
    Compute the histogram of oriented gradients for each image
    """
    hog_features = []

    for image in images:
        image = np.array(image.resize((128, 128)))
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
        fd = np.histogram(fd, bins=N_BINS)[0]
        hog_features.append(fd / np.sum(fd)) # Normalization

    return np.array(hog_features)

def compute_lbp(images):
    """
    Compute the local binary patterns for each image
    """
    lbp_features = []

    for image in images:
        image = np.array(image)
        lbp = local_binary_pattern(image, P=8, R=5)
        lbp_hist = np.histogram(lbp, bins=N_BINS, density=True)[0]
        lbp_features.append(lbp_hist / np.sum(lbp_hist)) # Normalization

    return np.array(lbp_features)

def compute_gray(images):
    """
    Compute the gray scale histogram for each image
    """
    gray_features = []

    for image in images:
        image = np.array(image)
        gray = np.histogram(image.flatten(), bins=N_BINS)[0]
        gray_features.append(gray / np.sum(gray))

    return np.array(gray_features)

def compute_nb_val(images):
    """
    Compute the number of different values in each image (normalized between 0 and 1)
    """
    nbval = []

    for image in images:
        image = np.array(image)
        nbval.append([len(np.unique(image)) / 255])

    return np.array(nbval)

def compute_std(images):
    """
    Compute the standard deviation for each image (normalized between 0 and 1)
    """
    std = []

    for image in images:
        image = np.array(image)
        std.append([np.std(image) / 255])

    return np.array(std)

def merge_data(hog, lbp, gray, nb_val, std):
    """
    Merge all information in a single vector
    """

    # Transform the number of values to a normalized vector
    tab_nb_val = np.repeat(nb_val, N_BINS, axis=1)
    tab_nb_val = tab_nb_val / np.sum(tab_nb_val, axis=1)[:, None]

    # Transform the standard deviation to a normalized vector
    tab_std = np.repeat(std, N_BINS, axis=1)
    tab_std = tab_std / np.sum(tab_std, axis=1)[:, None]

    # Concatenate all data
    merged_data = np.concatenate([lbp, gray, hog, nb_val, std], axis=1)

    return merged_data

def compute_features(images):
    """
    Compute and merge all features for each image
    """
    hog = compute_hog(images)
    lbp = compute_lbp(images)
    gray = compute_gray(images)
    nb_val = compute_nb_val(images)
    std = compute_std(images)
    return merge_data(hog, lbp, gray, nb_val, std)


def train_svm():
    """
    Train an SVM based on 5 features
    """
    
    print("Loading images ...")
    photos = load_photos()
    illustrations = load_illustrations()

    print("Computing features ...")
    data_photo = compute_features(photos)
    data_illustrations = compute_features(illustrations)

    print("Creating labels")
    labels_photos = np.ones(shape=(len(photos), 1))
    labels_illustrations = np.zeros(shape=(len(illustrations), 1))

    # Stack data and labels
    data = np.vstack((np.hstack((data_photo, labels_photos)), np.hstack((data_illustrations, labels_illustrations))))
    
    # Lists for accuracies and trained SVM
    accs = []
    svms = []

    # Train 300 different SVM with different shuffles
    print("Training SVM ...")
    for _ in range(300):
        np.random.shuffle(data)
        clf = LinearSVC(C=100, max_iter=10000)

        perc = 80
        partition = int(len(photos) + len(illustrations) * perc / 100)

        # Create the sets
        X_train, X_test = data[:partition,:-1], data[partition:,:-1]
        Y_train, Y_test = data[:partition,-1:].ravel(), data[partition:,-1:].ravel()

        # Fit
        clf.fit(X_train, Y_train)

        # Compute accuracy
        Y_pred = clf.predict(X_test)
        accs.append(accuracy_score(Y_test, Y_pred))

        # Save the model
        svms.append(clf)
        

    print("Accuracy :", round(np.max(accs), 2))

    # Save the best svm
    with open("svm.pkl", "wb") as f:
        pickle.dump(svms[np.argmax(accs)], f)

if __name__ == "__main__":
    train_svm()