import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import LinearSVC


# Packages = pyarrow pillow matplotlib numpy ultralytics datasets skimages sklearn

"""
Classify an image in one of the two classes : "photo" and "other"
"""

N_BINS = 15 # Number of bins used in the histograms

def compute_hog(image):
    """
    Compute the histogram of oriented gradients for an image
    """
    image = np.array(image.resize((128, 128)))
    fd = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
    fd = np.histogram(fd, bins=N_BINS)[0]
    return fd / np.sum(fd) # Normalization

def compute_lbp(image):
    """
    Compute the local binary patterns for an image
    """
    image = np.array(image)
    lbp = local_binary_pattern(image, P=8, R=5)
    lbp_hist = np.histogram(lbp, bins=N_BINS, density=True)[0]
    return lbp_hist / np.sum(lbp_hist) # Normalization

def compute_gray(image):
    """
    Compute the gray scale histogram for an image
    """
    image = np.array(image)
    gray = np.histogram(image.flatten(), bins=N_BINS)[0]
    return gray / np.sum(gray)

def compute_nb_val(image):
    """
    Compute the number of different values in the image (normalized between 0 and 1)
    """
    image = np.array(image)
    return np.array([len(np.unique(image)) / 255])

def compute_std(image):
    """
    Compute the standard deviation of the image (normalized between 0 and 1)
    """
    image = np.array(image)
    return np.array([np.std(image) / 255])

def merge_data(hog, lbp, gray, nb_val, std):
    """
    Merge all information in a single vector
    """

    # Transform the number of values to a normalized vector
    tab_nb_val = np.repeat(nb_val, N_BINS)
    tab_nb_val = tab_nb_val / np.sum(tab_nb_val)

    # Transform the standard deviation to a normalized vector
    tab_std = np.repeat(std, N_BINS)
    tab_std = tab_std / np.sum(tab_std)

    # Concatenate all data
    merged_data = np.concatenate([lbp, gray, hog, nb_val, std])

    return merged_data

def compute_features(image):
    """
    Compute and merge all features for an image
    """

    hog = compute_hog(image)
    lbp = compute_lbp(image)
    gray = compute_gray(image)
    nb_val = compute_nb_val(image)
    std = compute_std(image)
    return merge_data(hog, lbp, gray, nb_val, std)

def classify(svm, image):
    """
    Return the classification for a given image
    """
    
    x = compute_features(image)
    pred = svm.predict([x])[0]

    return pred 