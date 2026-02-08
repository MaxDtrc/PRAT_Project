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


# Packages = pyarrow pillow matplotlib numpy ultralytics datasets

N_BINS = 15

def load_photos():
    photos = []
    images_folder = os.path.join("extracted_illustrations_portraits", "photo")
    for img_name in os.listdir(images_folder): 
        img_path = os.path.join(images_folder, img_name) # Path
        img = Image.open(img_path) # Image file
        photos.append(img)
    return photos

def load_illustrations():
    illustrations = []
    images_folder = os.path.join("extracted_illustrations_portraits", "other")
    for img_name in os.listdir(images_folder): 
        img_path = os.path.join(images_folder, img_name) # Path
        img = Image.open(img_path) # Image file
        illustrations.append(img)
    return illustrations

def compute_hog(images):
    hog_features = None

    for image in images:

        image = np.array(image.resize((128, 128)))
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
        fd = np.histogram(fd, bins=N_BINS)[0]

        if hog_features is None:
            hog_features = fd / np.sum(fd)
        else:
            hog_features += fd / np.sum(fd)

    return hog_features / len(images)

def compute_lbp(images):
    lbp_features = None

    for image in images:
        image = np.array(image)
        lbp = local_binary_pattern(image, P=8, R=5)
        lbp_hist = np.histogram(lbp, bins=N_BINS, density=True)[0]
        
        if lbp_features is None:
            lbp_features = lbp_hist / np.sum(lbp_hist)
        else:
            lbp_features += lbp_hist / np.sum(lbp_hist)

    return lbp_features / len(images)

def compute_hist(images):
    hist_features = None

    for image in images:
        image = np.array(image)
        hist = np.histogram(image.flatten(), bins=N_BINS)[0]
        
        if hist_features is None:
            hist_features = hist / np.sum(hist)
        else:
            hist_features += hist / np.sum(hist)


    return hist_features / len(images)

def compute_mean2(images):
    mean = 0.0

    for image in images:
        image = np.array(image)
        mean += np.mean(image)

    return mean / len(images)

def compute_std(images):
    std = 0.0

    for image in images:
        image = np.array(image)
        std += np.std(image)

    return std / len(images)

def compute_mean(images):
    nbval = 0.0

    for image in images:
        image = np.array(image)
        nbval += len(np.unique(image))

    return nbval / len(images)

def merge_data(hog, lbp, gray, mean, std):
    mean = mean / 255
    tab_mean = np.ones(N_BINS) * mean
    tab_mean = tab_mean / np.sum(tab_mean)

    std = std / 255
    tab_std = np.ones(N_BINS) * std
    tab_std = tab_std / np.sum(tab_std)

    merged_data = np.concatenate([gray, lbp, hog, tab_std, tab_mean])

    return merged_data 

def predict(images, 
            photos_hog, 
            illus_hog, 
            photos_lbp, 
            illus_lbp, 
            photos_gray, 
            illus_gray, 
            photos_mean, 
            illus_mean, 
            photos_std, 
            illus_std):
    
    pred = []
    for image in images:
        img_hog = compute_hog([image])
        img_lbp = compute_lbp([image])
        img_gray = compute_hist([image])
        img_mean = compute_mean([image])
        img_std = compute_std([image])

        score = 0

        score += 1 if np.linalg.norm(img_hog - photos_hog) < np.linalg.norm(img_hog - illus_hog) else -1
        score += 1 if np.linalg.norm(img_lbp - photos_lbp) < np.linalg.norm(img_lbp - illus_lbp) else -1
        score += 1 if np.linalg.norm(img_gray - photos_gray) < np.linalg.norm(img_gray - illus_gray) else -1
        score += 1 if abs(img_mean - photos_mean) < abs(img_mean - illus_mean) else -1
        score += 1 if abs(img_std - photos_std) < abs(img_std - illus_std) else -1

        pred.append(1 if score >= 0 else 0)

    return np.array(pred)

def pred_merged(images, data_photos, data_illus):
    pred = []
    for image in images:
        img_hog = compute_hog([image])
        img_lbp = compute_lbp([image])
        img_gray = compute_hist([image])
        img_mean = compute_mean([image])
        img_std = compute_std([image])

        merged_data = merge_data(img_hog, img_lbp, img_gray, img_mean, img_std)
        score = 0

        score += 1 if np.linalg.norm(merged_data - data_photos) < np.linalg.norm(merged_data - data_illus) * 0.9 else -1

        pred.append(1 if score >= 0 else 0)

    return np.array(pred)


def main():
    print("Loading images ...")
    photos = load_photos()
    illustrations = load_illustrations()

    print("Computing HOG ...")
    photos_hog = compute_hog(photos)
    illus_hog = compute_hog(illustrations)

    print("Computing LBP ...")
    photos_lbp = compute_lbp(photos)
    illus_lbp = compute_lbp(illustrations)

    print("Computing Color Histogram")
    photos_gray = compute_hist(photos)
    illus_gray = compute_hist(illustrations)

    print("Computing mean ...")
    photos_mean = compute_mean(photos)
    illus_mean = compute_mean(illustrations)

    print("Computing std ...")
    photos_std = compute_std(photos)
    illus_std = compute_std(illustrations)

    merged_data_photo = merge_data(photos_hog, photos_lbp, photos_gray, photos_mean, photos_std)
    merged_data_illus = merge_data(illus_hog, illus_lbp, illus_gray, illus_mean, illus_std)

    labels_photos = np.ones(shape=(len(photos), 1))
    labels_illustrations = np.zeros(shape=(len(illustrations), 1))

    pred = predict(photos, photos_hog, illus_hog, photos_lbp, illus_lbp, photos_gray, illus_gray, photos_mean, illus_mean, photos_std, illus_std)
    acc = np.mean(np.where(pred == labels_photos, 1, 0))
    print("Accuracy photos:", acc)

    pred = pred_merged(photos, merged_data_photo, merged_data_illus)
    acc = np.mean(np.where(pred == labels_photos, 1, 0))
    print("Accuracy photos - merged:", acc)

    for i in range(len(photos)):
        if(pred[i] == 0):
            plt.imshow(photos[i], cmap='gray')
            plt.show()

    pred = predict(illustrations, photos_hog, illus_hog, photos_lbp, illus_lbp, photos_gray, illus_gray, photos_mean, illus_mean, photos_std, illus_std)
    acc = np.mean(np.where(pred == labels_illustrations, 1, 0))
    print("Accuracy other:", acc)

    pred = pred_merged(illustrations, merged_data_photo, merged_data_illus)
    acc = np.mean(np.where(pred == labels_illustrations, 1, 0))
    print("Accuracy other - merged:", acc)

    for i in range(len(illustrations)):
        if(pred[i] == 1):
            plt.imshow(illustrations[i], cmap='gray')
            plt.show()



if __name__ == "__main__":
    main()