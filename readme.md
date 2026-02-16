# Prat Project (2025-2026) - Analysis of historical photographs

Retrieving the circulation of photographs is very important for historians. This requires finding similar
images in different image databases, identifying whether an image (or a part of it) has been published in
a newspaper, all this taking into account different degradations or printing techniques. The aim of this
project is to contribute to this field by analyzing pages of newspapers and finding the photos.

## Code Structure

extract_data.py: main script to extract all images from pages placed in the folder "scans"

simple_classification: script containing the functions for the first classification system, not used in the final project

svm.py: script containing functions to use the pre-trained SVM

train/

    svm/

        eval_svm.py: script used to evaluate the performances of the SVM on a test set

        train_svm.py: script used to train the SVM, exported in svm.pkl

    yolov11/

        finlam: results of the training on the small FINLAM dataset

        finlam_la_liberte: results of the training on the large FINLAM dataset
        
    train_model.py: script used to train YOLOv11 on FINLAM


## Installation

- You need an installation of python (tested with python 3.12.3)

- Install the following packages : pip install pillow ultralytics pickle numpy scikit-image scikit-learn


## Usage

- Place all images to analyze in the folder "scans"

- Run the script extract_data.py (with "python3 extract_data.py")

- All extracted images will be placed in the folder "results". "photo" contains the photos, and "other" contains all other types of illustrations.

## Results

This model is mainly designed for pages from the newspaper "Le Petit Parisien". It can also works with pages from other newspaper, but there may be a loss of accuracy in the classification photos/others.