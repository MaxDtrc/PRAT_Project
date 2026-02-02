# Prat Project (2025-2026) - Analysis of historical photographs

Retrieving the circulation of photographs is very important for historians. This requires finding similar
images in different image databases, identifying whether an image (or a part of it) has been published in
a newspaper, all this taking into account different degradations or printing techniques. The aim of this
project is to contribute to this field by analyzing pages of newspapers and finding the photos.


## Installation

- You need an installation of python (tested with python 3.12.3)

- Install the following packages : pip install pillow ultralytics pickle numpy scikit-image scikit-learn


## Usage

- Place all images to analyze in the folder "scans"

- Run the script extract_data.py (with "python3 extract_data.py")

- All extracted images will be placed in the folder "results". "photo" contains the photos, and "other" contains all other types of illustrations.

## Results

This model is mainly designed for pages from the newspaper "Le Petit Parisien". It can also works with pages from other newspaper, but there may be a loss of accuracy in the classification photos/others.