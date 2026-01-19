# Prat Project (2025) - Analysis of historical photographs

- Goal : extracting images from newspapers scans (mainly designed for pages from 'Le Petit Parisien')


## Installation
You need an installation of python (tested with python 3.12.3)

Install the following packages : pip install pillow ultralytics pickle numpy scikit-image scikit-learn


## How to use
- Place all images to analyze in the folder "scans"

- Run "python3 extract_data.py"

- All extracted images will be placed in the folder "results". "photo" contains the photos, and "other" contains all other types of illustrations.

## Results

The global accuracy is around 95%. It means that some photos/illustrations will be misclassified.