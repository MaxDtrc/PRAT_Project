from PIL import Image
from ultralytics import YOLO
import os
from datetime import datetime
import pickle
from svm import classify

# Packages = pyarrow pillow matplotlib numpy ultralytics datasets

def rename_scans():
    """
    Rename all images in 'scans' with a unique id
    """
    i = 0
    for file in os.listdir('scans'):
        os.rename(os.path.join('scans', file), os.path.join('scans', str(i) + '.jpeg'))
        i += 1

def main():
    # Load the model
    model = YOLO("yolo_finlam_0.pt")

    # Load the svm
    with open('svm.pkl', 'rb') as f:
        clf = pickle.load(f) # Loading the svm

    # Get the date
    date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    folder = os.path.join("results", date)
    if os.path.exists(folder):
        print(f"An execution already exists for this date (date)")
        exit()
    
    # Create the results folders
    os.mkdir(os.path.join('results', date))
    folder_photo = os.path.join(folder, 'photo')
    os.mkdir(folder_photo)
    folder_other = os.path.join(folder, 'other')
    os.mkdir(folder_other)

    # Stats
    nb_photos = 0
    nb_other = 0

    # Inference for each image
    all_images = os.listdir("scans")
    all_images.sort()
    
    for img_name in all_images: 
        print('Extracting from', img_name, '...')

        img_path = os.path.join("scans", img_name) # Image path

        img = Image.open(img_path) # Open image file
        results = model(img_path, verbose=False) # Prediction of the bounding boxes

        for result in results:
            xyxy = result.boxes.xyxy # Coordinates
            classes = result.boxes.cls.int() # Classes
            conf = result.boxes.conf #Lvl of confidence

            xyxy = xyxy[classes == 2] # We keep only the bounding boxes of the class "illustration"

            for (i, box) in enumerate(xyxy):

                if conf[i] >= 0.8: # Threshold on the level of confidence (to keep only images)
                    img_crop = img.crop(box.numpy()) # Croping the illustration

                    file_name = img_name.replace('.jpeg', '') + "_" + str(i) + ".jpeg"

                    # Classification image/photo
                    cls = classify(clf, img_crop)

                    if cls == 0: # Other
                        img_crop.save(os.path.join(folder_other, file_name))
                        nb_other += 1
                    elif cls == 1: # Photo
                        img_crop.save(os.path.join(folder_photo, file_name))
                        nb_photos += 1

    print(f'\nDone !\n{nb_photos} photos and {nb_other} other illustrations extracted.\nAll files have been saved to results/{date}.')
   
if __name__ == "__main__":
    main()