from datasets import load_dataset
import numpy as np
from ultralytics import YOLO
import os
import sys

# Packages = pyarrow pillow matplotlib numpy ultralytics datasets

"""
Script used to train YOLOv11 on the FINLAM dataset
"""

model = 0 # Model to use

# Paths
HUGGING_FACE_PATH = ["teklia/newspapers-finlam", "teklia/newspapers-finlam-la-liberte"]
DATASET_PATH = ["Newspapers-finlam/data", "Newspapers-finlam-la-liberte/data"]
EXPORT_FOLDER_IMAGES = ["datasets/YOLO_FINLAM/images/", "datasets/YOLO_FINLAM_LA_LIBERTE/images/"]
EXPORT_FOLDER_LABELS = ["datasets/YOLO_FINLAM/labels/", "datasets/YOLO_FINLAM_LA_LIBERTE/labels/"]

EXTRACT = True # Extract the datasets 
SAVE_IMAGES = True # Save images to EXPORT_FOLDER_IMAGES 
SAVE_LABELS = True # Create the label files in EXPORT_FOLDER_LABELS

def export_images():
    """
    Export and format images/labels from the dataset
    """

    # Load the dataset
    dataset = load_dataset(HUGGING_FACE_PATH[model], cache_dir="./huggingface/")

    for BASE in ["train", "test", "val"]: # Extract images from each set
        print(BASE + " set ...")
        table = dataset[BASE]

        for i in range(len(table["page_image"])):
            img_bytes = table[i]["page_image"]
            img_name = table[i]["page_arkindex_id"] + ".jpg"
            img_coord = table[i]["zone_polygons"]
            img_label = table[i]["zone_classes"]

            if SAVE_IMAGES: # Export images
                img = img_bytes

                path = os.path.join(EXPORT_FOLDER_IMAGES[model], BASE)
                if not os.path.exists(path):
                    os.makedirs(path)
            
                img.save(os.path.join(path, str(img_name)))

            if SAVE_LABELS: # Export labels
                coord = np.array(img_coord)
                lab = np.array(img_label)

                labels_str = ""

                for (k, c) in enumerate(coord):
                    if(len(c) == 5):
                        c = c[:4] # Rectangular regions

                    if(len(c) == 4): # Rectangles

                        # Get coordinates
                        id = lab[k]
                        x_center = (np.min(c[:, 0]) + np.max(c[:, 0])) / 2
                        y_center = (np.min(c[:, 1]) + np.max(c[:, 1])) / 2
                        width = (np.max(c[:, 0]) - np.min(c[:, 0]))
                        height = (np.max(c[:, 1]) - np.min(c[:, 1]))

                        # Normalization
                        if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                            x_center = x_center/100.0
                            y_center = y_center/100.0
                            width = width/100.0
                            height = height/100.0

                        line = str(id) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)
                        labels_str += (line + '\n')

                labels_str = labels_str[:-1] # Remove the last '\n'
                
                # Export the file containing the labels for each region
                path = os.path.join(EXPORT_FOLDER_LABELS[model], BASE)
                if not os.path.exists(path):
                    os.makedirs(path)

                with open(os.path.join(path, str(img_name).replace(".jpg", ".txt")), 'w') as file:
                    file.write(labels_str)


def main():
    # Select the model to use based on the argument (0 = FINLAM, 1 = FINLAM - La Liberté)
    args = sys.argv[1:]
    global model
    if(len(args) > 0):
        model = int(args[0])

    # Define the path for the dataset
    path = "datasets/YOLO_FINLAM" + ("_LA_LIBERTE" if model else "")

    # Extract images from the dataset
    if(EXTRACT and not os.path.exists(EXPORT_FOLDER_IMAGES[model])):
        print("Extracting images ...")
        export_images()

        # Copy the .yaml file
        os.system("cp data_" + str(model) + ".yaml " + path + "/data.yaml")
        
        print("Done !")


    print("Training model ...")
    model = YOLO("yolo11n.pt") # Load the pre-trained model
    results = model.train(data=path + "/data.yaml", epochs=100, imgsz=640) # Train on FINLAM

    print("Done !")

if __name__ == "__main__":
    main()