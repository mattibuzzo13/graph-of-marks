# Install pycocotools if not already installed
# pip install pycocotools

from pycocotools.coco import COCO
import os

# Define paths to the annotation file and image directory
data_dir = 'coco'
ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
img_dir = os.path.join(data_dir, 'images', 'train2017')

# Initialize the COCO API for instance annotations
coco = COCO(ann_file)

# Get all image IDs
img_ids = coco.getImgIds()

# Load information of a random image
img_info = coco.loadImgs(img_ids[0])[0]

# Print image information
print(f"Image ID: {img_info['id']}")
print(f"File Name: {img_info['file_name']}")
print(f"Image Height: {img_info['height']}")
print(f"Image Width: {img_info['width']}")

# Load and display the image using matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_path = os.path.join(img_dir, img_info['file_name'])
img = mpimg.imread(img_path)
plt.imshow(img)
plt.axis('off')
plt.show()

# Load and display annotations for the image
ann_ids = coco.getAnnIds(imgIds=img_info['id'])
anns = coco.loadAnns(ann_ids)
coco.showAnns(anns)
