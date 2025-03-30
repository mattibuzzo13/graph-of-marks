import json
import os

# Define paths
data_dir = 'refcoco'
ann_file = os.path.join(data_dir, 'annotations', 'instances.json')
img_dir = os.path.join(data_dir, 'images', 'train2014')

# Load annotations
with open(ann_file, 'r') as f:
    refcoco_data = json.load(f)

# Example: Accessing the first annotation
first_annotation = refcoco_data['annotations'][0]
image_id = first_annotation['image_id']
bbox = first_annotation['bbox']
referring_expression = first_annotation['caption']

# Load the corresponding image
image_filename = f'COCO_train2014_{image_id:012d}.jpg'
image_path = os.path.join(img_dir, image_filename)

print(f"Image Path: {image_path}")
print(f"Bounding Box: {bbox}")
print(f"Referring Expression: {referring_expression}")
